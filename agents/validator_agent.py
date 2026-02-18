"""
validator_agent.py

"""

from __future__ import annotations

import os
import sys
import time
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Ensure project root on sys.path so sibling imports resolve
HERE = os.path.dirname(os.path.abspath(__file__))          # agents/
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))   # project root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Bootstrap logger
_boot_logger = logging.getLogger("ValidatorAgentBootstrap")
if not _boot_logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[ValidatorAgentBootstrap] %(asctime)s %(levelname)s - %(message)s"))
    _boot_logger.addHandler(ch)
_boot_logger.setLevel(logging.INFO)

# Try to import Autogen AssistantAgent; fallback to a stub if unavailable.
_AUTOGEN_AVAILABLE = False
AssistantAgent = None
try:
    from autogen import AssistantAgent  # type: ignore
    _AUTOGEN_AVAILABLE = True
    _boot_logger.info("Autogen AssistantAgent available.")
except Exception:
    _boot_logger.info("Autogen AssistantAgent not available; using fallback stub.")

    class AssistantAgent:
        """Minimal stub for local testing when Autogen is not installed."""
        def __init__(self, name: str, role: str, system_prompt: Optional[str] = None, llm_config: Optional[Any] = None):
            self.name = name
            self.role = role
            self.system_prompt = system_prompt
            self.llm_config = llm_config
            self._tools: Dict[str, Any] = {}

        def register(self, fn):
            self._tools[fn.__name__] = fn

        def register_tool(self, fn, name: Optional[str] = None, description: Optional[str] = None):
            key = name or fn.__name__
            self._tools[key] = fn

        def add_function(self, fn, name: Optional[str] = None, description: Optional[str] = None):
            self.register_tool(fn, name=name, description=description)

        def info(self):
            return {"name": self.name, "role": self.role, "tools": list(self._tools.keys())}


# Import the project's validator tool: try tools.data_validator then data_validator
try:
    from tools.data_validator import validate_batch, ValidatorConfig  # type: ignore
except Exception:
    try:
        from data_validator import validate_batch, ValidatorConfig  # type: ignore
    except Exception as e:
        raise ImportError(
            "Could not import required symbols from data_validator.py. "
            "Make sure data_validator.py exists and exports validate_batch and ValidatorConfig."
        ) from e

# Module logger
logger = logging.getLogger("ValidatorAgent")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[ValidatorAgent] %(asctime)s %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# ---------------------------
# Local registry for tool functions (deferred registration)
# ---------------------------
_GLOBAL_TOOL_REGISTRY: List[Dict[str, Any]] = []

def register_function(func=None, *, name: Optional[str] = None, description: Optional[str] = None):
    """
    Safe decorator: collect function metadata. Actual Autogen registration is deferred.
    """
    def _decorator(f):
        entry = {
            "fn": f,
            "name": name or f.__name__,
            "description": description or (f.__doc__ or "").strip()[:200]
        }
        _GLOBAL_TOOL_REGISTRY.append(entry)
        return f

    if func is None:
        return _decorator
    else:
        return _decorator(func)


# ---------------------------
# Agent configuration dataclass
# ---------------------------
@dataclass
class ValidatorAgentConfig:
    # Map to ValidatorConfig overrides if needed
    min_confidence_threshold: float = 0.6
    require_all_fields: bool = False
    max_retries: int = 1
    retry_delay_s: float = 0.25
    verbose: bool = True


# ---------------------------
# Utilities
# ---------------------------
def _sanitize_validation_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep key validation outputs but remove heavy artifacts.
    Expected raw structure usually contains per-document validated_fields, statuses, etc.
    """
    out: Dict[str, Any] = {}
    out["status"] = raw.get("status", "unknown")
    out["documents"] = []
    docs = raw.get("documents") or raw.get("validated_documents") or []
    for d in docs:
        doc_summary = {
            "document_id": d.get("document_id"),
            "document_type": d.get("document_type"),
            "validation_status": d.get("validation_status") or d.get("status"),
            "field_statuses": d.get("field_statuses") or d.get("validated_fields") or {},
            "validated_fields": d.get("validated_fields_json") or d.get("validated_fields") or {},
        }
        out["documents"].append(doc_summary)
    out["timing"] = raw.get("timing")
    out["errors"] = raw.get("errors")
    return out


# ---------------------------
# Tool functions
# ---------------------------
@register_function(name="validate_documents", description="Validate extracted fields from documents and mark statuses.")
def validate_documents(extractor_output: Dict[str, Any], agent_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Runs the project's validate_batch(...) function on extractor output (or classifier+extractor combined).
    Returns sanitized result suitable for DB storage / orchestration.

    Expected extractor_output structure:
      {"documents": [ { "document_id":..., "document_type":..., "fields": {...}, "confidences": {...} }, ... ] }

    Output:
      {"status":"success"/"failed", "result": <sanitized>, "timing": {...}, "error": Optional[str>}
    """
    start = time.perf_counter()
    logger.info("validate_documents called. docs=%d", len(extractor_output.get("documents", [])) if extractor_output else 0)

    ag_cfg = ValidatorAgentConfig()
    if agent_config and isinstance(agent_config, dict):
        for k, v in agent_config.items():
            if hasattr(ag_cfg, k):
                try:
                    setattr(ag_cfg, k, v)
                except Exception:
                    logger.debug("Ignoring agent_config override %s=%s", k, v)

    # Build ValidatorConfig for the underlying tool if available
    try:
        vcfg = ValidatorConfig()  # type: ignore
        # try to map a few common overrides
        try:
            if hasattr(vcfg, "confidence_threshold"):
                setattr(vcfg, "confidence_threshold", float(ag_cfg.min_confidence_threshold))
        except Exception:
            pass
        # apply explicit agent_config overrides if present
        if agent_config:
            for k, v in agent_config.items():
                if hasattr(vcfg, k):
                    try:
                        setattr(vcfg, k, v)
                    except Exception:
                        logger.debug("Failed to apply vcfg override %s=%s", k, v)
    except Exception:
        # If ValidatorConfig is not present or cannot be instantiated, set vcfg to None
        vcfg = None

    attempts = 0
    last_exc = None
    while attempts <= ag_cfg.max_retries:
        attempts += 1
        try:
            # Call validate_batch; it may expect list of documents or wrapped dict
            try:
                raw = validate_batch(extractor_output, vcfg)  # type: ignore
            except TypeError:
                # fallback: pass documents list
                docs = extractor_output.get("documents", [])
                raw = validate_batch(docs, vcfg)  # type: ignore

            elapsed = time.perf_counter() - start
            sanitized = _sanitize_validation_output(raw)
            sanitized["timing"] = {"elapsed_s": round(elapsed, 3)}
            logger.info("validate_documents completed in %.3f s", elapsed)
            return {"status": "success", "result": sanitized}
        except Exception as ex:
            last_exc = ex
            logger.exception("Validation failed on attempt %d/%d: %s", attempts, ag_cfg.max_retries, ex)
            if attempts > ag_cfg.max_retries:
                return {"status": "failed", "error": str(ex)}
            time.sleep(ag_cfg.retry_delay_s)

    return {"status": "failed", "error": str(last_exc)}


@register_function(name="validator_health_check", description="Check validation tool dependencies and sanity.")
def validator_health_check(agent_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Basic health check for validation tool and environment.
    """
    msgs: List[str] = []
    ok = True
    # Check ValidatorConfig presence
    try:
        _ = ValidatorConfig()
        msgs.append("ValidatorConfig_instantiable")
    except Exception:
        msgs.append("ValidatorConfig_unavailable")
        ok = False

    # Check data_validator functions presence by a dry-run with safe empty input
    try:
        sample = {"documents": []}
        try:
            _ = validate_batch(sample, None)
        except Exception:
            # some implementations may raise on empty input; that's ok as long as function exists
            msgs.append("validate_batch_callable")
        else:
            msgs.append("validate_batch_ok")
    except Exception:
        msgs.append("validate_batch_missing_or_broken")
        ok = False

    return {"ok": ok, "messages": msgs}


@register_function(name="summarize_validation", description="Summarize a validation result: counts per status and fields needing review.")
def summarize_validation(validation_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize validated documents:
      - counts: APPROVED, REVIEW_REQUIRED, REJECTED (or variants)
      - list of documents with missing or low-confidence fields
    """
    res = validation_result.get("result") if isinstance(validation_result, dict) else validation_result
    docs = res.get("documents", []) if isinstance(res, dict) else []
    counts: Dict[str, int] = {}
    needs_review: List[Dict[str, Any]] = []

    for d in docs:
        status = d.get("validation_status") or d.get("validation", {}).get("status") or "unknown"
        counts[status] = counts.get(status, 0) + 1
        field_statuses = d.get("field_statuses") or {}
        # detect missing/needs review fields
        missing = [k for k, v in (field_statuses.items() if isinstance(field_statuses, dict) else []) if v in ("REVIEW_REQUIRED", "MISSING", "INVALID", "REJECTED")]
        if missing:
            needs_review.append({"document_id": d.get("document_id"), "missing_fields": missing})

    return {"counts": counts, "needs_review": needs_review, "num_documents": len(docs)}


# ---------------------------
# Agent factory: attach functions from local registry
# ---------------------------
def create_validator_agent(agent_name: str = "ValidatorAgent", agent_cfg: Optional[ValidatorAgentConfig] = None) -> AssistantAgent:
    agent_cfg = agent_cfg or ValidatorAgentConfig()
    system_prompt = (
        "ValidatorAgent: validates extracted fields using deterministic rules and returns validation_status per document. "
        "Exposes functions: validate_documents, validator_health_check, summarize_validation. Tool-only agent."
    )

    agent = AssistantAgent(name=agent_name, role="Validates extracted document fields", system_prompt=system_prompt)

    # Attach functions collected in _GLOBAL_TOOL_REGISTRY
    for entry in _GLOBAL_TOOL_REGISTRY:
        fn = entry["fn"]
        try:
            if hasattr(agent, "register_tool"):
                agent.register_tool(fn, name=entry["name"], description=entry["description"])
            elif hasattr(agent, "add_function"):
                agent.add_function(fn, name=entry["name"], description=entry["description"])
            elif hasattr(agent, "register"):
                agent.register(fn)
            else:
                setattr(agent, fn.__name__, fn)
        except Exception:
            logger.exception("Failed to attach tool %s to agent; attaching as attribute instead", entry["name"])
            setattr(agent, fn.__name__, fn)

    logger.info("ValidatorAgent created and functions registered.")
    return agent


# ---------------------------
# Local basic test harness (calls the tools directly)
# ---------------------------
def basic_test(sample_input_json: Optional[str] = None) -> None:
    """
    Basic test:
      - If sample_input_json is provided, load extractor output JSON from that path.
      - Else, create a synthetic extractor_output with one document and run validate_documents.
    """
    print("=== ValidatorAgent basic_test ===")
    if sample_input_json:
        if not os.path.exists(sample_input_json):
            print("Sample JSON not found:", sample_input_json)
            return
        with open(sample_input_json, "r", encoding="utf-8") as fh:
            extractor_output = json.load(fh)
    else:
        # Synthetic minimal extractor output
        extractor_output = {
            "documents": [
                {
                    "document_id": "doc_test_1",
                    "document_type": "pan",
                    "fields": {"pan_number": "ABCDE1234F", "name": "TEST NAME", "date_of_birth": "01/01/1990"},
                    "confidences": {"pan_number": 0.95, "name": 0.9, "date_of_birth": 0.98}
                }
            ]
        }

    print("Running validator_health_check() ...")
    print(json.dumps(validator_health_check(), indent=2))
    print("Running validate_documents() ...")
    res = validate_documents(extractor_output)
    print("Validation result (summary):")
    try:
        print(json.dumps({
            "status": res.get("status"),
            "num_documents": len(res.get("result", {}).get("documents", [])) if res.get("result") else 0,
            "timing": res.get("result", {}).get("timing") if res.get("result") else None
        }, indent=2))
    except Exception:
        print(res)


# ---------------------------
# CLI entrypoint
# ---------------------------
if __name__ == "__main__":
    sample = os.environ.get("SAMPLE_VALIDATION_INPUT_JSON")
    if sample:
        basic_test(sample)
    else:
        print("To run a basic test: set SAMPLE_VALIDATION_INPUT_JSON=/path/to/extractor_output.json or call basic_test() programmatically.")
