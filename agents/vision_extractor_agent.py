"""
vision_extractor_agent.py

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
_boot_logger = logging.getLogger("VisionExtractorAgentBootstrap")
if not _boot_logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[VisionExtractorAgentBootstrap] %(asctime)s %(levelname)s - %(message)s"))
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

# Import the project's vision_extractor tool: try tools.vision_extractor then vision_extractor
try:
    from tools.vision_extractor import VisionExtractor, ExtractorConfig, VisionExtractorError  # type: ignore
except Exception:
    try:
        from vision_extractor import VisionExtractor, ExtractorConfig, VisionExtractorError  # type: ignore
    except Exception as e:
        raise ImportError(
            "Could not import required symbols from vision_extractor.py. "
            "Make sure vision_extractor.py exists and exports VisionExtractor, ExtractorConfig, VisionExtractorError."
        ) from e

# Module logger
logger = logging.getLogger("VisionExtractorAgent")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[VisionExtractorAgent] %(asctime)s %(levelname)s - %(message)s"))
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
class VisionExtractorAgentConfig:
    ollama_endpoint: Optional[str] = None
    model_name: Optional[str] = None
    timeout_seconds: Optional[int] = None
    max_retries: int = 2
    retry_delay_s: float = 0.8
    batch_size: int = 2
    confidence_threshold: float = 0.7
    enable_logging: bool = True


# ---------------------------
# Utility helpers
# ---------------------------
def _sanitize_extraction_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize extractor output into a compact, safe shape for the orchestrator/validator.
    Ensures each document contains an 'extracted_data' key (the format expected by the validator).
    Handles both extractor shape with nested 'extracted_data' and legacy top-level 'fields'/'confidences'.
    """
    out: Dict[str, Any] = {}
    out["status"] = raw.get("status", "unknown")
    out["documents"] = []

    for d in raw.get("documents", []) if isinstance(raw.get("documents", []), list) else []:
        # If the extractor already returned a structured 'extracted_data', keep it (deep copy safe).
        extracted_data = d.get("extracted_data")
        if extracted_data and isinstance(extracted_data, dict):
            # Normalize keys we care about
            fields = extracted_data.get("fields", {})
            # accept multiple confidence key names
            confidences = extracted_data.get("confidence_scores", extracted_data.get("confidences", {}))
            completeness = extracted_data.get("completeness_score", extracted_data.get("completeness"))
            validation_errors = extracted_data.get("validation_errors", [])
            # Keep the full extracted_data as-is for validator
            normalized_extracted = dict(extracted_data)
        else:
            # Build a canonical extracted_data object from legacy/alternate keys
            fields = d.get("fields", {}) or {}
            confidences = d.get("confidences", {}) or d.get("field_confidences", {}) or {}
            completeness = d.get("completeness") or d.get("completeness_score")
            validation_errors = d.get("validation_errors") or []

            normalized_extracted = {
                "fields": fields,
                "confidence_scores": confidences,
                "completeness_score": completeness,
                "validation_errors": validation_errors
            }

        # Build sanitized document that ALWAYS includes 'extracted_data'
        doc = {
            "document_id": d.get("document_id"),
            "document_type": d.get("document_type"),
            # top-level convenience (legacy codepaths may still read these)
            "fields": fields,
            "confidences": confidences,
            # include canonical extracted_data for validator & downstream tools
            "extracted_data": normalized_extracted,
            # keep metadata/debug fields if present
            "completeness_score": completeness,
            "validation_errors": validation_errors,
            "processing_metadata": d.get("processing_metadata"),
            "raw": d.get("raw") or d.get("original_extracted_json")
        }

        out["documents"].append(doc)

    out["timing"] = raw.get("timing")
    out["debug"] = raw.get("debug")
    return out


def _prepare_extractor_config(agent_cfg: VisionExtractorAgentConfig, overrides: Optional[Dict[str, Any]]) -> ExtractorConfig:
    cfg = ExtractorConfig()
    if agent_cfg.ollama_endpoint:
        try:
            setattr(cfg, "ollama_endpoint", agent_cfg.ollama_endpoint)
        except Exception:
            logger.debug("Ignoring ollama_endpoint override")
    if agent_cfg.model_name:
        try:
            setattr(cfg, "model_name", agent_cfg.model_name)
        except Exception:
            logger.debug("Ignoring model_name override")
    if agent_cfg.timeout_seconds is not None:
        try:
            setattr(cfg, "timeout_seconds", int(agent_cfg.timeout_seconds))
        except Exception:
            logger.debug("Ignoring timeout_seconds override")
    if agent_cfg.batch_size is not None:
        try:
            setattr(cfg, "batch_size", int(agent_cfg.batch_size))
        except Exception:
            logger.debug("Ignoring batch_size override")
    if agent_cfg.confidence_threshold is not None:
        try:
            setattr(cfg, "confidence_threshold", float(agent_cfg.confidence_threshold))
        except Exception:
            logger.debug("Ignoring confidence_threshold override")

        # apply any explicit overrides verbatim (allow pass-through of any ExtractorConfig property)
    if overrides:
        for k, v in overrides.items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, v)
                except Exception:
                    logger.debug("Failed to apply override %s: %s", k, v)
            else:
                # If the underlying ExtractorConfig supports dynamic attributes or you want to allow new keys,
                # you can still set them cautiously:
                try:
                    setattr(cfg, k, v)
                except Exception:
                    logger.debug("Ignored unknown override %s", k)

    return cfg


# ---------------------------
# Tool functions (decorated)
# ---------------------------
@register_function(name="extract_documents", description="Run LLM/OCR-based extraction on classified pages/documents.")
def extract_documents(classifier_output: Dict[str, Any], agent_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    classifier_output: expected structure returned by ClassifierAgent.classify_pages or orchestrator:
      {
        "pages": [ {page_index, image_array, metadata, predicted_class, ...}, ... ],
        "documents": [ aggregated doc groups ... ],
        "status": "success"
      }

    Returns a sanitized dict with extracted fields, confidences, timing, status, and optional debug.
    """
    start = time.perf_counter()
    logger.info("extract_documents called. classifier_output pages=%d", len(classifier_output.get("pages", [])) if classifier_output else 0)

    ag_cfg = VisionExtractorAgentConfig()
    if agent_config and isinstance(agent_config, dict):
        for k, v in agent_config.items():
            if hasattr(ag_cfg, k):
                try:
                    setattr(ag_cfg, k, v)
                except Exception:
                    logger.debug("Ignoring agent_config override %s=%s", k, v)

    extractor_cfg = _prepare_extractor_config(ag_cfg, None)

    attempts = 0
    last_exc = None
    while attempts <= ag_cfg.max_retries:
        attempts += 1
        try:
            extractor = VisionExtractor(extractor_cfg)
            # call extract_documents method if exists
            if hasattr(extractor, "extract_documents"):
                raw = extractor.extract_documents(classifier_output)
                logger.debug("Raw extractor output sample (first doc keys): %s", list(raw.get("documents", [{}])[0].keys()) if raw.get("documents") else "no-docs")
            elif hasattr(extractor, "extract"):
                raw = extractor.extract(classifier_output)
            else:
                raise VisionExtractorError("No suitable extract method on VisionExtractor instance")

            elapsed = time.perf_counter() - start
            sanitized = _sanitize_extraction_output(raw)
            # debug: log how many docs contain extracted_data
            try:
                num_docs = len(sanitized.get("documents", []))
                num_with_extracted = sum(1 for doc in sanitized.get("documents", []) if doc.get("extracted_data"))
                logger.info("Sanitized extractor output: documents=%d, with_extracted_data=%d", num_docs, num_with_extracted)
            except Exception:
                logger.debug("Sanitized output logging failed", exc_info=True)

            # normalize status: if extractor returned an explicit 'success' boolean, respect it
            if isinstance(raw, dict) and raw.get("success") is True:
                sanitized["status"] = "success"
            else:
                # if raw already had 'status' use it, otherwise fallback to 'unknown' or 'failed' when errors present
                if "status" not in sanitized or sanitized.get("status") in (None, "unknown"):
                    if isinstance(raw, dict) and raw.get("errors"):
                        sanitized["status"] = "failed"
                    else:
                        sanitized["status"] = "unknown"
            sanitized["timing"] = {"elapsed_s": round(elapsed, 3)}
            logger.info("extract_documents completed in %.3f s", elapsed)
            # IMPORTANT: return sanitized top-level structure (must contain 'documents' key) so downstream validator can consume it
            return sanitized

        except VisionExtractorError as vee:
            last_exc = vee
            logger.warning("VisionExtractorError attempt %d/%d: %s", attempts, ag_cfg.max_retries, vee)
            if attempts > ag_cfg.max_retries:
                return {"status": "failed", "error": str(vee)}
            time.sleep(ag_cfg.retry_delay_s)
        except Exception as ex:
            last_exc = ex
            logger.exception("Unexpected error in extract_documents: %s", ex)
            return {"status": "failed", "error": str(ex)}

    return {"status": "failed", "error": str(last_exc)}


@register_function(name="extractor_health_check", description="Check extractor dependencies (Ollama/requests/aiohttp, cv2) and model endpoint.")
def extractor_health_check(agent_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    msgs: List[str] = []
    ok = True

    cfg = _prepare_extractor_config(VisionExtractorAgentConfig(), agent_config or {})

    # Check requests / aiohttp
    try:
        import aiohttp  # type: ignore
        msgs.append("aiohttp_available")
    except Exception:
        msgs.append("aiohttp_missing")

    try:
        import requests  # type: ignore
        msgs.append("requests_available")
    except Exception:
        msgs.append("requests_missing")

    # cv2
    try:
        import cv2  # type: ignore
        msgs.append(f"cv2_available_version_{getattr(cv2, '__version__', 'unknown')}")
    except Exception:
        msgs.append("cv2_missing")
        ok = False

    # Ollama endpoint check (if endpoint provided)
    if getattr(cfg, "ollama_endpoint", None):
        try:
            import requests  # type: ignore
            resp = requests.get(cfg.ollama_endpoint, timeout=3)
            msgs.append(f"ollama_ping_status_{resp.status_code}")
            if resp.status_code != 200:
                ok = False
        except Exception as e:
            msgs.append(f"ollama_unreachable: {e}")
            ok = False
    else:
        msgs.append("ollama_endpoint_not_set")

    # Model name presence
    if getattr(cfg, "model_name", None):
        msgs.append(f"model_name_set: {cfg.model_name}")
    else:
        msgs.append("model_name_not_set")

    return {"ok": ok, "messages": msgs}


@register_function(name="summarize_extraction", description="Summarize extractor results: which fields present, low confidence fields.")
def summarize_extraction(extraction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize extractor output:
      - present_fields per document
      - low_confidence_fields list
    """
    res = extraction_result.get("result") if isinstance(extraction_result, dict) else extraction_result
    docs = res.get("documents", []) if isinstance(res, dict) else []
    present_fields = {}
    low_conf = []
    for d in docs:
        doc_id = d.get("document_id") or "unknown"
        fields = d.get("fields") or {}
        confs = d.get("confidences") or {}
        present_fields[doc_id] = list(fields.keys())
        for k, v in confs.items() if isinstance(confs, dict) else []:
            try:
                if float(v) < getattr(extraction_result, "confidence_threshold", 0.6):
                    low_conf.append({"document_id": doc_id, "field": k, "confidence": v})
            except Exception:
                pass
    return {"present_fields": present_fields, "low_confidence_fields": low_conf, "num_documents": len(docs)}


# ---------------------------
# Agent factory: attach functions from local registry
# ---------------------------
def create_vision_extractor_agent(agent_name: str = "VisionExtractorAgent", agent_cfg: Optional[VisionExtractorAgentConfig] = None) -> AssistantAgent:
    agent_cfg = agent_cfg or VisionExtractorAgentConfig()
    system_prompt = (
        "VisionExtractorAgent: extracts structured fields from classified document pages. "
        "Exposes functions: extract_documents, extractor_health_check, summarize_extraction. "
        "Tool-only agent: no LLM reasoning."
    )

    agent = AssistantAgent(name=agent_name, role="Extracts fields from document images", system_prompt=system_prompt)

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

    logger.info("VisionExtractorAgent created and functions registered.")
    return agent


# ---------------------------
# Local basic test harness (calls the tools directly)
# ---------------------------
def basic_test(sample_preproc_json: Optional[str] = None) -> None:
    """
    Basic test:
      - If sample_preproc_json path provided, loads classifier-like input from JSON (list/dict).
      - Otherwise constructs a minimal synthetic classifier_output with no image arrays (may cause extractor to return errors depending on tool).
    """
    print("=== VisionExtractorAgent basic_test ===")
    if sample_preproc_json:
        if not os.path.exists(sample_preproc_json):
            print("Sample JSON not found:", sample_preproc_json)
            return
        with open(sample_preproc_json, "r", encoding="utf-8") as fh:
            classifier_output = json.load(fh)
    else:
        # minimal synthetic input (likely insufficient if extractor expects images or base64)
        classifier_output = {
            "pages": [
                {"page_index": 0, "predicted_class": "pan_card_front", "image_array": None, "metadata": {}}
            ],
            "documents": []
        }

    print("Running extractor_health_check() ...")
    print(json.dumps(extractor_health_check(), indent=2))
    print("Running extract_documents() ... (this may fail if no images provided)")
    res = extract_documents(classifier_output)
    print("Extraction result (summary):")
    try:
        print(json.dumps({
            "status": res.get("status"),
            "num_documents": len(res.get("result", {}).get("documents", [])) if res.get("result") else 0,
            "timing": res.get("result", {}).get("timing") if res.get("result") else None
        }, indent=2))
    except Exception:
        print(res)


# ---------------------------
# CLI entrypoint for quick tests
# ---------------------------
if __name__ == "__main__":
    sample = os.environ.get("SAMPLE_EXTRACTOR_INPUT_JSON")
    if sample:
        basic_test(sample)
    else:
        print("To run a basic test: set SAMPLE_EXTRACTOR_INPUT_JSON=/path/to/classifier_output.json or call basic_test() programmatically.")
