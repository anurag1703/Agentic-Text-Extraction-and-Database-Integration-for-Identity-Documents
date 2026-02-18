"""
orchestrator_agent.py

Orchestrator Agent (tool-only) that coordinates the full pipeline:
  Preprocess -> Classify -> Extract -> Validate -> Store

Usage:
    python -m agents.orchestrator_agent
    or
    SAMPLE_ORCH_PDF=/path/to/doc.pdf python agents/orchestrator_agent.py

Exposes (registers):
 - run_pipeline(pdf_path: str, agent_config: Optional[dict]) -> dict
 - orchestrator_health_check(agent_config: Optional[dict]) -> dict
 - get_run_report(trace_id: str) -> dict

Design:
 - No LLM usage. Deterministic, config-driven pipeline runner.
 - Defers Autogen registration (local _GLOBAL_TOOL_REGISTRY).
 - Uses JSON manifests saved to outputs/runs/<trace_id>.json (optional).
"""

from __future__ import annotations

import os
import sys
import time
import json
import uuid
import logging
import base64
import numpy as np
import cv2
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root on sys.path
HERE = os.path.dirname(os.path.abspath(__file__))          # agents/
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))   # project root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# bootstrap logger
_boot_logger = logging.getLogger("OrchestratorAgentBootstrap")
if not _boot_logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[OrchestratorAgentBootstrap] %(asctime)s %(levelname)s - %(message)s"))
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

# Import agent tool functions (they exist in agents/<name> modules)
# They should expose module-level functions with same names as registered earlier.
try:
    from agents.preprocessor_agent import preprocess_pdf, preprocessor_health_check  # type: ignore
except Exception:
    try:
        from preprocessor_agent import preprocess_pdf, preprocessor_health_check  # type: ignore
    except Exception as e:
        raise ImportError("Could not import preprocess_pdf / preprocessor_health_check from preprocessor_agent.py") from e

try:
    from agents.classifier_agent import classify_pages, classifier_health_check  # type: ignore
except Exception:
    try:
        from classifier_agent import classify_pages, classifier_health_check  # type: ignore
    except Exception as e:
        raise ImportError("Could not import classify_pages / classifier_health_check from classifier_agent.py") from e

try:
    from agents.vision_extractor_agent import extract_documents, extractor_health_check  # type: ignore
except Exception:
    try:
        from vision_extractor_agent import extract_documents, extractor_health_check  # type: ignore
    except Exception as e:
        raise ImportError("Could not import extract_documents / extractor_health_check from vision_extractor_agent.py") from e

try:
    from agents.validator_agent import validate_documents, validator_health_check  # type: ignore
except Exception:
    try:
        from validator_agent import validate_documents, validator_health_check  # type: ignore
    except Exception as e:
        raise ImportError("Could not import validate_documents / validator_health_check from validator_agent.py") from e

try:
    from agents.database_agent import store_documents, fetch_document, query_documents, db_health_check  # type: ignore
except Exception:
    try:
        from agents.database_agent import store_documents, fetch_document, query_documents, db_health_check  # type: ignore
    except Exception as e:
        raise ImportError("Could not import DB functions from database_agent.py") from e

# Agent logger
logger = logging.getLogger("OrchestratorAgent")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[OrchestratorAgent] %(asctime)s %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# ---------------------------
# Local registry for deferred register_function
# ---------------------------
_GLOBAL_TOOL_REGISTRY: List[Dict[str, Any]] = []


def register_function(func=None, *, name: Optional[str] = None, description: Optional[str] = None):
    def _decorator(f):
        entry = {"fn": f, "name": name or f.__name__, "description": description or (f.__doc__ or "").strip()[:200]}
        _GLOBAL_TOOL_REGISTRY.append(entry)
        return f
    if func is None:
        return _decorator
    else:
        return _decorator(func)


# ---------------------------
# Config dataclass
# ---------------------------
@dataclass
class OrchestratorConfig:
    outputs_base: str = "outputs"
    runs_dir: str = "outputs/runs"
    save_manifest: bool = True
    preproc_overrides: Optional[Dict[str, Any]] = None
    classifier_overrides: Optional[Dict[str, Any]] = None
    extractor_overrides: Optional[Dict[str, Any]] = None
    validator_overrides: Optional[Dict[str, Any]] = None
    db_overrides: Optional[Dict[str, Any]] = None
    health_check_first: bool = False
    retry_backoff_s: float = 0.5
    max_retries_stage: int = 2
    timeout_stage_s: Optional[int] = None  # not enforced strictly but recorded
    verbose: bool = True


# ---------------------------
# Utilities
# ---------------------------
def _ensure_dirs(cfg: OrchestratorConfig):
    runs_dir = os.path.join(PROJECT_ROOT, cfg.runs_dir)
    os.makedirs(runs_dir, exist_ok=True)
    return runs_dir


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _gen_trace_id() -> str:
    return uuid.uuid4().hex


def _safe_write_manifest(runs_dir: str, trace_id: str, payload: Dict[str, Any]) -> None:
    try:
        path = os.path.join(runs_dir, f"{trace_id}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        logger.info("Saved run manifest: %s", path)
    except Exception as e:
        logger.exception("Failed saving manifest: %s", e)


def _attach_images_to_pages(pages: List[dict], preproc_result: dict) -> List[dict]:
    """
    Ensure each page dict contains:
      - 'image_array' (numpy HxWx3 uint8)
      - 'metadata' (dict)
    Accepts pages that may include 'preview_bytes' (bytes) or 'image_path'.
    """
    out_pages = []
    for i, p in enumerate(pages):
        page = dict(p)  # copy metadata
        # metadata field required by classifier
        meta = {k: v for k, v in page.items() if k not in ("image_array", "preview_bytes", "image_path", "preview_base64")}
        page["metadata"] = meta

        # Try to restore image_array from several fallbacks
        img = None

        # 1) Prefer image_path saved by preprocessor (PNG or JPEG on disk)
        if page.get("image_path"):
            try:
                img = cv2.imread(page["image_path"], cv2.IMREAD_COLOR)
            except Exception:
                img = None

        # 2) then preview_file / preview_path
        if img is None and page.get("preview_file"):
            try:
                img = cv2.imread(page["preview_file"], cv2.IMREAD_COLOR)
            except Exception:
                img = None

        # 3) preview_bytes (raw png/jpg bytes or base64 string) fallback
        if img is None:
            pb = page.get("preview_bytes") or page.get("preview") or page.get("preview_base64")
            if isinstance(pb, (bytes, bytearray)):
                arr = np.frombuffer(pb, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            elif isinstance(pb, str):
                try:
                    b = base64.b64decode(pb)
                    arr = np.frombuffer(b, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                except Exception:
                    img = None

        # If image restored, attach as image_array
        if img is not None:
            page["image_array"] = img
        # else leave image_array missing (classifier will fail and orchestrator will retry/stop)
        out_pages.append(page)
    return out_pages


# ---------------------------
# Orchestrator stage runner
# ---------------------------
@register_function(name="run_pipeline", description="Run full pipeline for a PDF: Preprocess->Classify->Extract->Validate->Store")
def run_pipeline(pdf_path: str, agent_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main orchestrator entrypoint.

    Returns a run_report dict:
      {
        "trace_id": str,
        "pdf_path": str,
        "document_id": Optional[str],
        "stages": {
          "preprocess": {...},
          "classify": {...},
          "extract": {...},
          "validate": {...},
          "store": {...}
        },
        "status": "success"|"failed",
        "errors": [...],
        "timing": {...}
      }
    """
    start_all = time.perf_counter()
    cfg = OrchestratorConfig()
    if agent_config and isinstance(agent_config, dict):
        # apply any allowed overrides
        for k, v in agent_config.items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, v)
                except Exception:
                    logger.debug("Ignoring orchestrator config override %s=%s", k, v)

    runs_dir = _ensure_dirs(cfg)
    trace_id = _gen_trace_id()
    run_report: Dict[str, Any] = {
        "trace_id": trace_id,
        "pdf_path": pdf_path,
        "start_time": _now_iso(),
        "stages": {},
        "errors": [],
        "status": "running"
    }

    # optional health checks
    if cfg.health_check_first:
        try:
            run_report["stages"]["health_checks"] = {
                "preprocessor": preprocessor_health_check(),
                "classifier": classifier_health_check(),
                "extractor": extractor_health_check(),
                "validator": validator_health_check(),
                "database": db_health_check()
            }
        except Exception as e:
            logger.exception("Health checks failed: %s", e)
            run_report["errors"].append(f"health_check_failed: {e}")
            run_report["status"] = "failed"
            run_report["end_time"] = _now_iso()
            if cfg.save_manifest:
                _safe_write_manifest(runs_dir, trace_id, run_report)
            return run_report

    # Stage helpers
    def _attempt(fn, *args, stage_name: str, max_retries: int = 2, **kwargs):
        attempts = 0
        last_err = None
        while attempts <= max_retries:
            attempts += 1
            t0 = time.perf_counter()
            try:
                res = fn(*args, **kwargs)
                # Immediately after res = fn(*args, **kwargs)
                # If the tool returns a dict with a 'status' field, consider non-success as a failure.
                if isinstance(res, dict) and res.get("status") in ("failed", "error", "fatal"):
                    # treat as failure and continue retry/backoff logic
                    raise RuntimeError(res.get("error") or res.get("status"))

                elapsed = time.perf_counter() - t0
                logger.info("Stage %s attempt %d/%d finished in %.3fs", stage_name, attempts, max_retries, elapsed)
                return {"success": True, "result": res, "elapsed_s": round(elapsed, 3), "attempts": attempts}
            except Exception as ex:
                last_err = ex
                logger.exception("Stage %s attempt %d/%d failed: %s", stage_name, attempts, max_retries, ex)
                time.sleep(cfg.retry_backoff_s * attempts)
                if attempts > max_retries:
                    return {"success": False, "error": str(last_err), "elapsed_s": None, "attempts": attempts}
        return {"success": False, "error": str(last_err), "elapsed_s": None, "attempts": attempts}

    # 1) Preprocess (ensure preprocessor returns images/preview bytes)
    logger.info("Orchestrator: starting preprocess for %s", pdf_path)

    # Build a local overrides dict to force image outputs from the preprocessor.
    # These keys must match the PreprocessorConfig fields you added in document_preprocessor.py.
    local_preproc_overrides = dict(cfg.preproc_overrides or {})
    # Force saving images and returning small preview bytes so classifier can load images.
    local_preproc_overrides.setdefault("save_images", True)
    # directory where preprocessor will save images (optional; will default inside preprocessor)
    local_preproc_overrides.setdefault("save_dir", os.path.join(os.getcwd(), "outputs", "preprocessed"))
    # request preview bytes attached to page dicts
    local_preproc_overrides.setdefault("return_preview_bytes", False)
    # preferred preview format and quality
    local_preproc_overrides.setdefault("preview_format", "png")
    local_preproc_overrides.setdefault("preview_quality", 85)

    # Call preprocess_pdf with our overrides
    preproc_attempt = _attempt(preprocess_pdf, pdf_path, local_preproc_overrides, stage_name="preprocess", max_retries=cfg.max_retries_stage)
    run_report["stages"]["preprocess"] = preproc_attempt
    if not preproc_attempt.get("success"):
        run_report["status"] = "failed"
        run_report["errors"].append({"stage": "preprocess", "error": preproc_attempt.get("error")})
        run_report["end_time"] = _now_iso()
        if cfg.save_manifest:
            _safe_write_manifest(runs_dir, trace_id, run_report)
        return run_report

    preproc_out = preproc_attempt["result"]

    # Build minimal pages container to pass forward.
    # If preprocessor returned sanitized pages without image arrays, prefer to pass the full pages if available.
    pages = preproc_out.get("pages", [])
    # Try to attach images / metadata so classifier receives expected shape
    pages_with_images = _attach_images_to_pages(pages, preproc_out)

    # if no images attached, try a fallback: if preprocessor supports saving images to disk, use those files
    num_with_images = sum(1 for p in pages_with_images if p.get("image_array") is not None)
    if num_with_images == 0:
        logger.warning("No images found in preprocessor output — classifier likely to fail. Consider running preprocessor with save_images=True or configuring preview_bytes")

    # 2) Classify
    logger.info("Orchestrator: starting classifier")
    # classifer expects list of pages; pass pages (if classifier needs image arrays, ensure preprocessor produced them or image paths)
    classify_attempt = _attempt(classify_pages, pages_with_images, cfg.classifier_overrides, stage_name="classify", max_retries=cfg.max_retries_stage)
    run_report["stages"]["classify"] = classify_attempt
    if not classify_attempt.get("success"):
        run_report["status"] = "failed"
        run_report["errors"].append({"stage": "classify", "error": classify_attempt.get("error")})
        run_report["end_time"] = _now_iso()
        if cfg.save_manifest:
            _safe_write_manifest(runs_dir, trace_id, run_report)
        return run_report

    classify_out = classify_attempt["result"]
    # classifier may return {"status":"success","result": { "pages":[...], "documents":[...] } }
    if isinstance(classify_out, dict) and "result" in classify_out:
        classifier_payload = classify_out["result"]
    else:
        classifier_payload = classify_out  # if classifier already returned the payload

    # Ensure extractor gets an object with either pages or documents:
    if not isinstance(classifier_payload, dict):
        classifier_payload = {"pages": classifier_payload}

    # Guarantee the pages key exists:
    classifier_payload.setdefault("pages", [])
    classifier_payload.setdefault("documents", [])

    # 3) Extract
    logger.info("Orchestrator: starting extractor")
    extract_input = classify_out
    extract_attempt = _attempt(extract_documents, classifier_payload, cfg.extractor_overrides, stage_name="extract", max_retries=cfg.max_retries_stage)
    run_report["stages"]["extract"] = extract_attempt
    if not extract_attempt.get("success"):
        run_report["status"] = "failed"
        run_report["errors"].append({"stage": "extract", "error": extract_attempt.get("error")})
        run_report["end_time"] = _now_iso()
        if cfg.save_manifest:
            _safe_write_manifest(runs_dir, trace_id, run_report)
        return run_report

    extract_out = extract_attempt["result"]

    # 4) Validate
    logger.info("Orchestrator: starting validator")
    validate_input = extract_out
    validate_attempt = _attempt(validate_documents, validate_input, cfg.validator_overrides, stage_name="validate", max_retries=cfg.max_retries_stage)
    run_report["stages"]["validate"] = validate_attempt
    if not validate_attempt.get("success"):
        run_report["status"] = "failed"
        run_report["errors"].append({"stage": "validate", "error": validate_attempt.get("error")})
        run_report["end_time"] = _now_iso()
        if cfg.save_manifest:
            _safe_write_manifest(runs_dir, trace_id, run_report)
        return run_report

    validate_out = validate_attempt["result"]
    if not (isinstance(validate_out, dict) and validate_out.get("status") == "success"):
        run_report["status"] = "failed"
        run_report["errors"].append({"stage": "validate", "error": validate_out})
        # save manifest and return early
        if cfg.save_manifest:
            _safe_write_manifest(runs_dir, trace_id, run_report)
        return run_report
    # else pass validate_out to store_documents


    # 5) Store
    logger.info("Orchestrator: starting store")
    # store expects validator-shaped payload; we pass validate_out (which should be sanitized)
    store_attempt = _attempt(store_documents, validate_out, cfg.db_overrides, stage_name="store", max_retries=cfg.max_retries_stage)
    run_report["stages"]["store"] = store_attempt
    if not store_attempt.get("success"):
        run_report["status"] = "failed"
        run_report["errors"].append({"stage": "store", "error": store_attempt.get("error")})
        run_report["end_time"] = _now_iso()
        if cfg.save_manifest:
            _safe_write_manifest(runs_dir, trace_id, run_report)
        return run_report

    store_out = store_attempt["result"]

    # finalize
    elapsed_all = time.perf_counter() - start_all
    run_report["status"] = "success"
    run_report["end_time"] = _now_iso()
    run_report["timing"] = {"elapsed_s": round(elapsed_all, 3)}
    run_report["stages"]["store"]["result"] = store_out

    # include selected sanitized outputs for auditing (no raw images)
    run_report["outputs"] = {
        "preprocess": {"metrics": preproc_out.get("metrics"), "num_pages": len(preproc_out.get("pages", []))},
        "classify_summary": classify_out.get("result") if isinstance(classify_out, dict) else classify_out,
        "extraction_summary": extract_out,
        "validation_summary": validate_out,
        "store_summary": store_out
    }

    if cfg.save_manifest:
        _safe_write_manifest(runs_dir, trace_id, run_report)

    return run_report


# ---------------------------
# Health check and report fetcher
# ---------------------------
@register_function(name="orchestrator_health_check", description="Run lightweight health checks for all agents.")
def orchestrator_health_check(agent_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        res = {
            "preprocessor": preprocessor_health_check(),
            "classifier": classifier_health_check(),
            "extractor": extractor_health_check(),
            "validator": validator_health_check(),
            "database": db_health_check()
        }
        ok = all(r.get("ok", True) if isinstance(r, dict) else True for r in res.values())
        return {"ok": ok, "details": res}
    except Exception as e:
        logger.exception("orchestrator_health_check error: %s", e)
        return {"ok": False, "error": str(e)}


@register_function(name="get_run_report", description="Load saved run manifest by trace_id.")
def get_run_report(trace_id: str) -> Dict[str, Any]:
    cfg = OrchestratorConfig()
    runs_dir = _ensure_dirs(cfg)
    path = os.path.join(PROJECT_ROOT, runs_dir, f"{trace_id}.json")
    if not os.path.exists(path):
        return {"status": "failed", "error": "manifest_not_found", "trace_id": trace_id}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return {"status": "success", "report": json.load(fh)}
    except Exception as e:
        logger.exception("get_run_report failed: %s", e)
        return {"status": "failed", "error": str(e)}


# ---------------------------
# Agent factory (attach functions from local registry)
# ---------------------------
def create_orchestrator_agent(agent_name: str = "OrchestratorAgent", agent_cfg: Optional[OrchestratorConfig] = None) -> AssistantAgent:
    agent_cfg = agent_cfg or OrchestratorConfig()
    system_prompt = "OrchestratorAgent: coordinates Preprocess->Classify->Extract->Validate->Store. Tool-only."

    agent = AssistantAgent(name=agent_name, role="Orchestrates document processing pipeline", system_prompt=system_prompt)

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

    logger.info("OrchestratorAgent created and functions registered.")
    return agent


# ---------------------------
# CLI / basic test harness
# ---------------------------
def basic_test(pdf_path: Optional[str] = None) -> None:
    print("=== OrchestratorAgent basic_test ===")
    sample = pdf_path or os.environ.get("SAMPLE_ORCH_PDF")
    if not sample:
        print("Provide SAMPLE_ORCH_PDF env var or call basic_test('/path/to/pdf')")
        return
    report = run_pipeline(sample)
    print("Run report status:", report.get("status"))
    print("Trace ID:", report.get("trace_id"))
    # write short summary
    print(json.dumps({
        "status": report.get("status"),
        "trace_id": report.get("trace_id"),
        "timing": report.get("timing"),
        "errors": report.get("errors")
    }, indent=2))


if __name__ == "__main__":
    sample = os.environ.get("SAMPLE_ORCH_PDF")
    if sample:
        # strip accidental quotes
        sample = sample.strip().strip('"').strip("'")
        basic_test(sample)
    else:
        print("To run: SAMPLE_ORCH_PDF=/path/to/doc.pdf python agents/orchestrator_agent.py")
