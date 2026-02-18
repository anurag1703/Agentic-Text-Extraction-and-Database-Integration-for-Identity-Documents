"""
preprocessor_agent.py

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
_boot_logger = logging.getLogger("PreprocessorAgentBootstrap")
if not _boot_logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[PreprocessorAgentBootstrap] %(asctime)s %(levelname)s - %(message)s"))
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


# Import the project's preprocessing tool: try tools.document_preprocessor then document_preprocessor
try:
    from tools.document_preprocessor import process_pdf_file, PreprocessorConfig, DocumentPreprocessorError  # type: ignore
except Exception:
    try:
        from document_preprocessor import process_pdf_file, PreprocessorConfig, DocumentPreprocessorError  # type: ignore
    except Exception as e:
        raise ImportError(
            "Could not import required symbols from document_preprocessor.py. "
            "Make sure document_preprocessor.py exists and exports process_pdf_file, PreprocessorConfig, DocumentPreprocessorError."
        ) from e

# Module logger
logger = logging.getLogger("PreprocessorAgent")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[PreprocessorAgent] %(asctime)s %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# ---------------------------
# Local registry for tool functions (deferred registration)
# ---------------------------
_GLOBAL_TOOL_REGISTRY: List[Dict[str, Any]] = []

def register_function(func=None, *, name: Optional[str] = None, description: Optional[str] = None):
    """
    Safe decorator: collects function metadata into a local registry.
    Actual Autogen registration happens later in create_preprocessor_agent().
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
# Agent and config dataclass
# ---------------------------
@dataclass
class PreprocessorAgentConfig:
    poppler_path: Optional[str] = None
    dpi: Optional[int] = None
    max_pages: Optional[int] = None
    timeout_seconds: Optional[int] = None
    verbose: bool = True
    max_retries: int = 2
    retry_delay_s: float = 0.5
    preview_limit_pages: int = 5


# ---------------------------
# Utility helpers
# ---------------------------
def _sanitize_page_meta(page: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    meta["page_index"] = int(page.get("page_index", 0))
    # width/height if present else try infer
    w = page.get("width")
    h = page.get("height")
    if w is None:
        arr = page.get("image_array")
        try:
            w = int(arr.shape[1]) if arr is not None else None
        except Exception:
            w = None
    if h is None:
        arr = page.get("image_array")
        try:
            h = int(arr.shape[0]) if arr is not None else None
        except Exception:
            h = None
    meta["width"] = w
    meta["height"] = h

    q = page.get("quality", {}) or {}
    meta["quality"] = {k: v for k, v in (q.items() if isinstance(q, dict) else []) if k in ("blur", "brightness", "contrast", "score")}

    meta["rotation_applied"] = bool(page.get("rotation_applied", False))
    meta["crop_info"] = page.get("crop_info")
    preview = page.get("preview_bytes")
    if isinstance(preview, (bytes, bytearray)):
        meta["preview_bytes_len"] = len(preview)
    return meta

def _build_preprocessor_config(agent_cfg: PreprocessorAgentConfig, overrides: Optional[Dict[str, Any]]) -> PreprocessorConfig:
    pcfg = PreprocessorConfig()
    if agent_cfg.poppler_path:
        try:
            setattr(pcfg, "poppler_path", agent_cfg.poppler_path)
        except Exception:
            logger.debug("Ignoring poppler_path override")
    if agent_cfg.dpi is not None:
        try:
            setattr(pcfg, "dpi", int(agent_cfg.dpi))
        except Exception:
            logger.debug("Ignoring dpi override")
    if agent_cfg.max_pages is not None:
        try:
            setattr(pcfg, "max_pages", int(agent_cfg.max_pages))
        except Exception:
            logger.debug("Ignoring max_pages override")
    if agent_cfg.timeout_seconds is not None:
        try:
            setattr(pcfg, "timeout_seconds", int(agent_cfg.timeout_seconds))
        except Exception:
            logger.debug("Ignoring timeout_seconds override")
    if overrides:
        for k, v in overrides.items():
            if hasattr(pcfg, k):
                try:
                    setattr(pcfg, k, v)
                except Exception:
                    logger.debug("Failed to apply override %s: %s", k, v)
    return pcfg


# ---------------------------
# Tool functions (decorated)
# ---------------------------
@register_function(name="preprocess_pdf", description="Preprocess a PDF path into cleaned pages and metadata.")
def preprocess_pdf(pdf_path: str, config_overrides: Optional[Dict[str, Any]] = None,
                   agent_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    start = time.perf_counter()
    logger.info("preprocess_pdf called for: %s", pdf_path)

    if not isinstance(pdf_path, str) or not pdf_path:
        raise ValueError("pdf_path must be a non-empty string")
    if not os.path.exists(pdf_path):
        return {"status": "failed", "error": f"file_not_found: {pdf_path}", "pdf_path": pdf_path}

    ag_cfg = PreprocessorAgentConfig()
    if agent_config:
        for k, v in agent_config.items():
            if hasattr(ag_cfg, k):
                try:
                    setattr(ag_cfg, k, v)
                except Exception:
                    logger.debug("Ignoring agent_config override %s=%s", k, v)

    try:
        pcfg = _build_preprocessor_config(ag_cfg, config_overrides)
    except Exception as e:
        logger.exception("Failed to build PreprocessorConfig")
        return {"status": "failed", "error": f"config_build_error: {e}", "pdf_path": pdf_path}

    attempts = 0
    last_err_msg = None
    while attempts <= ag_cfg.max_retries:
        attempts += 1
        try:
            pages = process_pdf_file(pdf_path, pcfg)
            elapsed = time.perf_counter() - start

            # Keep full pages for downstream (include 'image_path' and 'preview_bytes' if present)
            full_pages = pages

            # For metrics/dashboard we still create a sanitized view (no raw image arrays)
            sanitized_pages = [_sanitize_page_meta(p) for p in pages]

            widths = [p["width"] for p in sanitized_pages if p.get("width")]
            heights = [p["height"] for p in sanitized_pages if p.get("height")]
            metrics: Dict[str, Any] = {"num_pages": len(sanitized_pages), "warnings": []}
            if widths:
                metrics["avg_width"] = sum(widths) / len(widths)
            if heights:
                metrics["avg_height"] = sum(heights) / len(heights)

            low_blur = []
            for p in sanitized_pages:
                q = p.get("quality", {}) or {}
                blur = q.get("blur")
                if blur is not None:
                    try:
                        if float(blur) < getattr(pcfg, "blur_threshold", 120.0):
                            low_blur.append(p["page_index"])
                    except Exception:
                        pass
            if low_blur:
                metrics["warnings"].append({"low_blur_pages": low_blur})

            result = {
                "status": "success",
                "pdf_path": pdf_path,
                "pages": full_pages,               # <-- full page dicts including preview_bytes / image_path
                "pages_summary": sanitized_pages,  # sanitized view for quick display
                "metrics": metrics,
                "timing": {"elapsed_s": round(elapsed, 3)}
            }
            logger.info("preprocess_pdf succeeded: %s pages, elapsed=%.3f s", metrics["num_pages"], elapsed)
            return result

        except DocumentPreprocessorError as dpe:
            last_err_msg = str(dpe)
            logger.warning("DocumentPreprocessorError attempt %d/%d: %s", attempts, ag_cfg.max_retries, dpe)
            if attempts > ag_cfg.max_retries:
                logger.exception("Max retries reached in preprocess_pdf")
                return {"status": "failed", "error": f"preprocessor_error: {dpe}", "pdf_path": pdf_path}
            time.sleep(ag_cfg.retry_delay_s)
            continue
        except Exception as ex:
            logger.exception("Unexpected exception in preprocess_pdf: %s", ex)
            return {"status": "failed", "error": f"unexpected_error: {ex}", "pdf_path": pdf_path}

    return {"status": "failed", "error": f"failed_after_retries: {last_err_msg}", "pdf_path": pdf_path}


@register_function(name="preprocessor_health_check", description="Check runtime dependencies for preprocessor (poppler, pillow, cv2, pdf2image).")
def preprocessor_health_check(agent_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    msgs: List[str] = []
    ok = True
    ppath = None
    try:
        default_ppath = getattr(PreprocessorConfig(), "poppler_path", None)
    except Exception:
        default_ppath = None
    if agent_config and isinstance(agent_config, dict) and agent_config.get("poppler_path"):
        ppath = agent_config.get("poppler_path")
    elif default_ppath:
        ppath = default_ppath

    if ppath:
        if os.path.exists(ppath):
            msgs.append(f"poppler_path_ok: {ppath}")
        else:
            msgs.append(f"poppler_path_missing: {ppath}")
            ok = False
    else:
        msgs.append("poppler_path_not_set")

    try:
        from PIL import Image  # type: ignore
        msgs.append("PIL_available")
    except Exception:
        msgs.append("PIL_missing")
        ok = False

    try:
        import cv2  # type: ignore
        msgs.append(f"cv2_available_version_{getattr(cv2, '__version__', 'unknown')}")
    except Exception:
        msgs.append("cv2_missing")
        ok = False

    try:
        import pdf2image  # type: ignore
        msgs.append("pdf2image_available")
    except Exception:
        msgs.append("pdf2image_missing")
        ok = False

    return {"ok": ok, "messages": msgs}


@register_function(name="summarize_page_metrics", description="Create summary from preprocess output (low_quality_pages, big_pages, aspect_ratios).")
def summarize_page_metrics(preproc_output: Dict[str, Any]) -> Dict[str, Any]:
    pages = preproc_output.get("pages", []) if isinstance(preproc_output, dict) else []
    low_quality = []
    big_pages = []
    aspect_ratios = []
    for p in pages:
        idx = p.get("page_index")
        q = p.get("quality", {}) or {}
        blur = q.get("blur")
        if blur is not None:
            try:
                if float(blur) < 80:
                    low_quality.append(idx)
            except Exception:
                pass
        w = p.get("width")
        h = p.get("height")
        if w and h:
            try:
                if max(w, h) > 2000:
                    big_pages.append(idx)
                aspect_ratios.append(round(float(w) / float(h), 3))
            except Exception:
                pass
    return {"num_pages": len(pages), "low_quality_pages": low_quality, "big_pages": big_pages, "aspect_ratios": aspect_ratios}


# ---------------------------
# Agent factory: attach functions from local registry
# ---------------------------
def create_preprocessor_agent(agent_name: str = "PreprocessorAgent", agent_cfg: Optional[PreprocessorAgentConfig] = None) -> AssistantAgent:
    agent_cfg = agent_cfg or PreprocessorAgentConfig()
    system_prompt = (
        "PreprocessorAgent: provides deterministic PDF preprocessing tools. "
        "Exposes functions: preprocess_pdf, preprocessor_health_check, summarize_page_metrics. "
        "Tool-only agent: no LLM reasoning."
    )

    agent = AssistantAgent(name=agent_name, role="Preprocesses PDFs into clean image pages", system_prompt=system_prompt)

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

    logger.info("PreprocessorAgent created and functions registered.")
    return agent


# ---------------------------
# Local basic test harness (calls the tools directly)
# ---------------------------
def basic_test(sample_pdf_path: str) -> None:
    print("=== PreprocessorAgent basic_test ===")
    if not os.path.exists(sample_pdf_path):
        print("Sample PDF not found:", sample_pdf_path)
        return

    hc = preprocessor_health_check()
    print("Health:", json.dumps(hc, indent=2))

    res = preprocess_pdf(sample_pdf_path, config_overrides=None, agent_config=None)
    print("Preprocess result (summary):")
    print(json.dumps({
        "status": res.get("status"),
        "pdf_path": res.get("pdf_path"),
        "num_pages": len(res.get("pages", [])),
        "metrics": res.get("metrics")
    }, indent=2))

    if res.get("status") == "success" and res.get("pages"):
        print("First page metadata:", json.dumps(res["pages"][0], indent=2))


# ---------------------------
# CLI entrypoint for quick tests
# ---------------------------
if __name__ == "__main__":
    sample = os.environ.get("SAMPLE_PREPROCESS_PDF")
    if sample:
        basic_test(sample)
    else:
        print("To run a basic test: set SAMPLE_PREPROCESS_PDF to a PDF path and run this module.")
