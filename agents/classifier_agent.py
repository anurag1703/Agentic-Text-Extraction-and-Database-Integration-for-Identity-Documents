"""
classifier_agent.py
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
_boot_logger = logging.getLogger("ClassifierAgentBootstrap")
if not _boot_logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[ClassifierAgentBootstrap] %(asctime)s %(levelname)s - %(message)s"))
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


# Import the project's classifier tool: try tools.document_classifier then document_classifier
try:
    from tools.document_classifier import DocumentClassifier, ClassifierConfig, DocumentClassifierError  # type: ignore
except Exception:
    try:
        from document_classifier import DocumentClassifier, ClassifierConfig, DocumentClassifierError  # type: ignore
    except Exception as e:
        raise ImportError(
            "Could not import required symbols from document_classifier.py. "
            "Make sure document_classifier.py exists and exports DocumentClassifier, ClassifierConfig, DocumentClassifierError."
        ) from e

# Module logger
logger = logging.getLogger("ClassifierAgent")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[ClassifierAgent] %(asctime)s %(levelname)s - %(message)s"))
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
class ClassifierAgentConfig:
    model_path: Optional[str] = None
    device: Optional[str] = None
    confidence_threshold: float = 0.70
    batch_size: int = 4
    max_retries: int = 1
    retry_delay_s: float = 0.2
    verbose: bool = True


# ---------------------------
# Helper utilities
# ---------------------------
def _sanitize_classification_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove large image arrays, tensors, or other non-serializable objects from classifier output.
    Keep document/page labels and confidences.
    """
    out: Dict[str, Any] = {}
    # typical fields: 'pages', 'documents', 'status', etc.
    out['status'] = raw.get('status')
    out['summary'] = raw.get('summary') or {}
    pages = raw.get('pages', [])
    sanitized_pages = []
    for p in pages:
        sp: Dict[str, Any] = {}
        sp['page_index'] = p.get('page_index')
        sp['predicted_class'] = p.get('predicted_class') or p.get('class')
        sp['confidence'] = float(p.get('confidence', 0.0)) if p.get('confidence') is not None else 0.0
        sp['bbox'] = p.get('bbox')
        sanitized_pages.append(sp)
    out['pages'] = sanitized_pages
    # documents aggregated (if present)
    docs = raw.get('documents', [])
    sanitized_docs = []
    for d in docs:
        sd = {
            'document_id': d.get('document_id'),
            'document_type': d.get('document_type'),
            'pages': d.get('pages'),
            'confidence': d.get('confidence')
        }
        sanitized_docs.append(sd)
    out['documents'] = sanitized_docs
    return out


# ---------------------------
# Tool functions
# ---------------------------
@register_function(name="classify_pages", description="Classify preprocessed pages into document types/sides.")
def classify_pages(preprocessed_pages: List[Dict[str, Any]], agent_config: Optional[Dict[str, Any]] = None
                   ) -> Dict[str, Any]:
    """
    Input:
      - preprocessed_pages: list of sanitized page metadata objects OR full page dicts from preprocessor (if image arrays included)
    Output:
      - a JSON-friendly dict with keys: status, pages (with predicted class/confidence), documents (aggregated), timing
    """
    start = time.perf_counter()
    logger.info("classify_pages called. pages=%d", len(preprocessed_pages) if preprocessed_pages else 0)

    # build agent config
    cfg = ClassifierAgentConfig()
    if agent_config and isinstance(agent_config, dict):
        for k, v in agent_config.items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, v)
                except Exception:
                    logger.debug("Ignoring agent_config override %s=%s", k, v)

    # instantiate classifier object with retry logic
    attempts = 0
    last_exc = None
    while attempts <= cfg.max_retries:
        attempts += 1
        try:
            model_cfg = ClassifierConfig()
            if cfg.model_path:
                try:
                    setattr(model_cfg, "model_path", cfg.model_path)
                except Exception:
                    pass
            if cfg.device:
                try:
                    setattr(model_cfg, "device", cfg.device)
                except Exception:
                    pass
            model_cfg.confidence_threshold = cfg.confidence_threshold
            model_cfg.batch_size = cfg.batch_size

            classifier = DocumentClassifier(model_cfg)
            # load model if needed
            try:
                classifier.load_model()
            except Exception as e:
                # load failure may be fatal
                logger.warning("Classifier.load_model failed: %s", e)

            # call classify_and_aggregate if available (your tool provides this)
            if hasattr(classifier, "classify_and_aggregate"):
                raw = classifier.classify_and_aggregate(preprocessed_pages)
            elif hasattr(classifier, "predict"):
                # fallback - prepare minimal batch and call predict
                raw_pred = classifier.predict(preprocessed_pages)  # if implemented
                # normalize into consistent structure
                raw = {"status": "ok", "pages": raw_pred}
            else:
                raise DocumentClassifierError("No suitable classification method found on DocumentClassifier instance")

            elapsed = time.perf_counter() - start
            sanitized = _sanitize_classification_output(raw)
            sanitized['timing'] = {"elapsed_s": round(elapsed, 3)}
            logger.info("classify_pages completed in %.3f s", elapsed)
            return {"status": "success", "result": sanitized}

        except DocumentClassifierError as dce:
            last_exc = dce
            logger.warning("DocumentClassifierError attempt %d/%d: %s", attempts, cfg.max_retries, dce)
            if attempts > cfg.max_retries:
                return {"status": "failed", "error": str(dce)}
            time.sleep(cfg.retry_delay_s)
        except Exception as ex:
            last_exc = ex
            logger.exception("Unexpected error in classify_pages: %s", ex)
            return {"status": "failed", "error": str(ex)}

    return {"status": "failed", "error": str(last_exc)}


@register_function(name="classifier_health_check", description="Check classifier dependencies and model readiness.")
def classifier_health_check(agent_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    msgs: List[str] = []
    ok = True
    # Check model file path if provided in config or default ClassifierConfig
    model_path = None
    try:
        default_model_path = getattr(ClassifierConfig(), "model_path", None)
    except Exception:
        default_model_path = None
    if agent_config and isinstance(agent_config, dict) and agent_config.get("model_path"):
        model_path = agent_config.get("model_path")
    elif default_model_path:
        model_path = default_model_path

    if model_path:
        if os.path.exists(model_path):
            msgs.append(f"model_path_ok: {model_path}")
        else:
            msgs.append(f"model_path_missing: {model_path}")
            ok = False
    else:
        msgs.append("model_path_not_set")

    # Torch availability check
    try:
        import torch  # type: ignore
        msgs.append(f"torch_available_cuda_{torch.cuda.is_available()}")
    except Exception:
        msgs.append("torch_missing")
        ok = False

    # YOLO/Ultralytics check
    try:
        from ultralytics import YOLO  # type: ignore
        msgs.append("ultralytics_yolo_available")
    except Exception:
        msgs.append("ultralytics_missing")
        # not fatal if classifier uses another backend, but warn

    return {"ok": ok, "messages": msgs}


@register_function(name="summarize_classification", description="Summarize classifier output (counts per class, low_confidence_pages).")
def summarize_classification(classify_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: output of classify_pages (sanitized)
    Output: summary dict with class counts and low-confidence pages
    """
    res = classify_result.get('result') if isinstance(classify_result, dict) else classify_result
    pages = res.get('pages', []) if isinstance(res, dict) else []
    counts: Dict[str, int] = {}
    low_conf_pages: List[int] = []
    threshold = classify_result.get('result', {}).get('summary', {}).get('confidence_threshold') if isinstance(classify_result, dict) else None
    if threshold is None:
        threshold = 0.5
    for p in pages:
        cls = p.get('predicted_class') or "unknown"
        counts[cls] = counts.get(cls, 0) + 1
        try:
            conf = float(p.get('confidence', 0.0))
            if conf < threshold:
                low_conf_pages.append(p.get('page_index'))
        except Exception:
            pass
    return {"class_counts": counts, "low_confidence_pages": low_conf_pages, "num_pages": len(pages)}


# ---------------------------
# Agent factory: attach functions from local registry
# ---------------------------
def create_classifier_agent(agent_name: str = "ClassifierAgent", agent_cfg: Optional[ClassifierAgentConfig] = None) -> AssistantAgent:
    agent_cfg = agent_cfg or ClassifierAgentConfig()
    system_prompt = (
        "ClassifierAgent: deterministic page classification tools. "
        "Exposes functions: classify_pages, classifier_health_check, summarize_classification. "
        "Tool-only agent: no LLM reasoning."
    )

    agent = AssistantAgent(name=agent_name, role="Classifies pages into document types", system_prompt=system_prompt)

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

    logger.info("ClassifierAgent created and functions registered.")
    return agent


# ---------------------------
# Local basic test harness (calls the tools directly)
# ---------------------------
def basic_test(load_pages_from_json: Optional[str] = None) -> None:
    """
    Basic test:
      - If load_pages_from_json is provided, it loads a JSON list of preprocessed pages.
      - Otherwise it builds a tiny synthetic page to test classification function wiring.
    """
    print("=== ClassifierAgent basic_test ===")
    if load_pages_from_json:
        if not os.path.exists(load_pages_from_json):
            print("Pages JSON not found:", load_pages_from_json)
            return
        with open(load_pages_from_json, "r", encoding="utf-8") as fh:
            pages = json.load(fh)
    else:
        # synthetic minimal page metadata - depending on classifier implementation it might require actual image arrays
        pages = [{"page_index": 0, "width": 1024, "height": 768, "image_array": None}]
    print("Calling classifier_health_check() ...")
    print(json.dumps(classifier_health_check(), indent=2))
    print("Calling classify_pages() ...")
    out = classify_pages(pages)
    print("Classify result (summary):")
    print(json.dumps({
        "status": out.get("status"),
        "num_pages": len(out.get("result", {}).get("pages", [])) if out.get("result") else 0,
        "timing": out.get("result", {}).get("timing") if out.get("result") else None
    }, indent=2))
    if out.get("status") == "success":
        first_page = out.get("result", {}).get("pages", [None])[0]
        print("First page prediction (sanitized):", json.dumps(first_page, indent=2))


# ---------------------------
# CLI entrypoint for quick tests
# ---------------------------
if __name__ == "__main__":
    sample_json = os.environ.get("SAMPLE_CLASSIFY_PAGES_JSON")
    if sample_json:
        basic_test(sample_json)
    else:
        print("To run a basic test: set SAMPLE_CLASSIFY_PAGES_JSON=/path/to/pages.json or call basic_test() programmatically.")
