from __future__ import annotations

import time
import uuid
import math
import logging
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Iterable
import threading
import concurrent.futures

import numpy as np

# Attempt to import Ultralytics YOLO;
try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None  # type: ignore

# Torch for device detection
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore


# ------------------------------
# Exception hierarchy
# ------------------------------
class DocumentClassifierError(Exception): pass
class ModelLoadError(DocumentClassifierError): pass
class ClassificationConfidenceError(DocumentClassifierError): pass
class InvalidClassError(DocumentClassifierError): pass
class PageAssociationError(DocumentClassifierError): pass
class EmptyInputError(DocumentClassifierError): pass

# ------------------------------
# Config dataclass
# ------------------------------
@dataclass
class ClassifierConfig:
    # Model
    model_path: str = "models/Id_Classifier.pt"
    model_type: str = "classification"  # kept for clarity
    target_classes: List[str] = field(default_factory=lambda: [
        'aadhar_front', 'aadhar_back',
        'driving_license_front', 'driving_license_back',
        'pan_card_front'
    ])
    confidence_threshold: float = 0.70
    image_size: int = 640
    batch_size: int = 4

    # Confidence stratification
    CONFIDENCE_LEVELS: Dict[str, float] = field(default_factory=lambda: {
        'HIGH': 0.9,
        'MEDIUM': 0.70,
        'LOW': 0.0
    })

    # Runtime/perf
    device: Optional[str] = None  # "cuda" or "cpu" or None for auto-detect
    num_workers: int = 2  # for async batch predictions
    model_cache_enabled: bool = True

    # Retry logic
    borderline_retry_attempts: int = 1  # re-run inference for borderline MEDIUM if desired
    borderline_retry_delay: float = 0.15  # seconds

    # Logging / monitoring
    enable_logging: bool = True
    log_level: int = logging.INFO


# ------------------------------
# Utility functions
# ------------------------------
def _setup_logger(cfg: ClassifierConfig) -> logging.Logger:
    logger = logging.getLogger("DocumentClassifier")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "[DocumentClassifier] %(asctime)s %(levelname)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(cfg.log_level if cfg.enable_logging else logging.CRITICAL)
    return logger


def _detect_device(cfg: ClassifierConfig) -> str:
    if cfg.device:
        return cfg.device
    # Auto-detect: prefer CUDA if available
    try:
        if torch is not None and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _generate_doc_id(prefix: str = "DOC") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _map_raw_class_to_document_type(raw_cls: str) -> Tuple[str, str]:
    """
    Map YOLO raw predicted class name to document_type and side.
    Expected mappings per spec:
       'aadhar_front' -> ('aadhar', 'front')
       'aadhar_back'  -> ('aadhar', 'back')
       'driving_license_front' -> ('driving_license', 'front')
       'driving_license_back' -> ('driving_license', 'back')
       'pan_card_front' -> ('pan', 'single')
    Raise InvalidClassError for unrecognized class.
    """
    if not raw_cls:
        raise InvalidClassError("Empty class")
    raw_cls = str(raw_cls).lower()
    if raw_cls.startswith("aadhar"):
        side = "front" if raw_cls.endswith("front") else "back"
        return "aadhar", side
    if raw_cls.startswith("driving_license"):
        side = "front" if raw_cls.endswith("front") else "back"
        return "driving_license", side
    if raw_cls.startswith("pan_card") or raw_cls.startswith("pan"):
        return "pan", "single"
    raise InvalidClassError(f"Unknown prediction class: {raw_cls}")


# ------------------------------
# New helper utilities for robust probs/class extraction
# ------------------------------
def _resolve_class_name(model_names: Any, idx_or_name: Any) -> str:
    """
    Given model.names (dict or list) and either an index or a name-like value,
    return a string class name deterministically.
    """
    try:
        if model_names is None:
            return str(idx_or_name)
        if isinstance(model_names, dict):
            # idx_or_name may be int or string that maps to dict keys
            return str(model_names.get(idx_or_name, model_names.get(int(idx_or_name), str(idx_or_name))))
        if isinstance(model_names, (list, tuple)):
            try:
                if isinstance(idx_or_name, int):
                    return str(model_names[idx_or_name]) if idx_or_name < len(model_names) else str(idx_or_name)
                # sometimes idx_or_name is numeric string
                try:
                    ii = int(idx_or_name)
                    return str(model_names[ii]) if ii < len(model_names) else str(idx_or_name)
                except Exception:
                    # or it's already a name
                    return str(idx_or_name)
            except Exception:
                return str(idx_or_name)
    except Exception:
        return str(idx_or_name)
    return str(idx_or_name)


def _extract_probs_choice(probs: Any, model: Any, logger: Optional[logging.Logger] = None) -> Tuple[Optional[str], float]:
    """
    Try robust extraction of top class and confidence from a 'probs' object produced by ultralytics.
    Returns (class_name_or_None, confidence_float_in_0_1)
    """
    try:
        # Prefer direct top1/top1conf if present
        if hasattr(probs, "top1conf") and hasattr(probs, "top1"):
            try:
                top1conf = getattr(probs, "top1conf")
                top1 = getattr(probs, "top1")
                # handle torch tensors or simple types
                try:
                    import torch as _torch
                except Exception:
                    _torch = None
                if _torch is not None and isinstance(top1conf, _torch.Tensor):
                    conf = float(top1conf.cpu().detach().item())
                else:
                    conf = float(top1conf)
                # top1 can be int, tensor, or list
                if _torch is not None and isinstance(top1, _torch.Tensor):
                    idx = int(top1.cpu().detach().item())
                else:
                    try:
                        idx = int(top1)
                    except Exception:
                        # if it's iterable
                        try:
                            idx = int(list(top1)[0])
                        except Exception:
                            idx = top1
                name = _resolve_class_name(getattr(model, "names", None), idx)
                # normalize confidence
                if conf > 1.0:
                    conf = conf / 100.0
                conf = max(0.0, min(1.0, float(conf)))
                return name, conf
            except Exception:
                # fallthrough to more generic extraction
                if logger:
                    logger.debug("probs.top1/top1conf extraction failed; falling back to generic extraction")

        # Generic extraction path similar to prior logic but simpler and more deterministic
        p_arr = None
        try:
            import torch as _torch
        except Exception:
            _torch = None

        # If probs itself is a tensor
        if _torch is not None and isinstance(probs, _torch.Tensor):
            try:
                p_arr = probs.cpu().detach().numpy()
            except Exception:
                p_arr = None

        # If poses has .tensor attribute (Ultralytics Probs)
        if p_arr is None and hasattr(probs, "tensor"):
            try:
                t = getattr(probs, "tensor")
                if _torch is not None and isinstance(t, _torch.Tensor):
                    p_arr = t.cpu().detach().numpy()
                else:
                    p_arr = np.asarray(t)
            except Exception:
                p_arr = None

        # try __array__/numpy/tolist/cpu/detach sequence
        if p_arr is None and hasattr(probs, "__array__"):
            try:
                p_arr = np.asarray(probs)
            except Exception:
                p_arr = None

        if p_arr is None and hasattr(probs, "numpy"):
            try:
                p_arr = probs.numpy()
            except Exception:
                p_arr = None

        if p_arr is None and hasattr(probs, "tolist"):
            try:
                p_arr = np.array(probs.tolist())
            except Exception:
                p_arr = None

        if p_arr is None and hasattr(probs, "cpu"):
            try:
                maybe = probs.cpu()
                if _torch is not None and isinstance(maybe, _torch.Tensor):
                    p_arr = maybe.detach().numpy()
                else:
                    p_arr = np.asarray(maybe)
            except Exception:
                p_arr = None

        # fallback: iterate
        if p_arr is None:
            try:
                p_arr = np.array(list(probs))
            except Exception:
                p_arr = None

        # If we successfully got an array-like
        if p_arr is not None:
            try:
                p_np = np.array(p_arr).ravel().astype(float)
            except Exception:
                try:
                    p_np = np.array([float(x) for x in p_arr]).ravel()
                except Exception:
                    p_np = None
            if p_np is not None and p_np.size:
                idx = int(np.argmax(p_np))
                conf = float(p_np[idx])
                if conf > 1.0:
                    conf = conf / 100.0
                conf = max(0.0, min(1.0, conf))
                name = _resolve_class_name(getattr(model, "names", None), idx)
                return name, conf

        # Last resort: regex parse of repr
        try:
            txt = repr(probs)
            import re
            m = re.search(r"([a-z0-9_]+)\s+([01]\.\d+|\d\.\d+|\d+)", txt, flags=re.IGNORECASE)
            if m:
                cls_hint = m.group(1)
                prob_hint = m.group(2)
                try:
                    prob_val = float(prob_hint)
                    if prob_val > 1.0:
                        prob_val = prob_val / 100.0
                    prob_val = max(0.0, min(1.0, prob_val))
                    # cls_hint might already be a name
                    return cls_hint, float(prob_val)
                except Exception:
                    pass
        except Exception:
            pass

    except Exception:
        if logger:
            logger.exception("Unexpected error while extracting probs choice")
    return None, 0.0


# ------------------------------
# Core DocumentClassifier
# ------------------------------
class DocumentClassifier:
    def __init__(self, cfg: Optional[ClassifierConfig] = None):
        self.cfg = cfg or ClassifierConfig()
        self.logger = _setup_logger(self.cfg)
        self.device = _detect_device(self.cfg)
        self._model_lock = threading.Lock()
        self._model = None  # cached model instance
        self._model_loaded = False

    # --------------------------
    # Model loading and caching
    # --------------------------
    def load_model(self, force_reload: bool = False):
        """
        Load and cache the Ultralytics YOLO model. Raises ModelLoadError on failure.
        """
        if self.cfg.model_cache_enabled and self._model_loaded and not force_reload:
            self.logger.info("Using cached model instance")
            return self._model

        if YOLO is None:
            raise ModelLoadError("Ultralytics YOLO package not available. Install ultralytics.")

        with self._model_lock:
            try:
                self.logger.info(f"Loading YOLO model from {self.cfg.model_path} on device={self.device}")
                # instantiate YOLO model
                model = YOLO(self.cfg.model_path)
                self.logger.debug("model.names: %s", getattr(model, "names", None))
                # set device (Ultralytics will accept "cuda" or "cpu")
                try:
                    model.to(self.device)
                except Exception:
                    # Older/newer ultralytics versions may not have .to(); ignore if not supported
                    pass

                # basic sanity check: model.names should include expected classes
                model_classes = getattr(model, "names", None)
                if model_classes is None:
                    # can't verify but proceed
                    self.logger.warning("Unable to inspect model.names; skipping class verification")
                else:
                    # ensure at least the target classes are present OR map numeric classes to names
                    # model.names may be dict or list-like
                    try:
                        names_values = list(model_classes.values()) if isinstance(model_classes, dict) else list(model_classes)
                    except Exception:
                        names_values = []
                    missing = [c for c in self.cfg.target_classes if c not in names_values and c not in model_classes]
                    if missing:
                        # Not fatal but warn
                        self.logger.warning(f"Model appears to be missing expected classes: {missing}")

                self._model = model
                self._model_loaded = True
                self.logger.info("Model loaded successfully")
                return self._model
            except Exception as e:
                tb = traceback.format_exc()
                raise ModelLoadError(f"Failed to load YOLO model: {e}\n{tb}")

    def _prepare_batch(self, batch_items: List[Dict[str, Any]]) -> List[Any]:
        """
        Prepare images for Ultralytics YOLO predict call.
        Accepts items like {'image_array': np.ndarray, 'image': PIL.Image, 'image_path': str, 'metadata': ...}
        Returns list of numpy arrays or paths suitable for model.predict().
        """
        imgs: List[Any] = []
        for i, item in enumerate(batch_items):
            img = None

            if isinstance(item, dict):
                img = item.get("image_array", None)
                if img is None:
                    img = item.get("image", None)
                img_path = item.get("image_path", None)
                if img_path:
                    imgs.append(img_path)
                    continue

            if img is not None:
                try:
                    from PIL import Image
                    if isinstance(img, Image.Image):
                        img = np.array(img)
                except Exception:
                    pass

            if img is None:
                self.logger.warning("Batch item %d missing image data; inserting dummy image", i)
                imgs.append(np.zeros((10, 10, 3), dtype=np.uint8))
                continue

            try:
                arr = np.asarray(img)

                # Grayscale -> RGB
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)

                # Drop alpha channel if present
                if arr.ndim == 3 and arr.shape[2] == 4:
                    arr = arr[..., :3]

                # Ensure HWC and 3 channels
                if arr.ndim != 3 or arr.shape[2] not in (1, 3):
                    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] != 3:
                        arr = np.moveaxis(arr, 0, -1)
                    else:
                        self.logger.warning(
                            "Batch item %d produced unexpected image shape %s; using dummy",
                            i, getattr(arr, "shape", None)
                        )
                        imgs.append(np.zeros((10, 10, 3), dtype=np.uint8))
                        continue

                # Normalize dtype to uint8 in 0..255
                if arr.dtype in (np.float32, np.float64):
                    if np.nanmax(arr) <= 1.0:
                        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        arr = arr.clip(0, 255).astype(np.uint8)
                elif arr.dtype != np.uint8:
                    arr = arr.astype(np.uint8)

                imgs.append(arr)
            except Exception as e:
                self.logger.exception("Failed to prepare image for batch index %d: %s", i, e)
                imgs.append(np.zeros((10, 10, 3), dtype=np.uint8))

        return imgs
    
    # --------------------------
    # Prediction utilities
    # --------------------------
    def _predict_batch(self, batch_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run model.predict() over a small batch and return per-input predictions as dicts:
        { 'predicted_class': str, 'confidence': float, 'inference_time': float }
        Implements retry for borderline confidence if configured.
        """
        if not batch_items:
            return []

        model = self.load_model()
        imgs = self._prepare_batch(batch_items)
        start = time.perf_counter()
        try:
            # NOTE: Do not pass 'conf' to model.predict so we receive full probs vector.
            # We'll apply confidence thresholding after robust extraction.
            predict_kwargs = dict(imgsz=self.cfg.image_size, batch=len(imgs))
            try:
                result = model.predict(imgs, **predict_kwargs)
            except TypeError:
                # Older API or different signature - try model(imgs, **kwargs)
                result = model(imgs, **predict_kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            self.logger.error(f"Model inference failed: {e}\n{tb}")
            # Inference failure for whole batch: raise ModelLoadError
            raise ModelLoadError(f"Model inference error: {e}")

        end = time.perf_counter()
        batch_time = end - start

        # Normalize/inspect results in a robust way
        try:
            # If result is iterable, make a list; otherwise wrap single result
            try:
                res_list = list(result)
            except TypeError:
                res_list = [result]
        except Exception:
            # Last-resort wrap
            res_list = [result]

        self.logger.debug(f"Model returned result object type={type(result)} entries={len(res_list)}")

        parsed: List[Dict[str, Any]] = []
        try:
            for i, res in enumerate(res_list):
                chosen_class = None
                chosen_conf = 0.0

                # Common structured attributes
                boxes = getattr(res, "boxes", None)
                probs = getattr(res, "probs", None)
                try:
                    prob_attrs = {k: hasattr(probs, k) for k in ("tensor", "numpy", "tolist", "__array__", "cpu", "shape", "top1", "top1conf")}
                    self.logger.debug("PROBS DEBUG: type=%s repr=%s attrs=%s", type(probs), repr(probs)[:500], prob_attrs)
                except Exception:
                    self.logger.debug("PROBS DEBUG: could not introspect probs object")

                # 1) Try deterministic probs extraction (prefer top1/top1conf)
                if probs is not None:
                    try:
                        name, conf = _extract_probs_choice(probs, model, logger=self.logger)
                        if name is not None and conf is not None and conf > 0.0:
                            chosen_class = name
                            chosen_conf = float(conf)
                            self.logger.debug("Extracted from probs: class=%s conf=%.6f", chosen_class, chosen_conf)
                    except Exception:
                        chosen_class, chosen_conf = None, 0.0

                # 2) boxes with .cls / .conf (object-detection style) - only if probs didn't yield result
                if (chosen_class is None or chosen_conf == 0.0) and boxes is not None:
                    try:
                        cls_vals = getattr(boxes, "cls", None)
                        conf_vals = getattr(boxes, "conf", None)
                        if cls_vals is not None and conf_vals is not None:
                            cls_arr = np.array(cls_vals).ravel()
                            conf_arr = np.array(conf_vals).ravel()
                            if conf_arr.size:
                                idx_top = int(np.argmax(conf_arr))
                                class_idx = int(cls_arr[idx_top])
                                chosen_conf = float(conf_arr[idx_top])
                                chosen_class = _resolve_class_name(getattr(model, "names", None), class_idx)
                                # normalize confidence
                                if chosen_conf > 1.0:
                                    chosen_conf = chosen_conf / 100.0
                                chosen_conf = max(0.0, min(1.0, chosen_conf))
                        else:
                            # try boxes.data fallback
                            data = getattr(boxes, "data", None)
                            if data is not None:
                                arr = np.array(data)
                                if arr.size and arr.ndim == 2 and arr.shape[1] >= 6:
                                    conf_arr = arr[:, 4].astype(float)
                                    idx_top = int(np.argmax(conf_arr))
                                    chosen_conf = float(conf_arr[idx_top])
                                    class_idx = int(arr[idx_top, 5])
                                    chosen_class = _resolve_class_name(getattr(model, "names", None), class_idx)
                                    if chosen_conf > 1.0:
                                        chosen_conf = chosen_conf / 100.0
                                    chosen_conf = max(0.0, min(1.0, chosen_conf))
                    except Exception:
                        chosen_class, chosen_conf = None, 0.0

                # 3) res.pred (some versions)
                if (chosen_class is None or chosen_conf == 0.0) and hasattr(res, "pred"):
                    try:
                        pred_arr = np.array(getattr(res, "pred"))
                        if pred_arr.size and pred_arr.ndim == 2 and pred_arr.shape[1] >= 6:
                            conf_arr = pred_arr[:, 4].astype(float)
                            idx_top = int(np.argmax(conf_arr))
                            chosen_conf = float(conf_arr[idx_top])
                            class_idx = int(pred_arr[idx_top, 5])
                            chosen_class = _resolve_class_name(getattr(model, "names", None), class_idx)
                            if chosen_conf > 1.0:
                                chosen_conf = chosen_conf / 100.0
                            chosen_conf = max(0.0, min(1.0, chosen_conf))
                    except Exception:
                        pass

                # 4) other plausible attributes
                if (chosen_class is None or chosen_conf == 0.0):
                    try:
                        for attr in ("scores", "score", "scores_", "labels", "classes", "class_ids", "conf", "confidence"):
                            if hasattr(res, attr):
                                val = getattr(res, attr)
                                try:
                                    if np.isscalar(val):
                                        maybe_conf = float(val)
                                        if maybe_conf > chosen_conf:
                                            chosen_conf = maybe_conf
                                except Exception:
                                    pass
                                try:
                                    arr = np.array(val)
                                    if arr.size:
                                        if arr.ndim == 0:
                                            maybe_conf = float(arr.item())
                                            if maybe_conf > chosen_conf:
                                                chosen_conf = maybe_conf
                                        else:
                                            idx_top = int(np.argmax(arr))
                                            maybe_conf = float(arr.ravel()[idx_top])
                                            if maybe_conf > chosen_conf:
                                                chosen_conf = maybe_conf
                                except Exception:
                                    pass
                        for attr in ("labels", "classes", "class_ids"):
                            if (chosen_class is None or chosen_class == "unknown") and hasattr(res, attr):
                                lab = getattr(res, attr)
                                try:
                                    lab_arr = np.array(lab).ravel()
                                    if lab_arr.size:
                                        class_idx = int(lab_arr[0])
                                        chosen_class = _resolve_class_name(getattr(model, "names", None), class_idx)
                                except Exception:
                                    pass
                        # normalize chosen_conf if set
                        if chosen_conf and chosen_conf > 1.0:
                            chosen_conf = chosen_conf / 100.0
                        chosen_conf = float(max(0.0, min(1.0, chosen_conf or 0.0)))
                    except Exception:
                        pass

                # 5) probs_dict (older output)
                if (chosen_class is None or chosen_conf == 0.0) and hasattr(res, "probs_dict"):
                    try:
                        pd = getattr(res, "probs_dict")
                        if isinstance(pd, dict) and pd:
                            topk = sorted(pd.items(), key=lambda x: -x[1])[0]
                            chosen_class = topk[0]
                            chosen_conf = float(topk[1])
                            if chosen_conf > 1.0:
                                chosen_conf = chosen_conf / 100.0
                            chosen_conf = max(0.0, min(1.0, chosen_conf))
                    except Exception:
                        pass

                # 6) FINAL FALLBACK: regex parse of str(res)
                if (chosen_class is None or chosen_conf == 0.0):
                    try:
                        txt = str(res)
                        import re
                        m = re.search(r"([a-z0-9_]+)\s+([01]\.\d+|\d\.\d+|\d+)", txt, flags=re.IGNORECASE)
                        if m:
                            cls_hint = m.group(1)
                            prob_hint = m.group(2)
                            try:
                                prob_val = float(prob_hint)
                                if prob_val > 1.0:
                                    prob_val = prob_val / 100.0
                                chosen_conf = float(max(0.0, min(1.0, prob_val)))
                                chosen_class = cls_hint
                                self.logger.debug("Regex fallback parsed class=%s conf=%.4f from result str", chosen_class, chosen_conf)
                            except Exception:
                                pass
                    except Exception:
                        pass

                # Final guards
                if chosen_class is None:
                    chosen_class = "unknown"
                try:
                    chosen_conf = float(chosen_conf or 0.0)
                except Exception:
                    chosen_conf = 0.0

                # Extra debug: if probs existed but we still have zero conf, log for investigation
                if chosen_conf == 0.0 and probs is not None:
                    try:
                        self.logger.debug("Probs present but parsed confidence is 0.0; repr(probs)=%s", repr(probs)[:800])
                    except Exception:
                        self.logger.debug("Probs present but parsed confidence is 0.0; unable to repr(probs)")

                parsed.append({
                    "predicted_class": chosen_class,
                    "confidence": chosen_conf,
                    "inference_time": batch_time / max(1, len(imgs))
                })

        except Exception as e:
            tb = traceback.format_exc()
            self.logger.error(f"Failed to parse model predictions: {e}\n{tb}")
            # fallback - preserve input count
            for _ in range(len(batch_items)):
                parsed.append({"predicted_class": None, "confidence": 0.0, "inference_time": batch_time / max(1, len(imgs))})

        # Post-processing: ensure types & clipping
        try:
            target_set = set([str(x).lower() for x in (getattr(self.cfg, "target_classes", []) or [])])
            if target_set:
                for p in parsed:
                    try:
                        cls_name = (p.get("predicted_class") or "").strip()
                        if cls_name and cls_name.lower() not in target_set:
                            # Non-target predicted class -> mark as unknown
                            self.logger.debug("Dropping non-target predicted class '%s' (not in target_classes)", cls_name)
                            p["predicted_class"] = "unknown"
                            p["confidence"] = 0.0
                    except Exception:
                        # be defensive: do not break parsing flow
                        continue
        except Exception:
            # if any failure here, proceed without filtering but log for debug
            self.logger.exception("Failed while filtering non-target classes")

        for p in parsed:
            if p["predicted_class"] is None:
                p["predicted_class"] = "unknown"
            try:
                p["confidence"] = float(p.get("confidence", 0.0))
            except Exception:
                p["confidence"] = 0.0
            # clip to 0..1
            if p["confidence"] > 1.0:
                p["confidence"] = p["confidence"] / 100.0
            p["confidence"] = max(0.0, min(1.0, p["confidence"]))

        # Retry logic for borderline cases (MEDIUM)
        if self.cfg.borderline_retry_attempts > 0:
            for idx, p in enumerate(parsed):
                if self.cfg.CONFIDENCE_LEVELS["MEDIUM"] <= p["confidence"] < self.cfg.CONFIDENCE_LEVELS["HIGH"]:
                    attempt = 0
                    while attempt < self.cfg.borderline_retry_attempts:
                        attempt += 1
                        self.logger.info(f"Retrying borderline prediction (attempt {attempt}) for item idx {idx}")
                        time.sleep(self.cfg.borderline_retry_delay)
                        try:
                            single = [batch_items[idx]]
                            retr = self._predict_batch_single(single, model_override=model)
                            if retr:
                                p2 = retr[0]
                                if p2["confidence"] > p["confidence"]:
                                    p.update(p2)
                                    break
                        except Exception:
                            self.logger.exception("Retry inference failed")
                            break

        return parsed


    def _predict_batch_single(self, batch_items: List[Dict[str, Any]], model_override=None) -> List[Dict[str, Any]]:
        """
        Helper that runs a single-image inference via the same mechanics as _predict_batch but allows passing
        an already-loaded model to avoid recursion errors.
        """
        model = model_override or self.load_model()
        imgs = self._prepare_batch(batch_items)
        start = time.perf_counter()
        try:
            # Do not pass 'conf' to the model; we want raw scores.
            predict_kwargs = dict(imgsz=self.cfg.image_size, batch=1)
            try:
                result = model.predict(imgs, **predict_kwargs)
            except TypeError:
                result = model(imgs, **predict_kwargs)
        except Exception as e:
            self.logger.error(f"Single inference error: {e}")
            return [{"predicted_class": "unknown", "confidence": 0.0, "inference_time": 0.0}]

        end = time.perf_counter()
        inferred_time = end - start

        # Normalize result to list
        try:
            res_list = list(result)
        except Exception:
            res_list = [result]

        parsed = []
        try:
            for res in res_list:
                chosen_class = None
                chosen_conf = 0.0

                boxes = getattr(res, "boxes", None)
                probs = getattr(res, "probs", None)
                try:
                    prob_attrs = {k: hasattr(probs, k) for k in ("tensor", "numpy", "tolist", "__array__", "cpu", "shape", "top1", "top1conf")}
                    self.logger.debug("PROBS DEBUG: type=%s repr=%s attrs=%s", type(probs), repr(probs)[:500], prob_attrs)
                except Exception:
                    self.logger.debug("PROBS DEBUG: could not introspect probs object")

                # 1) Try deterministic probs extraction (prefer top1/top1conf)
                if probs is not None:
                    try:
                        name, conf = _extract_probs_choice(probs, model, logger=self.logger)
                        if name is not None and conf is not None and conf > 0.0:
                            chosen_class = name
                            chosen_conf = float(conf)
                            self.logger.debug("Extracted from single probs: class=%s conf=%.6f", chosen_class, chosen_conf)
                    except Exception:
                        chosen_class, chosen_conf = None, 0.0

                # 2) boxes fallback
                if (chosen_class is None or chosen_conf == 0.0) and boxes is not None:
                    try:
                        cls_vals = getattr(boxes, "cls", None)
                        conf_vals = getattr(boxes, "conf", None)
                        if cls_vals is not None and conf_vals is not None:
                            cls_arr = np.array(cls_vals).ravel()
                            conf_arr = np.array(conf_vals).ravel()
                            if conf_arr.size:
                                idx_top = int(np.argmax(conf_arr))
                                chosen_conf = float(conf_arr[idx_top])
                                class_idx = int(cls_arr[idx_top])
                                chosen_class = _resolve_class_name(getattr(model, "names", None), class_idx)
                                if chosen_conf > 1.0:
                                    chosen_conf = chosen_conf / 100.0
                                chosen_conf = max(0.0, min(1.0, chosen_conf))
                        else:
                            data = getattr(boxes, "data", None)
                            if data is not None:
                                arr = np.array(data)
                                if arr.size and arr.ndim == 2 and arr.shape[1] >= 6:
                                    conf_arr = arr[:, 4].astype(float)
                                    idx_top = int(np.argmax(conf_arr))
                                    chosen_conf = float(conf_arr[idx_top])
                                    class_idx = int(arr[idx_top, 5])
                                    chosen_class = _resolve_class_name(getattr(model, "names", None), class_idx)
                                    if chosen_conf > 1.0:
                                        chosen_conf = chosen_conf / 100.0
                                    chosen_conf = max(0.0, min(1.0, chosen_conf))
                    except Exception:
                        chosen_class, chosen_conf = None, 0.0

                # 3) res.pred
                if (chosen_class is None or chosen_conf == 0.0) and hasattr(res, "pred"):
                    try:
                        pred_arr = np.array(getattr(res, "pred"))
                        if pred_arr.size and pred_arr.ndim == 2 and pred_arr.shape[1] >= 6:
                            conf_arr = pred_arr[:, 4].astype(float)
                            idx_top = int(np.argmax(conf_arr))
                            chosen_conf = float(conf_arr[idx_top])
                            class_idx = int(pred_arr[idx_top, 5])
                            chosen_class = _resolve_class_name(getattr(model, "names", None), class_idx)
                            if chosen_conf > 1.0:
                                chosen_conf = chosen_conf / 100.0
                            chosen_conf = max(0.0, min(1.0, chosen_conf))
                    except Exception:
                        pass

                # 4) other fallbacks & normalizations (same pattern as batch)
                if (chosen_class is None or chosen_conf == 0.0):
                    try:
                        for attr in ("scores", "score", "labels", "classes", "class_ids", "conf", "confidence"):
                            if hasattr(res, attr):
                                val = getattr(res, attr)
                                try:
                                    if np.isscalar(val):
                                        maybe_conf = float(val)
                                        if maybe_conf > chosen_conf:
                                            chosen_conf = maybe_conf
                                except Exception:
                                    pass
                                try:
                                    arr = np.array(val)
                                    if arr.size:
                                        if arr.ndim == 0:
                                            maybe_conf = float(arr.item())
                                            if maybe_conf > chosen_conf:
                                                chosen_conf = maybe_conf
                                        else:
                                            idx_top = int(np.argmax(arr))
                                            maybe_conf = float(arr.ravel()[idx_top])
                                            if maybe_conf > chosen_conf:
                                                chosen_conf = maybe_conf
                                except Exception:
                                    pass
                        for attr in ("labels", "classes", "class_ids"):
                            if (chosen_class is None or chosen_class == "unknown") and hasattr(res, attr):
                                lab = getattr(res, attr)
                                try:
                                    lab_arr = np.array(lab).ravel()
                                    if lab_arr.size:
                                        class_idx = int(lab_arr[0])
                                        chosen_class = _resolve_class_name(getattr(model, "names", None), class_idx)
                                except Exception:
                                    pass
                        if chosen_conf and chosen_conf > 1.0:
                            chosen_conf = chosen_conf / 100.0
                        chosen_conf = float(max(0.0, min(1.0, chosen_conf or 0.0)))
                    except Exception:
                        pass

                if (chosen_class is None or chosen_conf == 0.0) and hasattr(res, "probs_dict"):
                    try:
                        pd = getattr(res, "probs_dict")
                        if isinstance(pd, dict) and pd:
                            topk = sorted(pd.items(), key=lambda x: -x[1])[0]
                            chosen_class = topk[0]
                            chosen_conf = float(topk[1])
                            if chosen_conf > 1.0:
                                chosen_conf = chosen_conf / 100.0
                            chosen_conf = max(0.0, min(1.0, chosen_conf))
                    except Exception:
                        pass

                if (chosen_class is None or chosen_conf == 0.0):
                    try:
                        txt = str(res)
                        import re
                        m = re.search(r"([a-z0-9_]+)\s+([01]\.\d+|\d\.\d+|\d+)", txt, flags=re.IGNORECASE)
                        if m:
                            cls_hint = m.group(1)
                            prob_hint = m.group(2)
                            try:
                                prob_val = float(prob_hint)
                                if prob_val > 1.0:
                                    prob_val = prob_val / 100.0
                                chosen_conf = float(max(0.0, min(1.0, prob_val)))
                                chosen_class = cls_hint
                                self.logger.debug("Regex fallback parsed class=%s conf=%.4f from single-result str", chosen_class, chosen_conf)
                            except Exception:
                                pass
                    except Exception:
                        pass

                if chosen_class is None:
                    chosen_class = "unknown"
                try:
                    chosen_conf = float(chosen_conf or 0.0)
                except Exception:
                    chosen_conf = 0.0

                if chosen_conf == 0.0 and probs is not None:
                    try:
                        self.logger.debug("Single: probs present but parsed confidence is 0.0; repr(probs)=%s", repr(probs)[:800])
                    except Exception:
                        self.logger.debug("Single: probs present but parsed confidence is 0.0; unable to repr(probs)")

                parsed.append({"predicted_class": chosen_class, "confidence": float(chosen_conf), "inference_time": inferred_time})
        except Exception:
            parsed.append({"predicted_class": "unknown", "confidence": 0.0, "inference_time": inferred_time})
        return parsed

    # --------------------------
    # Public classification API
    # --------------------------
    def classify_pages(self, preprocessed_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry to classify a list of preprocessed pages.
        Input: list of dicts as provided by document_preprocessor (image_array + metadata)
        Output: per-page classification dicts per spec.
        """

        if not preprocessed_pages:
            raise EmptyInputError("No preprocessed pages supplied to classifier")

        # Validate input format quickly
        for item in preprocessed_pages:
            if "image_array" not in item or "metadata" not in item:
                raise EmptyInputError("Each input item must contain 'image_array' and 'metadata'")

        # Batch loop - do not load all images in memory if many; process in chunks
        results_per_page: List[Dict[str, Any]] = []
        total_pages = len(preprocessed_pages)
        self.logger.info(f"Classifying {total_pages} pages with batch_size={self.cfg.batch_size}")

        idx = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg.num_workers) as executor:
            futures = []
            while idx < total_pages:
                batch_slice = preprocessed_pages[idx: idx + self.cfg.batch_size]
                futures.append(executor.submit(self._predict_and_assemble, batch_slice))
                idx += self.cfg.batch_size

            # Collect in original submission order to preserve page order
            for fut in futures:
                batch_results = fut.result()
                results_per_page.extend(batch_results)

        # Ensure results are sorted by original page_number (to preserve order)
        results_per_page.sort(key=lambda x: x.get("page_number", 0))
        return results_per_page

    def _predict_and_assemble(self, batch_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Modified to preserve original image_array and metadata while adding classification results.
        Returns per-page dicts that maintain all original data.
        """
        # Run predictions (unchanged)
        pred_list = self._predict_batch(batch_items)

        # Ensure prediction list length matches inputs
        if len(pred_list) != len(batch_items):
            self.logger.warning(
                "Prediction count mismatch: batch_items=%d, pred_list=%d. Padding missing predictions with 'unknown'.",
                len(batch_items), len(pred_list)
            )
            while len(pred_list) < len(batch_items):
                pred_list.append({"predicted_class": "unknown", "confidence": 0.0, "inference_time": 0.0})

        assembled = []
        for item, pred in zip(batch_items, pred_list):
            # PRESERVE ORIGINAL DATA
            original_image_array = item.get("image_array")
            original_metadata = item.get("metadata", {}) or {}
            
            page_num_raw = original_metadata.get("page_number", -1)
            try:
                page_num = int(page_num_raw) if isinstance(page_num_raw, (int, float, str)) and str(page_num_raw).isdigit() else int(page_num_raw) if isinstance(page_num_raw, int) else -1
            except Exception:
                page_num = -1
                
            raw_cls = pred.get("predicted_class", "unknown")
            try:
                confidence = float(pred.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            try:
                inference_time = float(pred.get("inference_time", 0.0))
            except Exception:
                inference_time = 0.0

            # Confidence enforcement - log if below threshold
            if confidence < self.cfg.confidence_threshold:
                self.logger.warning(f"Page {page_num}: confidence {confidence:.3f} < threshold {self.cfg.confidence_threshold:.3f}")

            # Map to document_type and side (CRITICAL: Move this logic here from orchestrator)
            try:
                doc_type, side = _map_raw_class_to_document_type(raw_cls)
            except InvalidClassError:
                doc_type = "unknown"
                side = "unknown"

            # PRESERVE ALL ORIGINAL DATA + ADD CLASSIFICATION RESULTS
            out = {
                'page_number': page_num,
                'predicted_class': raw_cls,
                'confidence': confidence,
                'processing_time': inference_time,
                'document_type': doc_type,  # ADDED: normalized type
                'side': side,               # ADDED: normalized side
                'image_array': original_image_array,  # PRESERVED: critical for extractor
                'metadata': original_metadata  # PRESERVED: all original metadata
            }
            assembled.append(out)
        return assembled

    # --------------------------
    # Document grouping & validation
    # --------------------------
    def group_pages_into_documents(self, per_page_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group consecutive pages into logical documents.
        CRITICAL: All pages in the grouping MUST preserve their original image_array and metadata.
        """
        if not per_page_results:
            return []

        docs: List[Dict[str, Any]] = []
        idx = 0
        n = len(per_page_results)

        while idx < n:
            pg = per_page_results[idx]
            p_type = pg.get("document_type", "unknown")
            p_side = pg.get("side", "unknown")

            # Helper to create a doc - PRESERVES ALL PAGE DATA
            def make_doc(pages: List[Dict[str, Any]], dtype: str):
                doc_id = _generate_doc_id(prefix=dtype.upper() if dtype else "DOC")
                
                # Determine completeness heuristics
                pages_by_side = {p.get("side", "unknown"): p for p in pages}
                missing_sides = []
                complete = True
                
                if dtype == "aadhar":
                    if "front" not in pages_by_side or "back" not in pages_by_side:
                        complete = False
                        if "front" not in pages_by_side: missing_sides.append("front")
                        if "back" not in pages_by_side: missing_sides.append("back")
                elif dtype == "driving_license":
                    if "front" not in pages_by_side or "back" not in pages_by_side:
                        complete = False
                        if "front" not in pages_by_side: missing_sides.append("front")
                        if "back" not in pages_by_side: missing_sides.append("back")
                elif dtype == "pan":
                    if len(pages) != 1:
                        complete = False
                else:
                    complete = False
                    
                return {
                    "document_id": doc_id,
                    "document_type": dtype,
                    "pages": pages,  # PRESERVES ALL ORIGINAL PAGE DATA INCLUDING image_array
                    "complete": complete,
                    "missing_sides": missing_sides
                }

            # Grouping logic:
            # If aadhar front encountered and next page is aadhar_back => pair them
            if p_type == "aadhar" and p_side == "front":
                # try to pair with next page if it's aadhar_back
                if idx + 1 < n:
                    next_pg = per_page_results[idx + 1]
                    if next_pg.get("document_type") == "aadhar" and next_pg.get("side") == "back":
                        docs.append(make_doc([pg, next_pg], "aadhar"))
                        idx += 2
                        continue
                # else front-only
                docs.append(make_doc([pg], "aadhar"))
                idx += 1
                continue

            # If aadhar back without preceding front: treat as single doc but mark incomplete
            if p_type == "aadhar" and p_side == "back":
                docs.append(make_doc([pg], "aadhar"))
                idx += 1
                continue

            # Driving license logic: if front followed by back -> pair
            if p_type == "driving_license" and p_side == "front":
                if idx + 1 < n:
                    next_pg = per_page_results[idx + 1]
                    if next_pg.get("document_type") == "driving_license" and next_pg.get("side") == "back":
                        docs.append(make_doc([pg, next_pg], "driving_license"))
                        idx += 2
                        continue
                # front-only accepted
                docs.append(make_doc([pg], "driving_license"))
                idx += 1
                continue

            if p_type == "driving_license" and p_side == "back":
                # back-only: make a doc with missing front
                docs.append(make_doc([pg], "driving_license"))
                idx += 1
                continue

            # PAN: single page only
            if p_type == "pan":
                docs.append(make_doc([pg], "pan"))
                idx += 1
                continue

            # Unknown or other types: group consecutive unknowns into separate docs each
            docs.append({
                "document_id": _generate_doc_id(prefix="UNKNOWN"),
                "document_type": p_type or "unknown",
                "pages": [pg],
                "complete": False,
                "missing_sides": []
            })
            idx += 1

        # Validate sequence integrity: ensure pages within each doc are consecutive in page numbers
        for d in docs:
            # consider only integer page numbers for sequentiality checks
            page_nums = [p.get("page_number") for p in d["pages"] if isinstance(p.get("page_number"), int)]
            if not page_nums:
                # nothing to check
                self.logger.debug(f"Document {d.get('document_id')} has no valid integer page numbers; skipping sequential check.")
                continue
            if sorted(page_nums) != list(range(min(page_nums), max(page_nums) + 1)):
                self.logger.warning(f"Document {d['document_id']} pages are not sequential: {page_nums}")
                d["complete"] = False
                d.setdefault("issues", []).append("non_sequential_pages")

        return docs
    
    def validate_document_sequence(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Strict validation of a sequence of classified pages (per-page dicts).
        Returns a dict with one of statuses:
        - { 'status': 'rejected', 'reason': <str>, 'issues': [...], 'pages': [...] }
        - { 'status': 'incomplete', 'documents': [...], 'warnings': [...], 'pages': [...] }
        - { 'status': 'ok', 'documents': [...], 'warnings': [...], 'pages': [...] }

        The returned 'documents' array follows the same per-document format as group_pages_into_documents
        (document_id, document_type, pages, complete, missing_sides).
        """
        # Defensive copy & sort by page_number
        pages_copy = sorted(list(pages), key=lambda p: (p.get("page_number", 0) if isinstance(p.get("page_number", 0), int) else 0))
        issues: List[str] = []
        warnings: List[str] = []

        # Determine trusted pages according to threshold
        thr = float(getattr(self.cfg, "confidence_threshold", 0.0))
        trusted_pages: List[Dict[str, Any]] = []
        untrusted_pages: List[Dict[str, Any]] = []
        for p in pages_copy:
            conf = p.get("confidence", 0.0)
            dtype = p.get("document_type", "unknown")
            # trusted if not unknown and above threshold
            trusted = (dtype != "unknown") and isinstance(conf, (int, float)) and (conf >= thr)
            p["_trusted"] = bool(trusted)
            if trusted:
                trusted_pages.append(p)
            else:
                untrusted_pages.append(p)

        # If there are no trusted pages -> cannot classify reliably
        types_present = set([p.get("document_type") for p in trusted_pages if p.get("document_type") and p.get("document_type") != "unknown"])
        if not types_present:
            return {
                "status": "rejected",
                "reason": "unable_to_classify",
                "issues": ["no_trusted_pages"],
                "pages": pages_copy
            }

        # If multiple distinct document types => STRICT REJECT (mixed documents)
        non_unknown_types = [t for t in types_present if t != "unknown"]
        if len(non_unknown_types) > 1:
            # construct helpful issue details: which pages for each detected type
            type_map = {}
            for p in trusted_pages:
                dt = p.get("document_type", "unknown")
                type_map.setdefault(dt, []).append(p.get("page_number"))
            issues = [f"mixed_documents: detected types {sorted(list(non_unknown_types))} with pages {type_map}"]
            return {
                "status": "rejected",
                "reason": "mixed_documents",
                "issues": issues,
                "pages": pages_copy
            }

        # At this point, exactly one non-unknown type among trusted pages (call it dtype)
        dtype = next(iter(non_unknown_types))
        # Helper to create document structure (keeps same shape used elsewhere)
        def _make_doc(pages_list: List[Dict[str, Any]], doc_type: str):
            doc_id = _generate_doc_id(prefix=doc_type.upper() if doc_type else "DOC")
            pages_by_side = {p.get("side", "unknown"): p for p in pages_list}
            missing = []
            complete = True
            if doc_type == "aadhar":
                if "front" not in pages_by_side or "back" not in pages_by_side:
                    complete = False
                    if "front" not in pages_by_side: missing.append("front")
                    if "back" not in pages_by_side: missing.append("back")
            elif doc_type == "driving_license":
                # front-only accepted; completeness only if both present
                if "front" not in pages_by_side or "back" not in pages_by_side:
                    if "front" not in pages_by_side: missing.append("front")
                    if "back" not in pages_by_side: missing.append("back")
                    complete = False
            elif doc_type == "pan":
                # must be exactly one page and front
                if len(pages_list) != 1:
                    complete = False
                # if the single page is not pan_card_front, treat as invalid (complete=False)
                if len(pages_list) == 1 and pages_list[0].get("predicted_class") != "pan_card_front":
                    complete = False
            else:
                complete = False
            return {
                "document_id": doc_id,
                "document_type": doc_type,
                "pages": pages_list,
                "complete": complete,
                "missing_sides": missing
            }

        # Build document-level pages (use only trusted pages for deciding type and completeness)
        # However, include untrusted pages in the output pages list for auditing
        # We'll attempt to group trusted pages into a document according to dtype rules
        if dtype == "pan":
            # Valid only if exactly one trusted page of pan_card_front and no other trusted pages
            pan_trusted = [p for p in trusted_pages if p.get("document_type") == "pan"]
            # If other trusted pages exist (shouldn't, because we enforced single type), we would have rejected earlier.
            if len(pan_trusted) != 1:
                return {
                    "status": "rejected",
                    "reason": "invalid_pan_package",
                    "issues": [f"expected exactly 1 pan page but found {len(pan_trusted)} trusted pages"],
                    "pages": pages_copy
                }
            # ensure the page's predicted class is pan_card_front
            first = pan_trusted[0]
            if first.get("predicted_class") != "pan_card_front":
                return {
                    "status": "rejected",
                    "reason": "invalid_pan_side",
                    "issues": ["pan page must be pan_card_front"],
                    "pages": pages_copy
                }
            doc = _make_doc([first], "pan")
            return {
                "status": "ok" if doc["complete"] else "incomplete",
                "documents": [doc],
                "warnings": [] if doc["complete"] else ["pan_card_unexpected_page_count_or_side"],
                "pages": pages_copy
            }

        if dtype == "aadhar":
            # find front/back among trusted pages
            a_trusted = [p for p in trusted_pages if p.get("document_type") == "aadhar"]
            # group by side presence
            front_pages = [p for p in a_trusted if p.get("side") == "front"]
            back_pages = [p for p in a_trusted if p.get("side") == "back"]
            # If more than 2 pages or multiple fronts/backs suspicious -> reject
            if len(a_trusted) > 2 or len(front_pages) > 1 or len(back_pages) > 1:
                return {
                    "status": "rejected",
                    "reason": "aadhar_cardinality_mismatch",
                    "issues": [f"found multiple sides/pages for aadhar: fronts={len(front_pages)}, backs={len(back_pages)}"],
                    "pages": pages_copy
                }
            # Accept but mark incomplete if missing one side
            pages_for_doc = []
            # prefer ordering front then back using page_number if both present
            if front_pages:
                pages_for_doc.append(front_pages[0])
            if back_pages:
                pages_for_doc.append(back_pages[0])
            doc = _make_doc(pages_for_doc, "aadhar")
            if doc["complete"]:
                return {"status": "ok", "documents": [doc], "warnings": [], "pages": pages_copy}
            else:
                # include warning plus mention any untrusted pages that might help
                missing = doc["missing_sides"]
                warn_msg = f"aadhar_incomplete_missing_{'_'.join(missing)}"
                warnings.append(warn_msg)
                return {"status": "incomplete", "documents": [doc], "warnings": warnings, "pages": pages_copy}

        if dtype == "driving_license":
            dl_trusted = [p for p in trusted_pages if p.get("document_type") == "driving_license"]
            front_pages = [p for p in dl_trusted if p.get("side") == "front"]
            back_pages = [p for p in dl_trusted if p.get("side") == "back"]
            # Reject if multiple fronts/backs beyond 2 pages total
            if len(dl_trusted) > 2 or len(front_pages) > 1 or len(back_pages) > 1:
                return {
                    "status": "rejected",
                    "reason": "driving_license_cardinality_mismatch",
                    "issues": [f"found multiple sides/pages for driving_license: fronts={len(front_pages)}, backs={len(back_pages)}"],
                    "pages": pages_copy
                }
            pages_for_doc = []
            if front_pages:
                pages_for_doc.append(front_pages[0])
            if back_pages:
                pages_for_doc.append(back_pages[0])
            # Special rule: if only back present -> incomplete (as per your priority)
            doc = _make_doc(pages_for_doc, "driving_license")
            if doc["complete"]:
                return {"status": "ok", "documents": [doc], "warnings": [], "pages": pages_copy}
            else:
                # front-only (accepted but incomplete=False) OR back-only (incomplete)
                # We'll accept both scenarios but set status 'incomplete' if front missing
                if "front" not in doc["missing_sides"]:
                    # front present but back missing -> accept (incomplete)
                    warnings.append("driving_license_back_missing")
                else:
                    # front missing (back-only) -> incomplete and warn
                    warnings.append("driving_license_front_missing")
                return {"status": "incomplete", "documents": [doc], "warnings": warnings, "pages": pages_copy}

        # If we arrive here, dtype was unexpected
        return {
            "status": "rejected",
            "reason": "unknown_document_type",
            "issues": [f"unsupported document type detected: {dtype}"],
            "pages": pages_copy
        }

    # --------------------------
    # High-level pipeline: classify -> group -> output
    # --------------------------
    def classify_and_aggregate(self, preprocessed_pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        End-to-end helper: performs per-page classification, strict validation and document-level aggregation.
        Returns structure depending on validation:
        - If rejected: { 'status': 'rejected', 'reason': ..., 'issues': [...], 'pages': [...], 'metrics': {...} }
        - If incomplete: { 'status': 'incomplete', 'documents': [...], 'warnings': [...], 'pages': [...], 'metrics': {...} }
        - If ok: { 'status': 'ok', 'documents': [...], 'pages': [...], 'metrics': {...} }
        """
        start_total = time.perf_counter()
        pages_out = self.classify_pages(preprocessed_pages)

        # compute raw metrics early
        confidences = [p.get("confidence", 0.0) for p in pages_out if isinstance(p.get("confidence", 0.0), (int, float))]
        avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
        metrics = {
            "num_pages": len(pages_out),
            "num_documents": 0,  # will fill later
            "avg_confidence": avg_conf,
            "min_confidence": min(confidences) if confidences else 0.0,
            "max_confidence": max(confidences) if confidences else 0.0,
            "total_time_s": None  # fill at end
        }

        # Strict validation step
        validation = self.validate_document_sequence(pages_out)

        # If rejected, return explicit structure (do not continue grouping or downstream writes)
        if validation.get("status") == "rejected":
            metrics["total_time_s"] = round(time.perf_counter() - start_total, 4)
            result = {
                "status": "rejected",
                "reason": validation.get("reason"),
                "issues": validation.get("issues", []),
                "pages": validation.get("pages", pages_out),
                "documents": [],
                "metrics": metrics
            }
            return result

        # If incomplete or ok, use the provided documents (validation returns documents)
        documents = validation.get("documents", [])
        warnings = validation.get("warnings", [])

        metrics["num_documents"] = len(documents)
        metrics["total_time_s"] = round(time.perf_counter() - start_total, 4)

        # Return appropriate status label
        status = validation.get("status", "ok")
        out = {
            "status": status,
            "pages": validation.get("pages", pages_out),
            "documents": documents,
            "warnings": warnings,
            "metrics": metrics
        }
        return out

# ------------------------------
# Example usage (to be removed in production import)
# ------------------------------
if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Document Classifier CLI")
    parser.add_argument("--input-json", required=True, help="Path to JSON file containing preprocessed pages list (image arrays not stored in JSON).")
    parser.add_argument("--model-path", help="Optional model path to override config")
    parser.add_argument("--preview", action="store_true", help="Print summaries")
    args = parser.parse_args()

    # NOTE: CLI here assumes the user created a JSON that references saved numpy files or similar.
    # For actual pipeline, call classify_and_aggregate() with in-memory preprocessed pages from your preprocessor.
    cfg = ClassifierConfig()
    if args.model_path:
        cfg.model_path = args.model_path

    classifier = DocumentClassifier(cfg)

    # Minimal demonstration scaffolding: load placeholder JSON describing page metadata
    in_path = Path(args.input_json)
    if not in_path.exists():
        print("Input JSON not found:", in_path)
        raise SystemExit(1)

    with open(in_path, "r", encoding="utf-8") as f:
        described = json.load(f)

    # described should be a list of dicts where each dict may have a field "npy_path" for the image_array
    pages = []
    for item in described:
        # Expected fields: 'npy_path' (path to .npy image array) and 'metadata'
        npy_path = item.get("npy_path")
        metadata = item.get("metadata", {})
        if not npy_path:
            continue
        arr = np.load(npy_path)
        pages.append({"image_array": arr, "metadata": metadata})

    out = classifier.classify_and_aggregate(pages)
    if args.preview:
        print("Metrics:", out["metrics"])
        for d in out["documents"]:
            print(f"Doc {d['document_id']} type={d['document_type']} complete={d['complete']} pages={[p['page_number'] for p in d['pages']]}")
    # Save results JSON
    out_path = in_path.with_suffix(".classified.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, default=lambda o: "<non-serializable>", indent=2)
    print("Saved classification output to", out_path)
