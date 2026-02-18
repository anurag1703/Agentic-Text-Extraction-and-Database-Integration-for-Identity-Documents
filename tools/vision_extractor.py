# vision_extractor.py
from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import dataclasses
import functools
import hashlib
import io
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import aiohttp     # async HTTP client (preferred)
import requests    # sync fallback for environments without aiohttp
import cv2
from datetime import datetime


# -----------------------
# Exception hierarchy
# -----------------------
class VisionExtractorError(Exception): pass
class OllamaConnectionError(VisionExtractorError): pass
class ExtractionTimeoutError(VisionExtractorError): pass
class JSONParseError(VisionExtractorError): pass
class LowConfidenceError(VisionExtractorError): pass
class DocumentSchemaError(VisionExtractorError): pass
class InvalidInputError(VisionExtractorError): pass


# -----------------------
# Config
# -----------------------
@dataclass
class ExtractorConfig:
    ollama_endpoint: str = "http://localhost:11434/api/generate"
    model_name: str = "qwen2.5vl:3b"
    timeout_seconds: int = 360
    max_retries: int = 4
    retry_delay: float = 1.0  # base delay in seconds
    batch_size: int = 2
    confidence_threshold: float = 0.7
    enable_field_validation: bool = True
    enable_async_processing: bool = True
    max_concurrent_requests: int = 1
    # limits for images
    jpeg_quality: int = 70   # when encoding to JPEG for embedding
    max_image_side: int = 384  # scale down larger images to preserve memory

    # logging
    enable_logging: bool = True
    log_level: int = logging.INFO


# -----------------------
# Prompt templates & schemas
# -----------------------
AADHAR_FRONT_FIELDS = {
    "uid_number": "12-digit UID number",
    "name": "Full name in English",
    "date_of_birth": "DD/MM/YYYY format",
    "gender": "MALE/FEMALE/OTHER"
}

AADHAR_BACK_FIELDS = {
    "address": "Complete address in English",
    "pin_code": "6-digit PIN code",
    "state": "State name"
}

PAN_FIELDS = {
    "pan_number": "10-character PAN number",
    "name": "Full name as on PAN",
    "fathers_name": "Father's name",
    "date_of_birth": "DD/MM/YYYY format",
    "signature_present": "Boolean for signature visibility"
}

DL_FRONT_FIELDS = {
    "dl_number": "Driving License number",
    "name": "Holder's full name",
    "fathers_name": "Father's/Husband's name",
    "date_of_birth": "DD/MM/YYYY",
    "address": "Current address",
    "issue_date": "License issue date",
    "valid_until": "Expiry date"
}


PROMPT_TEMPLATES = {
    "aadhar_front": """CRITICAL: You MUST return ONLY a valid JSON object BETWEEN THE MARKERS BEGIN_JSON and END_JSON. No additional text, commentary, or keys — EXACTLY one JSON object matching the schema below.

EXTRACTION TASK: Extract Aadhar FRONT side information with surgical accuracy.

REQUIRED JSON SCHEMA:
{
  "uid_number": "12-digit UID number or null",
  "name": "Full name in English or null",
  "date_of_birth": "DD/MM/YYYY format or null",
  "gender": "MALE/FEMALE/OTHER or null"
}

STRICT RULES (read carefully and follow exactly):
1) Return text EXACTLY as printed on the card for 'name' and 'uid_number' (preserve case, spacing, and punctuation). Do NOT transliterate, normalize casing, or alter characters.
2) Dates must be returned only in **DD/MM/YYYY** format. If you cannot parse a date with high confidence, return null.
3) UID (uid_number) must be exactly 12 digits (no spaces) when non-null. If masked (e.g., XXXX XXXX 1234) return the masked form exactly as printed.
4) Gender must be one of MALE, FEMALE, OTHER, or null. Do not invent beyond these values.
5) If a field is ambiguous or there are multiple plausible readings, return null — do not guess.
6) Allowed conservative OCR fixes: single-character swaps from the set {O↔0, I↔1, S↔5, B↔8, Z↔2, G↔6}. Apply only if the fix yields a syntactically valid UID or date and is unambiguous.
7) If you are supplied a `previous_extraction` JSON (in the request payload), treat it as a reference: prefer prior non-null values unless the current image provides a clearly superior read. Do NOT change previous values unless current reading is clearly better — do NOT make arbitrary edits.
8) Do not output any additional keys beyond the schema.

QUALITY CHECK (must be satisfied before emitting JSON):
- All date fields are either DD/MM/YYYY or null.
- uid_number (if non-null) is exactly 12 digits or a masked 12-char-like pattern.
- Names are returned exactly as on the image or null.
- No additional keys present.

BEGIN_JSON
{"uid_number":null,"name":null,"date_of_birth":null,"gender":null}
END_JSON
""",

    "aadhar_back": """CRITICAL: You MUST return ONLY a valid JSON object BETWEEN THE MARKERS BEGIN_JSON and END_JSON. No additional text, commentary, or keys — EXACTLY one JSON object matching the schema below.

EXTRACTION TASK: Extract Aadhar BACK side information with surgical accuracy.

REQUIRED JSON SCHEMA:
{
  "address": "Complete address in English or null",
  "pin_code": "6-digit PIN code or null",
  "state": "State name in full or null"
}

STRICT RULES:
1) Return address text EXACTLY as printed (do not normalize or trim meaningful punctuation). If address spans lines, return the line breaks as a single space unless the system expects explicit newline chars.
2) PIN code must be exactly 6 digits (no spaces) if non-null.
3) State must be the full state name (e.g., 'Maharashtra' not 'MH') or null.
4) If ambiguous for any field, return null.
5) Allowed conservative OCR fixes: only single-character swaps from {O↔0, I↔1, S↔5, B↔8, Z↔2, G↔6}, only if unambiguous.
6) If `previous_extraction` is provided, prefer non-null prior values unless the current image clearly shows a superior read.
7) Do not add any extra keys beyond the schema.

QUALITY CHECK:
- pin_code either null or 6 digits.
- address preserved exactly or null.
- state full name or null.
- No extra keys.

BEGIN_JSON
{"address":null,"pin_code":null,"state":null}
END_JSON
""",

    "pan_card": """CRITICAL: You MUST return ONLY a valid JSON object BETWEEN THE MARKERS BEGIN_JSON and END_JSON. No additional text, commentary, or keys — EXACTLY one JSON object matching the schema below.

EXTRACTION TASK: Extract PAN card information with surgical accuracy.

REQUIRED JSON SCHEMA:
{
  "pan_number": "10-character PAN number or null",
  "name": "Full name as on PAN or null",
  "fathers_name": "Father's name or null",
  "date_of_birth": "DD/MM/YYYY format or null",
  "signature_present": true/false/null
}

STRICT RULES:
1) pan_number must be returned exactly as on the card (preserve letters/digits and positions) OR null. Valid format is 5 letters + 4 digits + 1 letter (e.g., ABCDE1234F) OR masked forms (XXXXX1234X). If masked, return the mask exactly as printed.
2) If you can correct a single-character OCR confusion using the conservative set {O↔0, I↔1, S↔5, B↔8, Z↔2, G↔6} and this resolves format & checksum unambiguously, you may return a corrected pan_number; otherwise return null.
3) Dates must be DD/MM/YYYY or null. If you parse a date, normalize to DD/MM/YYYY.
4) signature_present must be boolean true/false or null (true if signature is clearly visible).
5) Names must be returned exactly as printed or null.
6) If `previous_extraction` is supplied, prefer non-null prior values unless the current image is clearly better. If you change a prior value, do not alter its formatting beyond replacing the entire field string.
7) If ambiguous, return null (do not guess).

QUALITY CHECK:
- pan_number matches allowed formats or null.
- date_of_birth is DD/MM/YYYY or null.
- signature_present is true/false/null.
- No extra keys.

BEGIN_JSON
{"pan_number":null,"name":null,"fathers_name":null,"date_of_birth":null,"signature_present":null}
END_JSON
""",

    "driving_license_front": """
CRITICAL: Return ONLY a valid JSON object BETWEEN BEGIN_JSON and END_JSON — no extra text, comments, or keys. 
Extract the printed details from the FRONT side of an Indian Driving Licence card.

GOAL:
Accurately extract the textual content exactly as printed on the card for each field in the schema below. 
Prioritize fields that appear in the **visual inspection zone** (text area containing DL number, holder details, address, and validity dates).

REQUIRED JSON SCHEMA:
{
  "dl_number": "Driving Licence number or null",
  "name": "Holder's full name or null",
  "fathers_name": "Father’s/Husband’s name or null",
  "date_of_birth": "DD/MM/YYYY or null",
  "address": "Current address or null",
  "issue_date": "Issue date in DD/MM/YYYY or null",
  "valid_until": "Expiry date in DD/MM/YYYY or null"
}

STRICT EXTRACTION RULES:
1. Extract text **exactly as printed** — preserve capitalization, punctuation, and spacing. Never reformat, correct, or normalize names or addresses.
2. Use DD/MM/YYYY format for all dates. If any date is unclear, partially cut off, or ambiguous, return null.
3. For the DL number, copy it exactly as printed (including spaces or hyphens). Only apply minimal OCR correction (O↔0, I↔1, S↔5, B↔8, Z↔2, G↔6) **if and only if** it produces a valid DL number pattern with no ambiguity.
4. Prefer clear, high-confidence text near the DL number and photo region for personal details (Name, Father’s Name, DOB). Prefer text blocks labeled “Address” or starting with “S/O”, “D/O”, “W/O” for the address field.
5. If multiple versions of a field are visible (e.g., duplicate prints), choose the **clearest and most complete** one. If unclear, return null.
6. Use any provided `previous_extraction` as a trusted fallback: keep previously non-null values unless the new image shows a clearly different, higher-confidence reading.
7. Output only the specified keys, with null for missing or uncertain fields.

QUALITY CHECK BEFORE RETURN:
- Dates are DD/MM/YYYY or null.
- dl_number matches printed layout exactly or null.
- No extra or missing keys.
- No commentary or metadata.

BEGIN_JSON
{"dl_number":null,"name":null,"fathers_name":null,"date_of_birth":null,"address":null,"issue_date":null,"valid_until":null}
END_JSON
"""

}

# Confidence stratification
CONFIDENCE_LEVELS = {
    'HIGH': 0.8,
    'MEDIUM': 0.6,
    'LOW': 0.4,
    'REJECT': 0.0
}


# -----------------------
# Logging helper
# -----------------------
def _setup_logger(cfg: ExtractorConfig) -> logging.Logger:
    logger = logging.getLogger("VisionExtractor")
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = "[VisionExtractor] %(asctime)s %(levelname)s - %(message)s"
        h.setFormatter(logging.Formatter(fmt))
        logger.addHandler(h)
    logger.setLevel(cfg.log_level if cfg.enable_logging else logging.CRITICAL)
    return logger


# -----------------------
# Utilities: image encoding + resizing
# -----------------------
def _rgb_numpy_to_jpeg_b64(arr: np.ndarray, quality: int = 70, max_side: Optional[int] = None) -> str:
    """
    Convert H,W,3 RGB numpy array to JPEG base64 string.
    Resize so longer side <= max_side if provided.
    """
    if arr is None:
        raise ValueError("No image array provided")
    img = arr.copy()
    # ensure uint8
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    # convert RGB to BGR for cv2
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if max_side:
        h, w = bgr.shape[:2]
        longer = max(h, w)
        if longer > max_side:
            scale = max_side / float(longer)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    success, enc = cv2.imencode(".jpg", bgr, encode_params)
    if not success:
        raise ValueError("Failed to encode image to JPEG")
    b64 = base64.b64encode(enc.tobytes()).decode("ascii")
    return b64


# -----------------------
# Field validators & utilities
# -----------------------
# Aadhaar number: supports 4-4-4 or masked formats (e.g. "XXXX XXXX 1234" or "1234 5678 9123")
_RE_UID = re.compile(
    r"\b(?:\d{4}\s?\d{4}\s?\d{4}|\d{12}|X{0,4}\s?X{0,4}\s?\d{4})\b"
)

# Indian PIN code: 6-digit number (kept same)
_RE_PIN = re.compile(r"\b\d{6}\b")

# PAN number: standard + masked formats like "ABCDE1234F" or "XXXXX1234X"
_RE_PAN = re.compile(
    r"\b(?:[A-Z]{5}\d{4}[A-Z]|X{5}\d{4}[A-Z]|[A-Z]{5}\d{4}X)\b",
    re.IGNORECASE
)

# Date: DD/MM/YYYY or DD-MM-YYYY (kept same)
_RE_DATE_DMY = re.compile(
    r"\b(0[1-9]|[12]\d|3[01])[/\-](0[1-9]|1[0-2])[/\-](\d{4})\b"
)


def _parse_date_dmy(value: str) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None
    m = _RE_DATE_DMY.match(value.strip())
    if not m:
        # try to parse common variants with datetime
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                d = datetime.strptime(value.strip(), fmt)
                return d.strftime("%d/%m/%Y")
            except Exception:
                continue
        return None
    # normalize to DD/MM/YYYY
    day, month, year = m.groups()[0], m.groups()[1], m.groups()[2]
    try:
        d = datetime(int(year), int(month), int(day))
        return d.strftime("%d/%m/%Y")
    except Exception:
        return None


def _validate_field(field_name: str, value: Any) -> Tuple[bool, Optional[str]]:
    """
    Returns (is_valid, normalized_value_or_none). Normalization applied when helpful.
    """
    if value is None:
        return False, None
    s = str(value).strip()
    if field_name == "uid_number":
        ok = bool(_RE_UID.match(s))
        return ok, s if ok else None
    if field_name == "pin_code":
        ok = bool(_RE_PIN.match(s))
        return ok, s if ok else None
    if field_name == "pan_number":
        s_up = re.sub(r"[^A-Za-z0-9]", "", s.upper())  # normalize
        # 1️⃣ Strict regex first
        if re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", s_up):
            return True, s_up
        # 2️⃣ If 11 chars, attempt duplicate-removal fix even if confidence unknown
        if len(s_up) == 11:
            corrected, debug_info = _attempt_safe_pan_auto_correction(s_up, logger=None, min_confidence_for_autofix=0.0)
            if corrected:
                return True, corrected
        # 3️⃣ Try OCR confusion correction as fallback
        corrected, debug_info = _attempt_safe_pan_auto_correction(s_up, logger=None, min_confidence_for_autofix=0.0)
        if corrected:
            return True, corrected
        return False, None
    if field_name in ("date_of_birth", "issue_date", "valid_until"):
        parsed = _parse_date_dmy(s)
        return (parsed is not None), parsed
    if field_name == "signature_present":
        s_low = s.lower()
        if s_low in ("true", "yes", "y", "1"):
            return True, True
        if s_low in ("false", "no", "n", "0"):
            return True, False
        # try boolean already
        if isinstance(value, bool):
            return True, value
        return False, None
    # names, address, state, dl_number etc: sanity checks
    if field_name in ("name", "fathers_name", "address", "state", "dl_number"):
        if len(s) == 0:
            return False, None
        return True, s
    # default: accept
    return True, s


def _attempt_safe_pan_auto_correction(raw_value: str,
                                      logger: Optional[logging.Logger] = None,
                                      confidence: Optional[float] = None,
                                      min_confidence_for_autofix: float = 0.0) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Try to auto-correct OCR/LLM misreads for PAN numbers safely.

    - Accepts 'raw_value' as extracted from model (string)
    - Optionally uses model confidence to decide whether to auto-fix
    - Returns (corrected_pan_or_None, debug_info)

    Safeguards:
      1. Only attempts correction if raw_value length ∈ {10, 11}
      2. For 11-char strings, removes one char at a time and tests if the result matches strict PAN pattern
      3. If >1 valid 10-char candidates → treat as ambiguous (no auto-fix)
      4. If model confidence < threshold → don't auto-fix (only suggest)
      5. Always returns debug_info dict for audit/logging
    """

    debug = {
        "original": raw_value,
        "attempted": None,
        "candidates": [],
        "chosen": None,
        "ambiguous": False,
        "auto_applied": False,
        "reason": None
    }

    if not raw_value or not isinstance(raw_value, str):
        debug["reason"] = "empty_or_non_string"
        return None, debug

    s = re.sub(r"[^A-Za-z0-9]", "", raw_value).upper().strip()
    debug["attempted"] = s

    # canonical PAN pattern
    pan_re = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")

    # already valid → no correction needed
    if pan_re.fullmatch(s):
        debug["reason"] = "already_valid"
        return s, debug

    # only handle plausible lengths
    if len(s) not in (10, 11):
        debug["reason"] = f"unsupported_length_{len(s)}"
        return None, debug

    # attempt single-character removal only for 11-char candidates
    if len(s) == 11:
        valid_cands = []
        for i in range(len(s)):
            cand = s[:i] + s[i+1:]
            if pan_re.fullmatch(cand):
                valid_cands.append(cand)
        debug["candidates"] = valid_cands

        if not valid_cands:
            debug["reason"] = "no_valid_candidate_after_removal"
            return None, debug
        if len(valid_cands) > 1:
            debug["ambiguous"] = True
            debug["reason"] = "multiple_valid_candidates"
            return None, debug

        corrected = valid_cands[0]
        debug["chosen"] = corrected

        # apply confidence guard
        if confidence is not None and confidence < min_confidence_for_autofix:
            debug["reason"] = f"low_confidence_{confidence:.3f}"
            return None, debug

        debug["auto_applied"] = True
        debug["reason"] = "single_valid_candidate_autofixed"
        if logger:
            logger.info(f"AUTO-FIXED PAN: {raw_value!r} → {corrected!r}")
        return corrected, debug

    # if length == 10 but invalid, try OCR confusion fix (I↔1, O↔0, etc.)
    letter_to_digit = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8'}
    digit_to_letter = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B'}
    pan_chars = list(s)
    for i in range(0, 5):
        if pan_chars[i].isdigit() and pan_chars[i] in digit_to_letter:
            pan_chars[i] = digit_to_letter[pan_chars[i]]
    for i in range(5, 9):
        if not pan_chars[i].isdigit() and pan_chars[i] in letter_to_digit:
            pan_chars[i] = letter_to_digit[pan_chars[i]]
    fixed = "".join(pan_chars)
    if pan_re.fullmatch(fixed):
        debug["chosen"] = fixed
        debug["auto_applied"] = True
        debug["reason"] = "ocr_confusion_fix_applied"
        if logger:
            logger.info(f"OCR-FIXED PAN: {raw_value!r} → {fixed!r}")
        return fixed, debug

    debug["reason"] = "no_fix_successful"
    return None, debug


# -----------------------
# Ollama client - async with retries + backoff
# -----------------------
class OllamaClient:
    def __init__(self, cfg: ExtractorConfig, logger: logging.Logger):
        """
        Initialize OllamaClient metadata only. Do NOT create aiohttp Connector/Session here
        to avoid 'no running event loop' errors when instantiated from synchronous code.
        """
        self.cfg = cfg
        self.logger = logger

        # concurrency control (keeps previous behavior)
        self.semaphore = asyncio.Semaphore(cfg.max_concurrent_requests)

        # store timeout/connector parameters for lazy creation
        self._connect_timeout = getattr(cfg, "connect_timeout", 20.0)
        self._read_timeout = getattr(cfg, "read_timeout", 360.0)
        self._total_timeout = getattr(cfg, "total_timeout", None)

        # build aiohttp.ClientTimeout instance to reuse later
        # (don't instantiate session now; only prepare timeout object)
        self._aio_timeout = aiohttp.ClientTimeout(
            total=self._total_timeout,
            sock_connect=self._connect_timeout,
            sock_read=self._read_timeout,
        )

        # connector/session will be created lazily inside _ensure_session()
        self._connector_limit = max(10, cfg.max_concurrent_requests * 2)
        self._keepalive = getattr(cfg, "keepalive_timeout", 60.0)
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._aio_session: Optional[aiohttp.ClientSession] = None

        # sync requests.Session (created lazily when generate_sync runs)
        self._sync_session: Optional[requests.Session] = None

        # small flag to track manual closing if needed
        self._session_closed = False

    async def _ensure_session(self):
        """
        Lazily create aiohttp Connector and ClientSession inside a running event loop.
        This avoids RuntimeError when the class is instantiated from synchronous code.
        Idempotent — safe to call multiple times.
        """
        # If already have a usable session, return quickly
        if self._aio_session is not None and not self._aio_session.closed:
            return

        # Must be running inside an event loop to create Connector/Session safely
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as e:
            # No running loop — re-raise with clearer message for debugging
            raise RuntimeError("Attempted to initialize aiohttp session without a running event loop") from e

        # Create connector and session now that a loop is running
        self.logger.debug("Creating aiohttp TCPConnector and ClientSession inside running loop")
        self._connector = aiohttp.TCPConnector(
            limit=self._connector_limit,
            keepalive_timeout=self._keepalive,
            force_close=False
        )
        self._aio_session = aiohttp.ClientSession(connector=self._connector, timeout=self._aio_timeout)

    async def close(self):
        """
        Gracefully close both async and sync sessions if present.
        Safe to call multiple times.
        """
        # close aiohttp session if present
        try:
            if self._aio_session is not None and not self._aio_session.closed:
                await self._aio_session.close()
                self._aio_session = None
        except Exception as e:
            self.logger.debug("Error closing aiohttp session: %s", e)

        # close sync requests session if present
        try:
            if hasattr(self, "_sync_session") and self._sync_session is not None:
                try:
                    self._sync_session.close()
                except Exception:
                    pass
                self._sync_session = None
        except Exception as e:
            self.logger.debug("Error closing sync requests session: %s", e)

        self._connector = None
        self._session_closed = True

    async def generate(self, prompt: str, image_b64: Optional[str] = None) -> Dict[str, Any]:
        """
        Async generate using Ollama's native image format
        """
        # === BEGIN PATCH: total elapsed timer (async generate) ===
        start_total = time.perf_counter()
        # We'll measure total time across retries and include it in final error/logs.
        # Ensure a local var to report elapsed when raising.
        total_elapsed = None
        
        await self._ensure_session()
        attempt = 0
        last_exc = None

        while attempt < self.cfg.max_retries:
            attempt += 1
            try:
                async with self.semaphore:
                    # Build payload with native image format
                    body = {
                        "model": self.cfg.model_name,
                        "prompt": prompt,
                        "max_tokens": getattr(self.cfg, "max_tokens", 1024),
                        "temperature": getattr(self.cfg, "temperature", 0.0),
                        "stop": ["END_JSON"],
                        "stream": False
                    }
                    
                    # Add images in native Ollama format
                    if image_b64:
                        body["images"] = [image_b64]

                    self.logger.debug("Ollama request (attempt %d) model=%s", attempt, self.cfg.model_name)

                    req_timeout = getattr(self, "_aio_timeout", None)
                    async with self._aio_session.post(
                        self.cfg.ollama_endpoint, 
                        json=body, 
                        timeout=req_timeout
                    ) as resp:
                        if resp.status != 200:
                            txt = await resp.text()
                            raise OllamaConnectionError(f"Ollama HTTP {resp.status}: {txt[:500]}")

                        # Parse response
                        response_data = await resp.json()

                        self.logger.debug("Ollama response preview: %s", str(response_data)[:200] + "..." if len(str(response_data)) > 200 else str(response_data))

                        # Try to extract JSON directly from the response data
                        try:
                            parsed = _extract_json_from_model_response(response_data)
                            return parsed
                        except JSONParseError as e:
                            # If JSON extraction fails, try to extract from response field
                            if "response" in response_data:
                                try:
                                    parsed = _extract_json_from_model_response(response_data["response"])
                                    return parsed
                                except JSONParseError:
                                    pass
                            # Return the raw response data for debugging
                            return response_data

            except (aiohttp.ClientError, asyncio.TimeoutError, OllamaConnectionError) as e:
                last_exc = e
                self.logger.warning("Ollama attempt %d failed: %s", attempt, repr(e))
                if attempt >= self.cfg.max_retries:
                    break
                backoff = self.cfg.retry_delay * (2 ** (attempt - 1))
                await asyncio.sleep(backoff)
            except Exception as e:
                last_exc = e
                self.logger.exception("Unexpected error while contacting Ollama on attempt %d", attempt)
                if attempt >= self.cfg.max_retries:
                    break
                await asyncio.sleep(self.cfg.retry_delay * (2 ** (attempt - 1)))

        total_elapsed = time.perf_counter() - start_total
        self.logger.error("Ollama generate total elapsed time after %d attempts: %.3f s", attempt, total_elapsed)
        raise OllamaConnectionError(f"Ollama generate failed after {self.cfg.max_retries} attempts: last_exc={repr(last_exc)}")

    def generate_sync(self, prompt: str, image_b64: Optional[str] = None) -> Dict[str, Any]:
        """
        Sync generate with URL-detection fallback (single focused retry if model returns a URL).
        """
        # === BEGIN PATCH: total elapsed timer (sync generate) ===
        start_total = time.perf_counter()
        total_elapsed = None
        # === END PATCH ===

        attempt = 0
        last_exc = None

        # Simplify wrapper instructions
        wrapper_instructions = (
            "\n\nIMPORTANT: Return ONLY the JSON object between BEGIN_JSON and END_JSON. "
            "No other text, explanations, or formatting."
        )

        def looks_like_url(s: str) -> bool:
            if not s:
                return False
            s = s.strip()
            if re.fullmatch(r"https?://\S+", s):
                return True
            if "http://" in s or "https://" in s:
                if len(s) < 400:
                    return True
            return False

        # ensure persistent sync session
        if not hasattr(self, "_sync_session") or self._sync_session is None:
            self._sync_session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=0, pool_block=False)
            self._sync_session.mount("http://", adapter)
            self._sync_session.mount("https://", adapter)

        connect_timeout = getattr(self.cfg, "connect_timeout", 20.0)
        read_timeout = getattr(self.cfg, "read_timeout", 360.0)
        request_timeout = (connect_timeout, read_timeout)

        while attempt < self.cfg.max_retries:
            attempt += 1
            try:
                payload_prompt = prompt
                if image_b64:
                    datauri = f"data:image/jpeg;base64,{image_b64}"
                    payload_prompt = f"{prompt}\n\n[IMAGE_START]{datauri}[IMAGE_END]"
                payload_prompt = payload_prompt + wrapper_instructions

                body = {"model": self.cfg.model_name, "prompt": payload_prompt, "max_tokens": 1024, "temperature": 0.0}

                self.logger.debug("Ollama sync request (attempt %d) model=%s endpoint=%s", attempt, self.cfg.model_name, self.cfg.ollama_endpoint)

                resp = self._sync_session.post(self.cfg.ollama_endpoint, json=body, timeout=request_timeout, stream=True)
                if resp.status_code != 200:
                    raise OllamaConnectionError(f"HTTP {resp.status_code}: {resp.text[:500]}")

                response_fragments: List[str] = []
                buffer = ""

                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if not line:
                        continue
                    buffer += line + "\n"
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, dict):
                            if "response" in parsed and parsed["response"] is not None:
                                response_fragments.append(str(parsed["response"]))
                            elif "text" in parsed and parsed["text"] is not None:
                                response_fragments.append(str(parsed["text"]))
                            else:
                                return parsed
                    except Exception:
                        matches = re.findall(r"(\{.*?\})(?=\s*\{|\s*$)", buffer, flags=re.DOTALL)
                        if matches:
                            for obj_txt in matches:
                                try:
                                    parsed = json.loads(obj_txt)
                                    if isinstance(parsed, dict):
                                        if "response" in parsed and parsed["response"] is not None:
                                            response_fragments.append(str(parsed["response"]))
                                        elif "text" in parsed and parsed["text"] is not None:
                                            response_fragments.append(str(parsed["text"]))
                                        else:
                                            return parsed
                                except Exception:
                                    continue
                            buffer = ""
                        continue

                concatenated = "".join(response_fragments).strip()
                if not concatenated:
                    concatenated = buffer.strip()

                if concatenated:
                    self.logger.debug("Ollama sync raw response preview (len=%d): %s", len(concatenated), (concatenated[:200] + "...") if len(concatenated) > 200 else concatenated)

                    if looks_like_url(concatenated):
                        self.logger.warning("Ollama returned URL-like response; performing one focused JSON-only retry (no URLs).")
                        schema_hint = ""
                        if "PAN" in payload_prompt.upper() or "pan" in payload_prompt:
                            schema_hint = '{"pan_number":"","name":"","fathers_name":"","date_of_birth":"","signature_present":false}'
                        elif "AADHAR" in payload_prompt.upper() or "aadhar" in payload_prompt:
                            schema_hint = '{"uid_number":"","name":"","date_of_birth":"","gender":""}'
                        elif "DRIVING" in payload_prompt.upper() or "driving" in payload_prompt.lower():
                            schema_hint = '{"dl_number":"","name":"","fathers_name":"","date_of_birth":"","address":""}'

                        focused = (
                            "DO NOT RETURN ANY URL OR LINK. Ignore any image links. "
                            "Return ONLY the JSON object (no explanation) between markers BEGIN_JSON and END_JSON.\n"
                        )
                        if schema_hint:
                            focused += f"Expected JSON schema keys example: {schema_hint}\n"
                        focused += "BEGIN_JSON\n{ }\nEND_JSON\n"

                        retry_prompt = focused + "\nOriginal request (for reference):\n" + prompt
                        retry_body = {"model": self.cfg.model_name, "prompt": retry_prompt, "max_tokens": 1024, "temperature": 0.0}
                        r2 = self._sync_session.post(self.cfg.ollama_endpoint, json=retry_body, timeout=request_timeout)
                        if r2.status_code != 200:
                            raise OllamaConnectionError(f"HTTP {r2.status_code}: {r2.text[:500]}")
                        buf2 = r2.text
                        m = re.search(r"BEGIN_JSON\s*(\{.*?\})\s*END_JSON", buf2, flags=re.DOTALL | re.IGNORECASE)
                        if m:
                            try:
                                return json.loads(m.group(1))
                            except Exception:
                                pass
                        jmatch = re.search(r"(\{.*\})", buf2, flags=re.DOTALL)
                        if jmatch:
                            try:
                                return json.loads(jmatch.group(1))
                            except Exception:
                                pass
                        return {"text": buf2.strip()[:2000]}

                    # Normal path: try marker extraction then full parse
                    m = re.search(r"BEGIN_JSON\s*(\{.*?\})\s*END_JSON", concatenated, flags=re.DOTALL | re.IGNORECASE)
                    if m:
                        try:
                            return json.loads(m.group(1))
                        except Exception:
                            repaired = re.sub(r",\s*(\}|])", r"\1", m.group(1))
                            try:
                                return json.loads(repaired)
                            except Exception:
                                pass
                    try:
                        return json.loads(concatenated)
                    except Exception:
                        mm = re.search(r"(\{.*\})", concatenated, flags=re.DOTALL)
                        if mm:
                            try:
                                return json.loads(mm.group(1))
                            except Exception:
                                pass
                        return {"text": concatenated}

                # fallback parsing of buffer
                buf_trim = buffer.strip()
                if buf_trim:
                    try:
                        return json.loads(buf_trim)
                    except Exception:
                        m = re.search(r"(\{.*\})", buf_trim, flags=re.DOTALL)
                        if m:
                            try:
                                return json.loads(m.group(1))
                            except Exception:
                                raise JSONParseError(f"Failed to parse Ollama response as JSON: {buf_trim[:1000]}")
                return {}
            except (requests.RequestException, OllamaConnectionError, JSONParseError) as e:
                last_exc = e
                self.logger.warning("Ollama sync attempt %d failed: %s", attempt, repr(e))
                if attempt >= self.cfg.max_retries:
                    break
                backoff = self.cfg.retry_delay * (2 ** (attempt - 1)) + (0.1 * attempt)
                time.sleep(backoff)
            except Exception as e:
                last_exc = e
                self.logger.exception("Unexpected error contacting Ollama sync")
                if attempt >= self.cfg.max_retries:
                    break
                time.sleep(self.cfg.retry_delay * (2 ** (attempt - 1)))
        total_elapsed = time.perf_counter() - start_total
        self.logger.error("Ollama sync generate total elapsed time after %d attempts: %.3f s", attempt, total_elapsed)
        raise OllamaConnectionError(f"Ollama generate_sync failed after {self.cfg.max_retries} attempts: last_exc={repr(last_exc)}")


# -----------------------
# Parser for model responses
# -----------------------
def _extract_json_from_model_response(resp: Any) -> Dict[str, Any]:
    """
    Robust JSON extractor from model output.
    
    Handles cases where:
    1. Response is already a valid JSON object (most common case)
    2. Response contains JSON between BEGIN_JSON/END_JSON markers
    3. Response is wrapped in markdown code blocks (```json ... ```)
    4. Response has a 'text' field containing the JSON
    """
    import ast

    if resp is None:
        raise JSONParseError("Empty model response")

    def strip_code_fences(s: str) -> str:
        # remove fenced code blocks ```...``` and inline backticks
        s = re.sub(r"```(?:json)?\s*(.*?)```", r"\1", s, flags=re.DOTALL | re.IGNORECASE)
        s = re.sub(r"`([^`]+)`", r"\1", s)
        return s.strip()

    def extract_between_markers(s: str) -> Optional[str]:
        # Look for BEGIN_JSON ... END_JSON or similar markers
        m = re.search(r"BEGIN_JSON\s*(\{.*?\})\s*END_JSON", s, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m2 = re.search(r"<json>(.*)</json>", s, flags=re.DOTALL | re.IGNORECASE)
        if m2:
            return m2.group(1).strip()
        return None

    def remove_trailing_commas(s: str) -> str:
        # remove trailing commas before } or ]
        s_prev = None
        out = s
        # loop because nested trailing commas may require multiple passes
        while s_prev != out:
            s_prev = out
            out = re.sub(r",\s*(\}|])", r"\1", out)
        return out

    def try_parse_json_candidate(s: str):
        # Try multiple parsing strategies on a candidate string; return parsed or raise
        # 1) direct json.loads
        try:
            parsed = json.loads(s)
            return parsed
        except Exception:
            pass
        # 2) try removing trailing commas then json.loads
        try:
            repaired = remove_trailing_commas(s)
            parsed = json.loads(repaired)
            return parsed
        except Exception:
            pass
        # 3) try converting single-quotes to double-quotes cautiously (only if there are no double quotes)
        if "'" in s and '"' not in s:
            try:
                cand = s.replace("'", '"')
                parsed = json.loads(cand)
                return parsed
            except Exception:
                pass
        # 4) try ast.literal_eval for python-style dicts/lists (True/False/None allowed)
        try:
            evaluated = ast.literal_eval(s)
            # convert python structures to JSON-compatible structures via dumps/loads
            jtxt = json.dumps(evaluated, default=str)
            parsed = json.loads(jtxt)
            return parsed
        except Exception:
            pass
        # if nothing worked, raise
        raise JSONParseError("Failed to parse candidate as JSON")

    # Define typical fields for validation
    typical_fields = ['pan_number', 'name', 'fathers_name', 'date_of_birth', 'signature_present', 
                     'uid_number', 'gender', 'address', 'pin_code', 'state', 'dl_number', 'issue_date', 'valid_until']

    def has_expected_fields(obj: Any) -> bool:
        """Check if object has at least one expected field"""
        if isinstance(obj, dict):
            return any(field in obj for field in typical_fields)
        return False

    # CASE 1: If response is already a dict with the expected structure, return it directly
    if isinstance(resp, dict):
        # Check if this is the final extracted data we want
        if has_expected_fields(resp):
            return resp
        
        # Check for nested response structure (common in Ollama responses)
        if 'response' in resp and isinstance(resp['response'], str):
            # Try to parse the response string
            try:
                parsed_response = json.loads(resp['response'])
                if has_expected_fields(parsed_response):
                    return parsed_response
            except:
                # If direct parse fails, process the response string
                return _extract_json_from_model_response(resp['response'])
        
        # Check for 'text' field (common in sync responses)
        if 'text' in resp and isinstance(resp['text'], str):
            return _extract_json_from_model_response(resp['text'])
        
        # Check all string values in the dict
        for key, value in resp.items():
            if isinstance(value, str):
                try:
                    parsed_value = _extract_json_from_model_response(value)
                    if has_expected_fields(parsed_value):
                        return parsed_value
                except:
                    continue

    # CASE 2: If response is a string, try different extraction strategies
    if isinstance(resp, str):
        candidates = []
        
        # First, try direct JSON parsing (most common case - response is already valid JSON)
        try:
            direct_parse = json.loads(resp)
            if has_expected_fields(direct_parse):
                return direct_parse
        except:
            pass
        
        # Second, try extracting between markers
        marked_json = extract_between_markers(resp)
        if marked_json:
            candidates.append(marked_json)
        
        # Third, try stripping code fences and using the result
        stripped = strip_code_fences(resp)
        if stripped and stripped != resp:
            candidates.append(stripped)
        
        # Fourth, look for JSON object patterns in the string
        # Improved pattern to find JSON objects with nested content
        json_patterns = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', resp)
        for pattern in json_patterns:
            if any(field in pattern for field in typical_fields):
                candidates.append(pattern)
        
        # Also include the original response as a candidate
        candidates.append(resp)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for cand in candidates:
            if cand not in seen:
                seen.add(cand)
                unique_candidates.append(cand)
        
        # Try parsing each candidate
        last_err = None
        for cand in unique_candidates:
            try:
                parsed = try_parse_json_candidate(cand)
                if isinstance(parsed, dict) and has_expected_fields(parsed):
                    return parsed
                if isinstance(parsed, list):
                    return {"result": parsed}
            except JSONParseError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue

    # CASE 3: For any other type, convert to string and try
    try:
        str_resp = str(resp)
        if str_resp:
            return _extract_json_from_model_response(str_resp)
    except:
        pass

    # If we reached here, nothing parsed successfully
    preview = str(resp)[:1000] + "..." if len(str(resp)) > 1000 else str(resp)
    raise JSONParseError(f"Unable to extract JSON from model response. Preview: {preview}")

# -----------------------
# Scoring heuristics
# -----------------------
def _field_confidence_from_raw(value: Any, validator_ok: bool) -> float:
    """
    Simple heuristic: if validator confirms -> high; else medium/low depending on presence.
    """
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 0.9 if validator_ok else 0.6
    s = str(value).strip()
    if not s:
        return 0.0
    if validator_ok:
        return 0.9
    # some heuristics for partial confidence:
    length = len(s)
    if length >= 6:
        return 0.6
    if length >= 3:
        return 0.4
    return 0.2


def _aggregate_completeness_score(conf_scores: Dict[str, float]) -> float:
    if not conf_scores:
        return 0.0
    vals = list(conf_scores.values())
    # completeness = average confidence weighted by presence
    return float(sum(vals) / len(vals))


# -----------------------
# Extractor core class
# -----------------------
class VisionExtractor:
    def __init__(self, cfg: Optional[ExtractorConfig] = None):
        self.cfg = cfg or ExtractorConfig()
        self.logger = _setup_logger(self.cfg)
        self.ollama = OllamaClient(self.cfg, self.logger)
        # Avoid get_event_loop() at construction time — prefer running loop detection later
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop right now; will create/set one when running extraction
            self._loop = None

    # -------------------
    # Input validation
    # -------------------
    def _validate_classifier_input(self, classified_pages: Dict[str, Any]) -> None:
        """
        Validate the classifier -> extractor input shape; see integration spec.
        Expects dict with 'documents' and 'pages' structures (from document_classifier outputs).
        """
        if not isinstance(classified_pages, dict):
            raise InvalidInputError("Input must be a dict as provided by classifier output")
        if "documents" not in classified_pages and "pages" not in classified_pages:
            raise InvalidInputError("Input must contain 'documents' or 'pages' key from classifier output")
        # no further strictness here; downstream functions will enforce page-level items

    # -------------------
    # Routing
    # -------------------
    def _route_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Given a document-level dict from classifier (with 'pages' list), route and aggregate.
        Returns per-document extraction plan: { document_id, document_type, pages: [ {page info} ] }
        """
        doc_id = doc.get("document_id") or f"DOC_{hashlib.md5(json.dumps(doc, default=str).encode()).hexdigest()[:8]}"
        dtype = doc.get("document_type", "unknown")
        pages = doc.get("pages", [])
        return {"document_id": doc_id, "document_type": dtype, "pages": pages, "complete": doc.get("complete", False)}

    # -------------------
    # Prompt preparation
    # -------------------
    def _prepare_prompt(self, document_type: str, side: str) -> str:
        key = None
        if document_type == "aadhar":
            key = "aadhar_front" if side == "front" else "aadhar_back"
        elif document_type == "pan":
            key = "pan_card"
        elif document_type == "driving_license":
            key = "driving_license_front"
        else:
            raise DocumentSchemaError(f"Unsupported document type for prompt: {document_type}/{side}")
        return PROMPT_TEMPLATES[key]
    
    async def health_check(self) -> bool:
        """
        Check if Ollama is reachable and model is loaded
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Check if Ollama is running
                async with session.get("http://localhost:11434/api/tags", timeout=10) as resp:
                    if resp.status != 200:
                        self.logger.error("Ollama health check failed: HTTP %s", resp.status)
                        return False
                    
                    data = await resp.json()
                    models = [model["name"] for model in data.get("models", [])]
                    if self.cfg.model_name not in models:
                        self.logger.error("Model %s not found in Ollama. Available models: %s", 
                                        self.cfg.model_name, models)
                        return False
                    
                    self.logger.info("Ollama health check passed. Model %s is available.", self.cfg.model_name)
                    return True
                    
        except Exception as e:
            self.logger.error("Ollama health check failed: %s", e)
            return False
        
    # -------------------
    # Single-page extraction (async)
    # -------------------
    async def _extract_page_async(self, page: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract fields from a single page dict using Ollama async client.
        Returns extraction result for that side with confidence scores etc.
        """
        start = time.perf_counter()
        page_meta = page.get("metadata", {}) or {}
        img_arr = page.get("image_array")
        if img_arr is None:
            raise InvalidInputError("Page missing 'image_array'")

        # prepare prompt
        doc_type = page.get("document_type")
        side = page.get("side", "single")
        prompt = self._prepare_prompt(doc_type, side)

        # convert image to jpeg base64
        try:
            img_b64 = _rgb_numpy_to_jpeg_b64(img_arr, quality=self.cfg.jpeg_quality, max_side=self.cfg.max_image_side)
        except Exception as e:
            self.logger.exception("Failed to encode image for page %s: %s", page.get("page_number"), e)
            img_b64 = None

        # call Ollama
        model_resp = await self.ollama.generate(prompt, image_b64=img_b64)

        # If parsing fails, capture full debug artifacts: raw string(s), response type, and prompt
        try:
            parsed = _extract_json_from_model_response(model_resp)
        except JSONParseError as e:
            # Make a helpful debug dump
            try:
                raw_txt = model_resp if isinstance(model_resp, str) else json.dumps(model_resp, default=str, ensure_ascii=False)[:20000]
            except Exception:
                raw_txt = str(model_resp)[:20000]

            short_preview = (raw_txt[:1000] + "...") if len(raw_txt) > 1000 else raw_txt
            debug_fn = f"failed_parse_doc_{page.get('page_number')}_{int(time.time())}.debug.txt"
            try:
                with open(debug_fn, "w", encoding="utf-8") as fh:
                    fh.write("=== PROMPT ===\n")
                    fh.write(prompt + "\n\n")
                    fh.write("=== RAW MODEL RESPONSE (truncated to 20k) ===\n")
                    fh.write(raw_txt)
                self.logger.error("Saved failed model response to %s", debug_fn)
            except Exception:
                self.logger.exception("Failed to write debug file for parse failure")

            # raise a detailed JSONParseError so upstream preserves context
            raise JSONParseError(
                f"Model returned unparsable response for page {page.get('page_number')}: {e}. "
                f"Preview: {short_preview}"
            )

        # At this point parsed should be a dict with extracted fields
        extracted_fields = {}
        confidence_scores: Dict[str, float] = {}
        validation_errors: List[str] = []

        # choose schema
        schema = {}
        key_map = None
        if doc_type == "aadhar":
            schema = AADHAR_FRONT_FIELDS if side == "front" else AADHAR_BACK_FIELDS
        elif doc_type == "pan":
            schema = PAN_FIELDS
        elif doc_type == "driving_license":
            schema = DL_FRONT_FIELDS
        else:
            raise DocumentSchemaError(f"Unsupported document type during parsing: {doc_type}")

        # Populate fields - we expect keys in parsed to match schema keys, but model may vary => try fuzzy matching
        for field_name in schema.keys():
            raw_val = None
            # attempt direct lookup
            if field_name in parsed:
                raw_val = parsed[field_name]
            else:
                # try case-insensitive or underscore/space variants
                for k, v in parsed.items():
                    if isinstance(k, str) and k.lower().replace(" ", "_") == field_name.lower():
                        raw_val = v
                        break
                # last resort: approximate - extract by field name token presence in stringified JSON
            # Validate
            is_valid, normalized = (False, None)
            if field_name == "pan_number":
                self.logger.warning(f"[DEBUG-PAN] Raw model PAN value (pre-validation): {repr(raw_val)}")

            if self.cfg.enable_field_validation:
                is_valid, normalized = _validate_field(field_name, raw_val)
            else:
                normalized = raw_val
                is_valid = True if raw_val is not None else False

            extracted_fields[field_name] = normalized
            score = _field_confidence_from_raw(raw_val, is_valid)
            confidence_scores[field_name] = float(round(score, 4))
            if not is_valid:
                validation_errors.append(f"{field_name}_invalid_or_missing")

        # cross-field consistency checks (example: DOB on front vs other sources)
        # minimal: if pan and date_of_birth doesn't parse -> add validation error
        # (more complex checks may be added)
        completeness_score = _aggregate_completeness_score(confidence_scores)
        elapsed = time.perf_counter() - start

        return {
            "page_number": page.get("page_number"),
            "side": side,
            "extracted": extracted_fields,
            "confidence_scores": confidence_scores,
            "completeness_score": float(round(completeness_score, 4)),
            "validation_errors": validation_errors,
            "processing_metadata": {
                "extraction_time": float(round(elapsed, 4)),
                "model_used": self.cfg.model_name,
                "retry_attempts": self.cfg.max_retries,
                "enhancements_applied": page_meta.get("enhancements_applied", []),
            }
        }

    # -------------------
    # Single-page extraction (sync wrapper)
    # -------------------
    def _extract_page_sync(self, page: Dict[str, Any]) -> Dict[str, Any]:
        # prepare prompt and image
        start = time.perf_counter()
        page_meta = page.get("metadata", {}) or {}
        img_arr = page.get("image_array")
        if img_arr is None:
            raise InvalidInputError("Page missing 'image_array'")

        doc_type = page.get("document_type")
        side = page.get("side", "single")
        prompt = self._prepare_prompt(doc_type, side)

        try:
            img_b64 = _rgb_numpy_to_jpeg_b64(img_arr, quality=self.cfg.jpeg_quality, max_side=self.cfg.max_image_side)
        except Exception as e:
            self.logger.exception("Failed to encode image for page %s: %s", page.get("page_number"), e)
            img_b64 = None

        model_resp = self.ollama.generate_sync(prompt, image_b64=img_b64)
        try:
            parsed = _extract_json_from_model_response(model_resp)
        except JSONParseError as e:
            raise JSONParseError(f"Model returned unparsable response for page {page.get('page_number')}: {e}")

        extracted_fields = {}
        confidence_scores: Dict[str, float] = {}
        validation_errors: List[str] = []

        if doc_type == "aadhar":
            schema = AADHAR_FRONT_FIELDS if side == "front" else AADHAR_BACK_FIELDS
        elif doc_type == "pan":
            schema = PAN_FIELDS
        elif doc_type == "driving_license":
            schema = DL_FRONT_FIELDS
        else:
            raise DocumentSchemaError(f"Unsupported document type during parsing: {doc_type}")

        for field_name in schema.keys():
            raw_val = None
            if field_name in parsed:
                raw_val = parsed[field_name]
            else:
                for k, v in parsed.items():
                    if isinstance(k, str) and k.lower().replace(" ", "_") == field_name.lower():
                        raw_val = v
                        break
            if field_name == "pan_number":
                self.logger.warning(f"[DEBUG-PAN] Raw model PAN value (pre-validation): {repr(raw_val)}")

            if self.cfg.enable_field_validation:
                is_valid, normalized = _validate_field(field_name, raw_val)
            else:
                normalized = raw_val
                is_valid = True if raw_val is not None else False
            extracted_fields[field_name] = normalized
            confidence_scores[field_name] = float(round(_field_confidence_from_raw(raw_val, is_valid), 4))
            if not is_valid:
                validation_errors.append(f"{field_name}_invalid_or_missing")

        completeness_score = _aggregate_completeness_score(confidence_scores)
        elapsed = time.perf_counter() - start
        return {
            "page_number": page.get("page_number"),
            "side": side,
            "extracted": extracted_fields,
            "confidence_scores": confidence_scores,
            "completeness_score": float(round(completeness_score, 4)),
            "validation_errors": validation_errors,
            "processing_metadata": {
                "extraction_time": float(round(elapsed, 4)),
                "model_used": self.cfg.model_name,
                "retry_attempts": self.cfg.max_retries,
                "enhancements_applied": page_meta.get("enhancements_applied", []),
            }
        }

    # -------------------
    # Document-level extraction
    # -------------------
    async def _extract_document_async(self, doc_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all pages of a single document in parallel up to concurrency limits.
        Returns aggregated EXTRACTION_OUTPUT for that document.
        """
        doc_id = doc_plan["document_id"]
        doc_type = doc_plan["document_type"]
        pages = doc_plan["pages"]  # list of page dicts

        tasks = []
        for p in pages:
            tasks.append(self._extract_page_async(p))

        results = []
        if tasks:
            # limit concurrency by cfg.batch_size or client semaphore
            # use asyncio.gather with return_exceptions to gather all results
            gathered = await asyncio.gather(*tasks, return_exceptions=True)
            for r in gathered:
                if isinstance(r, Exception):
                    # convert to partial extraction result with error flagged
                    self.logger.exception("Page extraction error inside document %s: %s", doc_id, r)
                    results.append({
                        "page_number": None,
                        "side": "unknown",
                        "extracted": {},
                        "confidence_scores": {},
                        "completeness_score": 0.0,
                        "validation_errors": [str(r)],
                        "processing_metadata": {"extraction_time": 0.0, "model_used": self.cfg.model_name, "retry_attempts": self.cfg.max_retries, "enhancements_applied": []}
                    })
                else:
                    results.append(r)
        else:
            results = []

        # Aggregate per-document fields: include front/back keys under 'extracted_data'
        fields_agg: Dict[str, Any] = {}
        confidence_agg: Dict[str, float] = {}
        validation_errors_agg: List[str] = []
        completeness_vals: List[float] = []
        processing_meta_agg = {"extraction_time": 0.0, "model_used": self.cfg.model_name, "retry_attempts": 0, "enhancements_applied": []}
        for r in results:
            # merge extracted dicts (prefix side for duplicate keys? keep simple: merge and prefer front)
            for k, v in r.get("extracted", {}).items():
                # if key already exists and value is empty, overwrite with non-empty
                prev = fields_agg.get(k)
                if (prev is None or prev == "") and (v is not None and v != ""):
                    fields_agg[k] = v
                else:
                    # prefer existing (e.g. front has priority)
                    fields_agg.setdefault(k, v)
            # merge confidences: keep max for same field
            for fk, fv in r.get("confidence_scores", {}).items():
                confidence_agg[fk] = max(confidence_agg.get(fk, 0.0), float(fv))
            validation_errors_agg.extend(r.get("validation_errors", []))
            completeness_vals.append(r.get("completeness_score", 0.0))
            # aggregate timings
            pm = r.get("processing_metadata", {})
            processing_meta_agg["extraction_time"] += pm.get("extraction_time", 0.0)
            processing_meta_agg["retry_attempts"] = max(processing_meta_agg.get("retry_attempts", 0), pm.get("retry_attempts", 0))
            if pm.get("enhancements_applied"):
                processing_meta_agg["enhancements_applied"].extend(pm.get("enhancements_applied", []))

        overall_completeness = float(round(float(sum(completeness_vals) / (len(completeness_vals) or 1)), 4))

        # Basic sanity: if overall confidence below threshold -> flag low confidence
        overall_conf_avg = float(round(sum(confidence_agg.values()) / (len(confidence_agg) or 1), 4))
        if overall_conf_avg < self.cfg.confidence_threshold:
            # we won't raise an exception, but include flag
            validation_errors_agg.append("overall_confidence_below_threshold")

        # Compose final output structure per spec
        extra_out = {
            "document_id": doc_id,
            "document_type": doc_type,
            "side": "multi" if len(pages) > 1 else (pages[0].get("side") if pages else "single"),
            "extracted_data": {
                "fields": fields_agg,
                "confidence_scores": confidence_agg,
                "completeness_score": overall_completeness,
                "validation_errors": list(set(validation_errors_agg)),
            },
            "processing_metadata": {
                "extraction_time": float(round(processing_meta_agg["extraction_time"], 4)),
                "model_used": processing_meta_agg.get("model_used"),
                "retry_attempts": processing_meta_agg.get("retry_attempts"),
                "enhancements_applied": list(set(processing_meta_agg.get("enhancements_applied", [])))
            }
        }
        return extra_out

    def _extract_document_sync(self, doc_plan: Dict[str, Any]) -> Dict[str, Any]:
        # sync wrapper: iterate pages sequentially
        doc_id = doc_plan["document_id"]
        doc_type = doc_plan["document_type"]
        pages = doc_plan["pages"] or []

        fields_agg = {}
        confidence_agg = {}
        validation_errors = []
        completeness_vals = []
        proc_meta = {"extraction_time": 0.0, "model_used": self.cfg.model_name, "retry_attempts": 0, "enhancements_applied": []}

        for p in pages:
            try:
                r = self._extract_page_sync(p)
            except Exception as e:
                self.logger.exception("Page extraction failed sync for doc %s: %s", doc_id, e)
                r = {
                    "page_number": p.get("page_number"),
                    "side": p.get("side"),
                    "extracted": {},
                    "confidence_scores": {},
                    "completeness_score": 0.0,
                    "validation_errors": [str(e)],
                    "processing_metadata": {"extraction_time": 0.0, "model_used": self.cfg.model_name, "retry_attempts": self.cfg.max_retries}
                }
            for k, v in r.get("extracted", {}).items():
                prev = fields_agg.get(k)
                if (prev is None or prev == "") and (v is not None and v != ""):
                    fields_agg[k] = v
                else:
                    fields_agg.setdefault(k, v)
            for fk, fv in r.get("confidence_scores", {}).items():
                confidence_agg[fk] = max(confidence_agg.get(fk, 0.0), float(fv))
            validation_errors.extend(r.get("validation_errors", []))
            completeness_vals.append(r.get("completeness_score", 0.0))
            pm = r.get("processing_metadata", {})
            proc_meta["extraction_time"] += pm.get("extraction_time", 0.0)
            proc_meta["retry_attempts"] = max(proc_meta["retry_attempts"], pm.get("retry_attempts", 0))

        overall_completeness = float(round(sum(completeness_vals) / (len(completeness_vals) or 1), 4))
        overall_conf_avg = float(round(sum(confidence_agg.values()) / (len(confidence_agg) or 1), 4))
        if overall_conf_avg < self.cfg.confidence_threshold:
            validation_errors.append("overall_confidence_below_threshold")

        extra_out = {
            "document_id": doc_id,
            "document_type": doc_type,
            "side": "multi" if len(pages) > 1 else (pages[0].get("side") if pages else "single"),
            "extracted_data": {
                "fields": fields_agg,
                "confidence_scores": confidence_agg,
                "completeness_score": overall_completeness,
                "validation_errors": list(set(validation_errors)),
            },
            "processing_metadata": {
                "extraction_time": float(round(proc_meta["extraction_time"], 4)),
                "model_used": proc_meta.get("model_used"),
                "retry_attempts": proc_meta.get("retry_attempts"),
                "enhancements_applied": list(set(proc_meta.get("enhancements_applied", [])))
            }
        }
        return extra_out

    # -------------------
    # Public API: extract_documents
    # -------------------
    def extract_documents(self, classifier_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        High-level entrypoint.
        Accepts classifier_output (the output of document_classifier.classify_and_aggregate or similar).
        Returns structured extractor output matching the spec for validator_agent downstream.
        """
        start_total = time.perf_counter()
        self._validate_classifier_input(classifier_output)

        documents_input = classifier_output.get("documents", []) or []
        pages_input = classifier_output.get("pages", []) or []

        # If classifier provided documents, use them. Else try grouping pages into one document each.
        doc_plans = []
        if documents_input:
            for d in documents_input:
                doc_plans.append(self._route_document(d))
        else:
            # fallback: treat each page as single document
            for p in pages_input:
                doc_plans.append({
                    "document_id": p.get("metadata", {}).get("page_id", f"PAGE_{p.get('page_number', 'NA')}"),
                    "document_type": p.get("document_type", "unknown"),
                    "pages": [p],
                    "complete": p.get("complete", False)
                })

        # Process documents either asynchronously or sync based on config
        results = []
        errors = []
        total_ex_time = 0.0
        failed = 0
        success = 0

        if self.cfg.enable_async_processing:
            # run with asyncio event loop
            async def _run_all():
                tasks = []
                sem = asyncio.Semaphore(self.cfg.max_concurrent_requests)
                for dp in doc_plans:
                    # For each doc, create coroutine
                    async def _wrapper(plan):
                        async with sem:
                            try:
                                return await self._extract_document_async(plan)
                            except Exception as e:
                                self.logger.exception("Document extraction failed for %s: %s", plan.get("document_id"), e)
                                return {"document_id": plan.get("document_id"), "error": str(e)}
                    tasks.append(_wrapper(dp))
                return await asyncio.gather(*tasks, return_exceptions=False)

            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                gathered = loop.run_until_complete(_run_all())
                # close internal aio session
                try:
                    loop.run_until_complete(self.ollama.close())
                except Exception:
                    pass
                loop.close()
            except Exception as e:
                self.logger.exception("Async extraction failed; falling back to sync: %s", e)
                # fallback to sync extraction for each document
                for dp in doc_plans:
                    try:
                        out = self._extract_document_sync(dp)
                        gathered = gathered if 'gathered' in locals() else []
                        gathered.append(out)
                    except Exception as e2:
                        self.logger.exception("Fallback sync extraction failed for %s: %s", dp.get("document_id"), e2)
                        gathered = gathered if 'gathered' in locals() else []
                        gathered.append({"document_id": dp.get("document_id"), "error": str(e2)})
            # process gathered outputs
            for g in gathered:
                if isinstance(g, dict) and g.get("error"):
                    failed += 1
                    errors.append({"document_id": g.get("document_id"), "error": g.get("error")})
                else:
                    success += 1
                    results.append(g)
        else:
            # synchronous sequential processing
            for dp in doc_plans:
                try:
                    out = self._extract_document_sync(dp)
                    results.append(out)
                    success += 1
                except Exception as e:
                    self.logger.exception("Document extraction sync failed for %s: %s", dp.get("document_id"), e)
                    errors.append({"document_id": dp.get("document_id"), "error": str(e)})
                    failed += 1

        # Compose final outer structure per spec
        total_elapsed = time.perf_counter() - start_total
        processing_metrics = {
            "total_extraction_time": float(round(total_elapsed, 4)),
            "successful_extractions": success,
            "failed_extractions": failed,
            "average_confidence": None
        }
        # compute average confidence across all documents (simple mean of doc completeness)
        if results:
            avg_conf = float(round(sum([d["extracted_data"]["completeness_score"] for d in results]) / len(results), 4))
            processing_metrics["average_confidence"] = avg_conf
        else:
            processing_metrics["average_confidence"] = 0.0

        final_out = {
            "success": (failed == 0),
            "documents": results,
            "errors": errors,
            "processing_metrics": processing_metrics
        }
        return final_out


# -----------------------
# Example integration notes (for maintainers)
# -----------------------
# - This extractor expects input shaped like document_classifier.classify_and_aggregate()
#   which produces 'documents' and 'pages'. See document_classifier.py for exact shapes and
#   grouping rules. :contentReference[oaicite:2]{index=2}
#
# - The preprocessor provides page dicts containing 'image_array' numpy H,W,3 (RGB) and 'metadata'.
#   See document_preprocessor.py for details. :contentReference[oaicite:3]{index=3}
#
# - Ollama integration: endpoint configured via ExtractorConfig. We embed images as base64 JPEG
#   data URIs into the prompt. If your Ollama local deployment requires multipart/mime please adapt
#   OllamaClient.generate/_sync to use that transport.
#
# -----------------------
# If executed as script, a small smoke-test harness is provided.
# -----------------------
if __name__ == "__main__":
    import argparse
    import numpy as np
    import os
    from pprint import pprint

    parser = argparse.ArgumentParser(description="Vision Extractor Smoke Test")
    parser.add_argument("--sample-image", help="Path to sample image (PNG/JPG) to run through extractor (single page)", required=False)
    parser.add_argument("--async", dest="use_async", action="store_true", help="Enable async processing (default)")
    parser.set_defaults(use_async=True)
    args = parser.parse_args()

    cfg = ExtractorConfig()
    extractor = VisionExtractor(cfg)

    # Build fake classifier output if sample-image provided
    if args.sample_image and os.path.exists(args.sample_image):
        img = cv2.imread(args.sample_image)  # BGR
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sample_page = {
            "page_number": 1,
            "document_type": "pan",       # adjust to test different schemas: aadhar/driving_license/pan
            "side": "single",
            "image_array": rgb,
            "metadata": {"source": args.sample_image}
        }
        classifier_out = {
            "pages": [sample_page],
            "documents": [
                {
                    "document_id": "PAN_TEST_1",
                    "document_type": "pan",
                    "pages": [sample_page],
                    "complete": True
                }
            ]
        }
    else:
        # simple synthetic blank test will fail to extract but demonstrates flow
        blank = np.full((800, 1200, 3), 230, dtype=np.uint8)
        sample_page = {
            "page_number": 1,
            "document_type": "aadhar",
            "side": "front",
            "image_array": blank,
            "metadata": {}
        }
        classifier_out = {"pages": [sample_page], "documents": [{"document_id": "AADHAR_TEST", "document_type": "aadhar", "pages": [sample_page], "complete": False}]}

    out = extractor.extract_documents(classifier_out)
    pprint(out)
