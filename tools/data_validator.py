# document_validator.py
from __future__ import annotations

import re
import uuid
import json
import math
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


# -------------------------
# Config
# -------------------------
@dataclass
class ValidatorConfig:
    # Confidence thresholds
    conf_auto_approve: float = 0.90
    conf_manual_review: float = 0.85
    # Age ranges
    aadhar_min_age: int = 5
    aadhar_max_age: int = 120
    pan_min_age_individual: int = 18
    dl_min_age: int = 18
    # Date range cutoffs
    dl_issue_min_year: int = 1980
    dl_max_validity_years: int = 20
    # Parallelism
    max_workers: int = 4
    # Logging
    log_level: int = logging.INFO


# -------------------------
# Logging
# -------------------------
log = logging.getLogger("DocumentValidator")
log.addHandler(logging.StreamHandler())
log.setLevel(ValidatorConfig().log_level)

# -------------------------
# Utilities: date parsing, age calculation, normalization
# -------------------------
_DATE_PAT = re.compile(r"\b(0[1-9]|[12]\d|3[01])[/\-](0[1-9]|1[0-2])[/\-](\d{4})\b")


def parse_date_dmy(value: Optional[str]) -> Optional[datetime]:
    """
    Parse common DMY date strings and return a timezone-aware datetime (UTC).
    Accepts DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD and strict regex matches.
    """
    if not value or not isinstance(value, str):
        return None
    s = value.strip()
    # strict DMY pattern first
    m = _DATE_PAT.match(s)
    if m:
        d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(year=y, month=mth, day=d, tzinfo=timezone.utc)
        except Exception:
            return None
    # try common alternatives
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return None


def today_date() -> datetime:
    return datetime.now(timezone.utc)


def calc_age_years(birth_dt: datetime, ref: Optional[datetime] = None) -> float:
    """
    Calculate age in whole years. Works when birth_dt/ref are timezone-aware or naive.
    Converts to date objects to avoid tz-aware vs naive subtraction errors.
    """
    ref = ref or today_date()
    # Convert to date() for both operands
    try:
        b_date = birth_dt.date() if isinstance(birth_dt, datetime) else birth_dt
    except Exception:
        # not a datetime-like object
        raise
    r_date = ref.date() if isinstance(ref, datetime) else ref
    days = (r_date - b_date).days
    years = days / 365.2425
    return float(math.floor(years))


def normalize_str_optional(s: Any) -> Optional[str]:
    if s is None:
        return None
    st = str(s).strip()
    return st if st != "" else None


# -------------------------
# Verhoeff algorithm for Aadhaar checksum
# -------------------------
# Implementation based on classic Verhoeff tables
# (self-contained)
_d_table = [
    [0,1,2,3,4,5,6,7,8,9],
    [1,2,3,4,0,6,7,8,9,5],
    [2,3,4,0,1,7,8,9,5,6],
    [3,4,0,1,2,8,9,5,6,7],
    [4,0,1,2,3,9,5,6,7,8],
    [5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2],
    [7,6,5,9,8,2,1,0,4,3],
    [8,7,6,5,9,3,2,1,0,4],
    [9,8,7,6,5,4,3,2,1,0],
]
_p_table = [
    [0,1,2,3,4,5,6,7,8,9],
    [1,5,7,6,2,8,3,0,9,4],
    [5,8,0,3,7,9,6,1,4,2],
    [8,9,1,6,0,4,3,5,2,7],
    [9,4,5,3,1,2,6,8,7,0],
    [4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5],
    [7,0,4,6,9,1,3,2,5,8],
]
_inv_table = [0,4,3,2,1,5,6,7,8,9]


def verhoeff_validate(numstr: str) -> bool:
    """Return True if numstr passes Verhoeff checksum. Expects digits only."""
    if not numstr or not numstr.isdigit():
        return False
    c = 0
    # process digits right-to-left
    for i, ch in enumerate(reversed(numstr)):
        d = int(ch)
        c = _d_table[c][_p_table[(i % 8)][d]]
    return c == 0


# -------------------------
_pan_re = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")


def _char_value_for_pan(ch: str) -> int:
    if ch.isalpha():
        return ord(ch.upper()) - ord('A') + 1
    return int(ch)


def compute_pan_checksum(pan9: str) -> str:
    """
    compute checksum letter for first 9 chars of PAN (positions 0..8).
    Returns uppercase letter A..Z.
    """
    weights = [1,2]*5  # length 10; we use first 9 weights for first 9 chars then derive check from sum
    # But algorithm uses all first 9 chars; for checksum we use weights on first 9 then map remainder to letter
    s = 0
    for i, ch in enumerate(pan9):
        val = _char_value_for_pan(ch)
        w = weights[i]
        prod = val * w
        # if two-digit result, sum digits
        prod_sum = sum(int(d) for d in str(prod))
        s += prod_sum
    remainder = s % 26
    # map 0->A, 1->B, ..., 25->Z
    letter = chr(ord('A') + remainder)
    return letter


def pan_validate_with_checksum(pan: str, confidence: Optional[float] = None,
                               min_confidence_override: float = 0.90) -> Tuple[bool, List[str]]:
    """
    Robust PAN validation.

    - Normalizes input by removing non-alphanumeric chars and uppercasing.
    - Strictly validates format against regex: 5 letters + 4 digits + 1 letter.
    - Computes checksum using current compute_pan_checksum(...) and accepts if matches.
    - If checksum mismatches:
        1) Attempt safe OCR confusion corrections (O<->0, I<->1, S<->5, B<->8, Z<->2, G<->6).
        2) Try single-character deletion for 11-char inputs (duplicate char from OCR).
        3) If still invalid and confidence is provided and >= min_confidence_override,
           accept with a warning "checksum_override_by_confidence".
    - Returns (is_valid, errors_list). When accepted by override, errors_list will contain
      a descriptive warning for audit.
    """
    errors: List[str] = []
    if not pan or not isinstance(pan, str):
        errors.append("pan_missing_or_not_string")
        return False, errors

    # Normalize
    p_raw = re.sub(r"[^A-Za-z0-9]", "", pan).upper().strip()
    if not p_raw:
        errors.append("pan_empty_after_normalize")
        return False, errors

    # Strict pattern check quickly
    if _pan_re.match(p_raw):
        # verify checksum
        pan9 = p_raw[:9]
        expected = compute_pan_checksum(pan9)
        if p_raw[-1] == expected:
            return True, []
        else:
            errors.append(f"pan_checksum_mismatch_expected_{expected}")
    else:
        errors.append("pan_format_invalid")

    # SAFE CORRECTION ATTEMPTS
    confusion_map = {
        "O": ["0"], "0": ["O"],
        "I": ["1", "L"], "1": ["I", "L"],
        "S": ["5"], "5": ["S"],
        "B": ["8"], "8": ["B"],
        "Z": ["2"], "2": ["Z"],
        "G": ["6"], "6": ["G"],
    }

    def generate_confusion_candidates(s: str, max_replacements: int = 2):
        indices = [i for i, ch in enumerate(s) if ch in confusion_map]
        if not indices:
            return
        from itertools import combinations, product
        for r in range(1, min(max_replacements, len(indices)) + 1):
            for idx_subset in combinations(indices, r):
                substitution_lists = [confusion_map[s[i]] for i in idx_subset]
                for subs in product(*substitution_lists):
                    lst = list(s)
                    for pos, sub_ch in zip(idx_subset, subs):
                        lst[pos] = sub_ch
                    yield "".join(lst)

    checked = set()
    # 1) Try confusion-based substitutions (for 10-char strings)
    if len(p_raw) == 10:
        for cand in generate_confusion_candidates(p_raw, max_replacements=2) or []:
            if cand in checked:
                continue
            checked.add(cand)
            if _pan_re.match(cand):
                if cand[-1] == compute_pan_checksum(cand[:9]):
                    # corrected candidate valid
                    return True, []

    # 2) If length 11 -> try single-character deletion (ambiguous if multiple candidates)
    if len(p_raw) == 11:
        valid_cands = []
        pan_re = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")
        for i in range(len(p_raw)):
            cand = p_raw[:i] + p_raw[i+1:]
            if cand in checked:
                continue
            checked.add(cand)
            if pan_re.fullmatch(cand) and cand[-1] == compute_pan_checksum(cand[:9]):
                valid_cands.append(cand)
        if len(valid_cands) == 1:
            return True, []
        if len(valid_cands) > 1:
            errors.append("pan_ambiguous_after_deletion_candidates")
            return False, errors

    # 3) Last-resort: one-character substitution attempts (single replacment)
    if len(p_raw) == 10:
        for i, ch in enumerate(p_raw):
            for sub in confusion_map.get(ch, []):
                cand = p_raw[:i] + sub + p_raw[i+1:]
                if cand in checked:
                    continue
                checked.add(cand)
                if _pan_re.match(cand) and cand[-1] == compute_pan_checksum(cand[:9]):
                    return True, []

    # 4) CONFIDENCE OVERRIDE (audit trail)
    # If user supplied a confidence and it's >= threshold, accept but *warn*.
    try:
        if confidence is not None and isinstance(confidence, (int, float)):
            if float(confidence) >= float(min_confidence_override):
                # Accept despite checksum mismatch — but record a warning for reviewers/audit.
                errors.append("checksum_override_by_confidence")
                return True, errors
    except Exception:
        pass

    # If none of the above succeeded, invalid.
    return False, errors


# -------------------------
# PIN code (Postal Index Number) validation
# -------------------------
_pin_re = re.compile(r"^[1-9]\d{5}$")  # first digit cannot be 0


_PIN_PREFIX_RANGES = [
    (110000, 119999),  # Delhi and nearby (example 11xxxx)
    (120000, 139999),  # Haryana / Punjab ranges included in 12-13 etc
    (140000, 159999),  # Punjab / Chandigarh / Haryana subset
    (160000, 199999),  # Himachal / Jammu etc
    (200000, 289999),  # UP / Uttarakhand
    (300000, 349999),  # Rajasthan
    (360000, 399999),  # Gujarat + small exceptions
    (400000, 449999),  # Maharashtra / Goa
    (450000, 489999),  # Madhya Pradesh
    (490000, 499999),  # Chhattisgarh
    (500000, 539999),  # Telangana / Andhra
    (560000, 599999),  # Karnataka
    (600000, 669999),  # Tamil Nadu / Puducherry
    (670000, 699999),  # Kerala / Lakshadweep
    (700000, 749999),  # West Bengal / Sikkim / Andaman & Nicobar etc
    (750000, 799999),  # Odisha / NE states subset
    (800000, 859999),  # Bihar / Jharkhand
    (900000, 999999),  # Army Postal Service / special zones
]


def pin_in_valid_range(pin: str) -> bool:
    try:
        num = int(pin)
    except Exception:
        return False
    for lo, hi in _PIN_PREFIX_RANGES:
        if lo <= num <= hi:
            return True
    return False


def validate_pin_code(pin: Optional[str]) -> Tuple[bool, List[str]]:
    errs = []
    if pin is None:
        errs.append("pin_missing")
        return False, errs
    s = str(pin).strip()
    if not _pin_re.match(s):
        errs.append("pin_format_invalid")
        return False, errs
    if s == "000000":
        errs.append("pin_all_zeros")
        return False, errs
    if not pin_in_valid_range(s):
        errs.append("pin_outside_known_ranges")
        # still consider as warning -> but per your spec, treat as invalid
        return False, errs
    return True, []


# -------------------------
_INDIA_STATES_NORMALIZED = {
    s.lower() for s in [
        "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa","Gujarat",
        "Haryana","Himachal Pradesh","Jharkhand","Karnataka","Kerala","Madhya Pradesh","Maharashtra",
        "Manipur","Meghalaya","Mizoram","Nagaland","Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu",
        "Telangana","Tripura","Uttar Pradesh","Uttarakhand","West Bengal",
        # union territories
        "Andaman and Nicobar Islands","Chandigarh","Dadra and Nagar Haveli and Daman and Diu","Delhi",
        "Jammu and Kashmir","Ladakh","Lakshadweep","Puducherry"
    ]
}


def normalize_state_name(s: Any) -> Optional[str]:
    if s is None:
        return None
    st = str(s).strip()
    if not st:
        return None
    st_norm = st.lower()
    # allow short forms like MH -> Maharashtra, common abbreviations
    abbr = {
        "mh":"maharashtra","dl":"delhi","ka":"karnataka","tn":"tamil nadu","ap":"andhra pradesh",
        "wb":"west bengal","gj":"gujarat","hr":"haryana","pb":"punjab","mp":"madhya pradesh",
        "od":"odisha","or":"odisha","tg":"telangana"
    }
    if st_norm in _INDIA_STATES_NORMALIZED:
        # return canonical capitalization by capitalizing words
        return " ".join([w.capitalize() for w in st_norm.split()])
    if st_norm in abbr:
        return " ".join([w.capitalize() for w in abbr[st_norm].split()])
    # try fuzzy: match if any known state startswith token
    for known in _INDIA_STATES_NORMALIZED:
        if known.startswith(st_norm):
            return " ".join([w.capitalize() for w in known.split()])
    return None


# -------------------------
# DL number basic check
# -------------------------
# Basic pattern: 2 letters (state code) followed by up to digits/letters; length 10-16
_dl_re_basic = re.compile(r"^[A-Z]{2}[A-Z0-9\-\/]{8,14}$", re.IGNORECASE)


def validate_dl_number(dl: Optional[str]) -> Tuple[bool, List[str]]:
    errs = []
    if dl is None:
        errs.append("dl_missing")
        return False, errs
    s = str(dl).strip()
    if not (10 <= len(s) <= 16):
        errs.append("dl_length_out_of_range")
        # but continue matching pattern
    if not _dl_re_basic.match(s):
        errs.append("dl_pattern_invalid")
        return False, errs
    return True, []


# -------------------------
# Field-level validators dispatcher
# -------------------------
def validate_aadhar_front(fields: Dict[str, Any], confidences: Dict[str, float], cfg: ValidatorConfig) -> Tuple[Dict[str, Any], List[str]]:
    validated = {}
    errors = []
    # uid_number
    uid = normalize_str_optional(fields.get("uid_number"))
    uid_ok = False
    uid_errors = []
    if uid:
        digits = re.sub(r"\D", "", uid)
        if len(digits) == 12 and verhoeff_validate(digits):
            if digits != "0"*12 and digits not in ("123456789012","012345678901"):  # trivial sequential checks
                uid_ok = True
            else:
                uid_errors.append("uid_sequential_or_allzeros")
        else:
            uid_errors.append("uid_verhoeff_failed_or_length")
    else:
        uid_errors.append("uid_missing_null")
    validated["uid_number"] = {"value": uid, "normalized_value": digits if uid else None, "status": "VALID" if uid_ok else "INVALID", "confidence": float(confidences.get("uid_number", 0.0)), "validation_rules": ["12digits","verhoeff","non_sequential"], "errors": uid_errors}
    if not uid_ok:
        errors.append("uid_invalid")

    # date_of_birth
    dob_raw = normalize_str_optional(fields.get("date_of_birth"))
    dob_errors = []
    dob_ok = False
    dob_norm = None
    if dob_raw:
        parsed = parse_date_dmy(dob_raw)
        if parsed:
            age = calc_age_years(parsed, today_date())
            if age < 0:
                dob_errors.append("dob_future")
            elif age < cfg.aadhar_min_age or age > cfg.aadhar_max_age:
                dob_errors.append("dob_age_out_of_range")
            else:
                dob_ok = True
                dob_norm = parsed.strftime("%d/%m/%Y")
        else:
            dob_errors.append("dob_parse_failed")
    else:
        dob_errors.append("dob_missing")
    validated["date_of_birth"] = {"value": dob_raw, "normalized_value": dob_norm, "status": "VALID" if dob_ok else "INVALID", "confidence": float(confidences.get("date_of_birth", 0.0)), "validation_rules": ["date_parse_ddmmyyyy","age_range"], "errors": dob_errors}
    if not dob_ok:
        errors.append("dob_invalid")

    # gender
    g_raw = normalize_str_optional(fields.get("gender"))
    g_errs = []
    g_ok = False
    g_norm = None
    if g_raw:
        gu = g_raw.strip().upper()
        if gu in ("MALE","FEMALE","OTHER"):
            g_ok = True
            g_norm = gu
        else:
            g_errs.append("gender_invalid_value")
    else:
        g_errs.append("gender_missing")
    validated["gender"] = {"value": g_raw, "normalized_value": g_norm, "status": "VALID" if g_ok else "INVALID", "confidence": float(confidences.get("gender", 0.0)), "validation_rules": ["allowed_values:MALE/FEMALE/OTHER"], "errors": g_errs}
    if not g_ok:
        errors.append("gender_invalid")

    return validated, errors


def validate_aadhar_back(fields: Dict[str, Any], confidences: Dict[str, float]) -> Tuple[Dict[str, Any], List[str]]:
    validated = {}
    errors = []
    pin = normalize_str_optional(fields.get("pin_code"))
    pin_ok, pin_errs = validate_pin_code(pin)
    validated["pin_code"] = {"value": pin, "normalized_value": pin, "status": "VALID" if pin_ok else "INVALID", "confidence": float(confidences.get("pin_code", 0.0)), "validation_rules": ["6digits","first_digit_not_zero","pin_range"], "errors": pin_errs}
    if not pin_ok:
        errors.append("pin_invalid")

    state_raw = normalize_str_optional(fields.get("state"))
    state_norm = normalize_state_name(state_raw)
    state_errs = []
    if not state_norm:
        state_errs.append("state_not_normalized_or_invalid")
    validated["state"] = {"value": state_raw, "normalized_value": state_norm, "status": "VALID" if state_norm else "INVALID", "confidence": float(confidences.get("state", 0.0)), "validation_rules": ["state_list_normalize"], "errors": state_errs}
    if not state_norm:
        errors.append("state_invalid")
    return validated, errors


def validate_pan(fields: Dict[str, Any], confidences: Dict[str, float], cfg: ValidatorConfig) -> Tuple[Dict[str, Any], List[str]]:
    validated = {}
    errors = []
    pan_raw = normalize_str_optional(fields.get("pan_number"))
    pan_ok = False
    pan_errs = []
    if pan_raw:
        pan_up = pan_raw.strip().upper().replace(" ", "")
        field_conf = confidences.get("pan_number") or confidences.get("pan_number_confidence") or None
        ok, ok_errs = pan_validate_with_checksum(pan_up, confidence=field_conf, min_confidence_override=0.90)
        if ok:
            pan_ok = True
        else:
            pan_errs.extend(ok_errs)
    else:
        pan_errs.append("pan_missing")

    validated["pan_number"] = {"value": pan_raw, "normalized_value": pan_up if pan_raw else None, "status": "VALID" if pan_ok else "INVALID", "confidence": float(confidences.get("pan_number", 0.0)), "validation_rules": ["regex_AAAAA9999A","checksum"], "errors": pan_errs}
    if not pan_ok:
        errors.append("pan_invalid")

    # DOB validation for PAN
    dob_raw = normalize_str_optional(fields.get("date_of_birth"))
    dob_ok = False
    dob_norm = None
    dob_errs = []
    if dob_raw:
        parsed = parse_date_dmy(dob_raw)
        if parsed:
            dob_norm = parsed.strftime("%d/%m/%Y")
            # If individual PAN (4th char == 'P') -> age >= 18
            if pan_raw and len(pan_raw.strip()) >= 4:
                fourth = pan_up[3]
                if fourth == "P":
                    age = calc_age_years(parsed, today_date())
                    if age < cfg.pan_min_age_individual:
                        dob_errs.append("pan_dob_underage_for_individual_pan")
                    else:
                        dob_ok = True
                else:
                    dob_ok = True
            else:
                dob_ok = True
        else:
            dob_errs.append("pan_dob_parse_failed")
    else:
        dob_errs.append("pan_dob_missing")
    validated["date_of_birth"] = {"value": dob_raw, "normalized_value": dob_norm, "status": "VALID" if dob_ok else "INVALID", "confidence": float(confidences.get("date_of_birth", 0.0)), "validation_rules": ["date_parse_ddmmyyyy","age_check_if_individual_pan"], "errors": dob_errs}
    if not dob_ok:
        errors.append("pan_dob_invalid")

    # signature_present: accept True/False/null; treat null as False
    sig_raw = fields.get("signature_present", None)
    sig_conf = float(confidences.get("signature_present", 0.0))
    sig_ok = True  # be permissive per new requirement
    sig_errs = []

    if sig_raw is None:
        # treat null as False (explicit requirement)
        sig_norm = False
    elif isinstance(sig_raw, bool):
        sig_norm = sig_raw
    elif isinstance(sig_raw, str):
        low = sig_raw.strip().lower()
        if low in ("true", "yes", "1", "y"):
            sig_norm = True
        elif low in ("false", "no", "0", "n"):
            sig_norm = False
        else:
            # unrecognized string -> treat as False but record a warning
            sig_norm = False
            sig_errs.append("signature_value_unrecognized_treated_as_false")
    else:
        # unexpected type -> coerce to False but record warning
        sig_norm = False
        sig_errs.append("signature_unexpected_type_treated_as_false")

    validated["signature_present"] = {
        "value": sig_raw,
        "normalized_value": sig_norm,
        "status": "VALID" if sig_ok else "INVALID",
        "confidence": sig_conf,
        "validation_rules": ["boolean_or_null_allowed", "null_treated_as_false"],
        "errors": sig_errs
    }
    # Do not auto-fail for null; treat as False (per update)
    return validated, errors


def validate_dl_front(fields: Dict[str, Any], confidences: Dict[str, float], cfg: ValidatorConfig) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate driving-license front fields.
    This function validates:
      - dl_number (existing logic)
      - date_of_birth (existing logic)
      - issue_date (if present) -> parse, not-in-future
      - valid_until (if present) -> parse, not-before-issue, expired flag if < today

    Returns:
      validated: mapping field -> validation object (value, normalized_value, status, confidence, validation_rules, errors)
      errors: list of short error codes (for document-level aggregation)
    """
    validated: Dict[str, Any] = {}
    errors: List[str] = []

    # --- DL number ---
    dl_raw = normalize_str_optional(fields.get("dl_number"))
    dl_ok, dl_errs = validate_dl_number(dl_raw)
    validated["dl_number"] = {
        "value": dl_raw,
        "normalized_value": dl_raw,
        "status": "VALID" if dl_ok else "INVALID",
        "confidence": float(confidences.get("dl_number", 0.0)),
        "validation_rules": ["state_code_prefix", "length_10_16"],
        "errors": dl_errs or []
    }
    if not dl_ok:
        errors.append("dl_number_invalid")

    # --- Date of birth ---
    dob_raw = normalize_str_optional(fields.get("date_of_birth"))
    dob_norm = None
    dob_ok = False
    dob_errs: List[str] = []
    if dob_raw:
        parsed = parse_date_dmy(dob_raw)
        if parsed:
            try:
                age = calc_age_years(parsed, today_date())
                if age < getattr(cfg, "dl_min_age", 18):
                    dob_errs.append("dl_dob_underage")
                else:
                    dob_ok = True
                    # normalized string in DD/MM/YYYY
                    dob_norm = parsed.strftime("%d/%m/%Y")
            except Exception:
                dob_errs.append("dl_dob_age_calc_failed")
        else:
            dob_errs.append("dl_dob_parse_failed")
    else:
        dob_errs.append("dl_dob_missing")

    validated["date_of_birth"] = {
        "value": dob_raw,
        "normalized_value": dob_norm,
        "status": "VALID" if dob_ok else "INVALID",
        "confidence": float(confidences.get("date_of_birth", 0.0)),
        "validation_rules": ["date_parse_ddmmyyyy", "min_age_18"],
        "errors": dob_errs or []
    }
    if not dob_ok:
        errors.append("dl_dob_invalid")

    # --- Issue date (optional) ---
    issue_raw = normalize_str_optional(fields.get("issue_date"))
    issue_norm = None
    issue_ok = False
    issue_errs: List[str] = []
    issue_dt = None
    if issue_raw:
        parsed_issue = parse_date_dmy(issue_raw)
        if parsed_issue:
            issue_dt = parsed_issue
            # check not in future
            try:
                if parsed_issue.date() > today_date().date():
                    issue_errs.append("issue_date_future")
                else:
                    issue_ok = True
                    issue_norm = parsed_issue.strftime("%d/%m/%Y")
            except Exception:
                # defensively flag but don't crash
                issue_errs.append("issue_date_future_check_failed")
        else:
            issue_errs.append("issue_date_parse_failed")
    # if issue_date missing, we do not force an error — front may or may not contain it
    validated["issue_date"] = {
        "value": issue_raw,
        "normalized_value": issue_norm,
        "status": "VALID" if issue_ok else ("INVALID" if issue_raw else "MISSING"),
        "confidence": float(confidences.get("issue_date", 0.0)),
        "validation_rules": ["date_parse_ddmmyyyy", "not_future"],
        "errors": issue_errs or []
    }
    if issue_errs:
        errors.extend(issue_errs)

    # --- Valid until / expiry (optional) ---
    valid_raw = normalize_str_optional(fields.get("valid_until"))
    valid_norm = None
    valid_ok = False
    valid_errs: List[str] = []
    valid_dt = None
    if valid_raw:
        parsed_valid = parse_date_dmy(valid_raw)
        if parsed_valid:
            valid_dt = parsed_valid
            # check not before issue_date (if issue_date known)
            try:
                if issue_dt and parsed_valid.date() < issue_dt.date():
                    valid_errs.append("valid_until_before_issue")
                # expired check: valid_until < today
                if parsed_valid.date() < today_date().date():
                    valid_errs.append("valid_until_expired")
                # mark ok if parsed and not invalid relative to issue date (we still allow expired to be signalled)
                if not any(e in valid_errs for e in ("valid_until_before_issue",)):
                    valid_ok = True
                    valid_norm = parsed_valid.strftime("%d/%m/%Y")
            except Exception:
                valid_errs.append("valid_until_checks_failed")
        else:
            valid_errs.append("valid_until_parse_failed")

    validated["valid_until"] = {
        "value": valid_raw,
        "normalized_value": valid_norm,
        "status": "VALID" if valid_ok else ("INVALID" if valid_raw else "MISSING"),
        "confidence": float(confidences.get("valid_until", 0.0)),
        "validation_rules": ["date_parse_ddmmyyyy", "not_before_issue", "not_expired_flagged"],
        "errors": valid_errs or []
    }
    if valid_errs:
        errors.extend(valid_errs)

    # Final: return validated map + accumulated short-codes for document-level errors
    return validated, errors


def validate_dl_back(fields: Dict[str, Any], confidences: Dict[str, float], cfg: ValidatorConfig) -> Tuple[Dict[str, Any], List[str]]:
    validated = {}
    errors = []
    issue_raw = normalize_str_optional(fields.get("issue_date"))
    valid_raw = normalize_str_optional(fields.get("valid_until"))
    issue_ok = False; valid_ok = False
    issue_norm = None; valid_norm = None
    issue_errs = []; valid_errs = []
    if issue_raw:
        p = parse_date_dmy(issue_raw)
        if p:
            if p.year < cfg.dl_issue_min_year:
                issue_errs.append("issue_date_before_1980")
            elif p > today_date():
                issue_errs.append("issue_date_in_future")
            else:
                issue_ok = True
                issue_norm = p.strftime("%d/%m/%Y")
        else:
            issue_errs.append("issue_date_parse_failed")
    else:
        issue_errs.append("issue_date_missing")

    if valid_raw:
        q = parse_date_dmy(valid_raw)
        if q:
            if q < today_date():
                valid_errs.append("valid_until_in_past")
            else:
                # compare to issue_date if both present
                if issue_ok and issue_norm:
                    pi = datetime.strptime(issue_norm, "%d/%m/%Y").replace(tzinfo=timezone.utc)
                    if q <= pi:
                        valid_errs.append("valid_until_not_after_issue")
                    else:
                        yrs = (q - pi).days / 365.2425
                        if yrs > cfg.dl_max_validity_years:
                            valid_errs.append("valid_until_too_long_gt_20yrs")
                        else:
                            valid_ok = True
                            valid_norm = q.strftime("%d/%m/%Y")
                else:
                    # no issue date present - still ensure not unreasonably far in future
                    valid_ok = True
                    valid_norm = q.strftime("%d/%m/%Y")
        else:
            valid_errs.append("valid_until_parse_failed")
    else:
        valid_errs.append("valid_until_missing")

    validated["issue_date"] = {"value": issue_raw, "normalized_value": issue_norm, "status": "VALID" if issue_ok else "INVALID", "confidence": float(confidences.get("issue_date", 0.0)), "validation_rules": ["date_parse",">'1980'","not_future"], "errors": issue_errs}
    validated["valid_until"] = {"value": valid_raw, "normalized_value": valid_norm, "status": "VALID" if valid_ok else "INVALID", "confidence": float(confidences.get("valid_until", 0.0)), "validation_rules": ["date_parse","after_issue_date","max_20_years"], "errors": valid_errs}
    if not issue_ok:
        errors.append("dl_issue_invalid")
    if not valid_ok:
        errors.append("dl_valid_until_invalid")
    return validated, errors


# -------------------------
# Cross-field checks
# -------------------------
def cross_field_checks(document: Dict[str, Any], validated_fields: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Perform cross-field checks:
    - DOB consistency across sides
    - Name presence/length checks (relaxed)
    - Address consistency basic checks (presence)
    - Date sequences (issue_date < valid_until)
    Returns (status: PASS|FAIL|WARNING, details)
    """
    checks = []
    errors = []
    status = "PASS"
    fields = document.get("extracted_data", {}).get("fields", {})

    # DOB consistency: if multiple occurrences (e.g., front/back) check same normalized value
    dob_norms = []
    for k, v in validated_fields.items():
        if k == "date_of_birth":
            val = v.get("normalized_value")
            if val:
                dob_norms.append(val)
    if len(set(dob_norms)) > 1:
        checks.append("dob_mismatch_across_sides")
        errors.append("dob_inconsistent")
        status = "FAIL"

    # Name presence/length (RELAXED):
    # Accept any kind of name; only flag if name missing everywhere or extremely short.
    name_keys = []
    for possible in ("name", "full_name", "customer_name", "applicant_name", "fathers_name", "guardian_name"):
        if possible in fields and fields.get(possible):
            name_keys.append(str(fields.get(possible)).strip())
    # If no name fields at all -> warning (incomplete)
    if not name_keys:
        checks.append("name_missing_all_fields")
        errors.append("name_missing")
        status = "WARNING" if status == "PASS" else status
    else:
        # check each found name for a minimum reasonable length (2 chars)
        short_names = [n for n in name_keys if len(n) < 2]
        if short_names:
            checks.append("name_too_short")
            errors.append("name_length_too_short")
            status = "WARNING" if status == "PASS" else status
        # otherwise accept different name forms as valid (no mismatch error)

    # license issue/validity date sequence check
    if "issue_date" in fields and "valid_until" in fields:
        i = parse_date_dmy(str(fields.get("issue_date") or ""))
        v = parse_date_dmy(str(fields.get("valid_until") or ""))
        if i and v and v <= i:
            checks.append("date_sequence_invalid")
            errors.append("issue_after_or_equal_to_valid_until")
            status = "FAIL"

    return status, {"checks_performed": checks, "errors": errors}


# -------------------------
# Confidence weighted scoring & overall decision
# -------------------------
def compute_weighted_confidence(conf_map: Dict[str, float]) -> float:
    if not conf_map:
        return 0.0
    vals = [float(v) for v in conf_map.values()]
    return float(round(sum(vals) / len(vals), 4))


def map_confidence_to_status(conf: float, cfg: ValidatorConfig) -> str:
    if conf >= cfg.conf_auto_approve:
        return "HIGH"
    if conf >= cfg.conf_manual_review:
        return "MEDIUM"
    if conf >= cfg.conf_low:
        return "LOW"
    return "REJECT"


# -------------------------
# Document-level validation routing
# -------------------------
def validate_document(doc: Dict[str, Any], cfg: ValidatorConfig) -> Dict[str, Any]:
    doc_id = doc.get("document_id")
    dtype = doc.get("document_type")
    side = doc.get("side", "single")
    extracted = doc.get("extracted_data", {}).get("fields", {})
    confs = doc.get("extracted_data", {}).get("confidence_scores", {})

    validated_fields = {}
    errors_accum = []
    warnings_accum = []

    # dispatch based on document type + side
    if dtype == "aadhar":
        # Treat "multi" like both sides
        if side in ("front", "single", "multi"):
            vf, errs = validate_aadhar_front(extracted, confs, cfg)
            validated_fields.update(vf)
            errors_accum.extend(errs)
        if side in ("back", "single", "multi"):
            vb, errs = validate_aadhar_back(extracted, confs)
            validated_fields.update(vb)
            errors_accum.extend(errs)
    elif dtype == "pan":
        vp, errs = validate_pan(extracted, confs, cfg)
        validated_fields.update(vp)
        errors_accum.extend(errs)
    elif dtype == "driving_license":
        """
        Simplified DL validation: only validate front-side fields. User will supply
        the front image containing all DL fields (dl_number, name, dob, address, issue_date, valid_until, etc.).
        If fields like issue_date/valid_until are present in extracted input, they will be validated by the front validator.
        """
        try:
            v_front, errs_front = validate_dl_front(extracted, confs, cfg)
        except Exception as e:
            # defensive: capture validator failure as an error so document goes to manual review if needed
            v_front = {}
            errs_front = [f"validate_dl_front_exception:{e}"]

        # Only front-based validated fields are accepted; back-side validator is NOT called.
        validated_fields.update(v_front or {})
        errors_accum.extend(errs_front or [])

    else:
        # unknown doc type -> mark incomplete
        errors_accum.append("document_type_unknown")

    # Cross-field checks
    cross_status, cross_details = cross_field_checks(doc, validated_fields)
    if cross_status == "FAIL":
        errors_accum.extend(cross_details.get("errors", []))
    elif cross_status == "WARNING":
        warnings_accum.extend(cross_details.get("errors", []))

    # compute confidence score
    confidence_score = compute_weighted_confidence(confs)
    # derive validation_status
    # NEW POLICY: only two final outcomes:
    #  - AUTO_APPROVE / APPROVED if overall confidence_score >= cfg.conf_auto_approve
    #  - MANUAL_REVIEW / REVIEW_REQUIRED for everything else
    # Note: we *do not* set REJECT here. All non-auto-approved docs go to manual review.
    try:
        if float(confidence_score) >= float(getattr(cfg, "conf_auto_approve", getattr(cfg, "conf_auto_approve", 0.90))):
            validation_status = "APPROVED"
            processing_recommendation = "AUTO_APPROVE"
        else:
            validation_status = "REVIEW_REQUIRED"
            processing_recommendation = "MANUAL_REVIEW"
    except Exception:
        # Defensive fallback: if something goes wrong computing thresholds, fall back to manual review
        validation_status = "REVIEW_REQUIRED"
        processing_recommendation = "MANUAL_REVIEW"

    # --- Ensure all extractor fields are present in validated_fields for traceability ---
    # If the extractor provided fields that the per-type validators did not explicitly validate,
    # include them here with the extractor's raw value & the extractor-provided confidence.
    # This ensures validated_fields is self-contained and includes confidences for all fields.
    try:
        # 'extracted' and 'confs' are already defined earlier in validate_document()
        if isinstance(extracted, dict):
            for k, v in extracted.items():
                if k not in validated_fields:
                    # take confidence from confs map if present
                    ext_conf = None
                    try:
                        ext_conf = float(confs.get(k, 0.0)) if isinstance(confs, dict) else 0.0
                    except Exception:
                        ext_conf = 0.0
                    validated_fields[k] = {
                        "value": v,
                        "normalized_value": v,
                        "status": None,                     # validator didn't validate this field
                        "confidence": float(ext_conf or 0.0),
                        "validation_rules": [],
                        "errors": []
                    }
    except Exception:
        # defensive: don't fail validation because of unexpected extractor structure
        pass

    # Build validated_fields output structure matching required schema
    validated_fields_out = {}
    # capture extractor raw values & confidences (if present) for easy auditing
    extractor_fields = doc.get("extracted_data", {}).get("fields", {}) if isinstance(doc, dict) else {}
    extractor_confs = doc.get("extracted_data", {}).get("confidence_scores", {}) if isinstance(doc, dict) else {}

    for fname, info in validated_fields.items():
        # safe-get original extracted info
        original_val = extractor_fields.get(fname) if isinstance(extractor_fields, dict) else None
        original_conf = extractor_confs.get(fname) if isinstance(extractor_confs, dict) else None

        validated_fields_out[fname] = {
            "value": info.get("value"),
            "normalized_value": info.get("normalized_value"),
            "status": info.get("status"),
            "confidence": float(info.get("confidence", 0.0)),
            "validation_rules": info.get("validation_rules", []),
            "errors": info.get("errors", []),
            # Added for better debugability: original extractor outputs (raw value & extractor confidence)
            "original_extracted_value": original_val,
            "original_extracted_confidence": float(original_conf) if original_conf is not None else None
        }

    doc_out = {
        "document_id": doc_id,
        "document_type": dtype,
        "validation_status": validation_status,
        "confidence_score": float(round(confidence_score, 4)),
        "validated_fields": validated_fields_out,
        "cross_field_validation": {
            "status": "PASS" if not cross_details.get("errors") else ("FAIL" if cross_status=="FAIL" else "WARNING"),
            "checks_performed": cross_details.get("checks_performed", []),
            "errors": cross_details.get("errors", [])
        },
        "processing_recommendation": processing_recommendation,
        "original_extracted": doc  # preserve full original extraction for audit
    }
    return doc_out


# -------------------------
# Batch processing entrypoint
# -------------------------
def validate_batch(extractor_output: Dict[str, Any], cfg: Optional[ValidatorConfig] = None) -> Dict[str, Any]:
    """
    Enhanced to validate input structure and provide clear error messages.
    """
    cfg = cfg or ValidatorConfig()
    start_total = time.perf_counter()

    # STRICT INPUT VALIDATION
    if not isinstance(extractor_output, dict):
        raise ValueError("Input must be a dict as produced by vision_extractor.extract_documents()")

    if "documents" not in extractor_output:
        raise ValueError("Extractor output missing 'documents' key")

    docs = extractor_output.get("documents", []) or []

    # Validate each document structure
    for i, doc in enumerate(docs):
        if not isinstance(doc, dict):
            raise ValueError(f"Document {i} is not a dictionary")
        
        required_keys = ["document_id", "document_type", "extracted_data"]
        for key in required_keys:
            if key not in doc:
                raise ValueError(f"Document {i} missing required key: {key}")
        
        # Validate extracted_data structure
        ext_data = doc.get("extracted_data", {})
        if not isinstance(ext_data, dict):
            raise ValueError(f"Document {doc.get('document_id')} has invalid extracted_data")
            
        if "fields" not in ext_data:
            raise ValueError(f"Document {doc.get('document_id')} missing 'fields' in extracted_data")

    docs = extractor_output.get("documents", []) or []
    results = []
    errors = []
    tstart = time.perf_counter()
    # parallel processing
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        futures = {ex.submit(validate_document, d, cfg): d.get("document_id", str(uuid.uuid4())) for d in docs}
        for fut in as_completed(futures):
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                did = futures.get(fut, "unknown")
                log.exception("Validation failed for %s", did)
                errors.append({"document_id": did, "error": str(e)})

    # overall status aggregation
    counts = {"APPROVED":0,"REVIEW_REQUIRED":0,"REJECTED":0,"INCOMPLETE":0}
    total_conf = 0.0
    for r in results:
        s = r.get("validation_status")
        counts[s] = counts.get(s,0) + 1
        total_conf += float(r.get("confidence_score", 0.0))

    total_documents = len(results)
    avg_conf = float(round((total_conf / total_documents) if total_documents else 0.0,4))

    # derive overall_status using simple policy
    if counts["REJECTED"] > 0:
        overall_status = "REJECTED"
    elif counts["REVIEW_REQUIRED"] > 0:
        overall_status = "REVIEW_REQUIRED"
    elif counts["INCOMPLETE"] > 0:
        overall_status = "INCOMPLETE"
    else:
        overall_status = "APPROVED"

    end_total = time.perf_counter()
    validation_out = {
        "validation_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "overall_status": overall_status,
        "documents": results,
        "summary_metrics": {
            "total_documents": total_documents,
            "approved_count": counts.get("APPROVED",0),
            "review_required_count": counts.get("REVIEW_REQUIRED",0),
            "rejected_count": counts.get("REJECTED",0),
            "incomplete_count": counts.get("INCOMPLETE",0),
            "average_confidence": avg_conf,
            "processing_time_ms": float(round((end_total - start_total)*1000.0,4))
        },
        "processing_errors": errors,
        "next_actions": ["db_insert", "manual_review", "reprocess", "alert"]
    }
    return validation_out


# === Replace the existing __main__ CLI block with this ===
if __name__ == "__main__":
    import argparse, sys, os, re
    from datetime import timezone, datetime

    parser = argparse.ArgumentParser(description="Document Validator - validates vision extractor output JSON")
    parser.add_argument("--input", "-i", help="Path to extractor JSON file (or '-' for stdin)", required=True)
    parser.add_argument("--output", "-o", help="Path to write validation JSON (file or directory). Default stdout if '-' provided", default="-")
    args = parser.parse_args()

    # Read input JSON
    if args.input == "-":
        raw = sys.stdin.read()
    else:
        if not os.path.exists(args.input):
            print(f"Input file not found: {args.input}", file=sys.stderr)
            sys.exit(2)
        with open(args.input, "r", encoding="utf-8") as f:
            raw = f.read()

    try:
        inp = json.loads(raw)
    except Exception as e:
        print("Failed to parse input JSON:", e, file=sys.stderr)
        sys.exit(3)

    # Run validation
    out = validate_batch(inp, ValidatorConfig())

    # Use timezone-aware timestamp (replace any previous naive timestamps)
    try:
        out["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        pass

    txt = json.dumps(out, ensure_ascii=False, indent=2)

    # Handle output path
    if args.output == "-" or args.output is None:
        # Print to stdout
        print(txt)
        sys.exit(0)

    out_path = os.path.abspath(args.output)

    try:
        # Ensure parent directories exist when user provided a file path
        parent = out_path if os.path.isdir(out_path) else os.path.dirname(out_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        if os.path.isdir(out_path):
            # output is a directory -> derive filename from input
            if args.input and args.input != "-" and os.path.isfile(args.input):
                base = os.path.splitext(os.path.basename(args.input))[0]
                # strip common suffixes like ".extracted" or ".extracted.json"
                base = re.sub(r'\.extracted(?:\..*)?$', '', base, flags=re.IGNORECASE)
                filename = f"{base}.validation.json"
            else:
                filename = f"{out['validation_id']}.validation.json"
            dest = os.path.join(out_path, filename)
            with open(dest, "w", encoding="utf-8") as f:
                f.write(txt)
            print("Wrote validation output to", dest)
        else:
            # out_path is a file path specified by the user -> write directly
            # If user still wants derivation even when specifying a file path, change behavior here.
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(txt)
            print("Wrote validation output to", out_path)

    except PermissionError as pe:
        print(f"Permission error when writing output: {pe}", file=sys.stderr)
        print("Try running the script with a file path you have write access to, or run as an elevated user.", file=sys.stderr)
        sys.exit(4)
    except Exception as e:
        print(f"Failed to write output: {e}", file=sys.stderr)
        sys.exit(5)

