from __future__ import annotations

import os
import json
import uuid
import aiosqlite
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone


# -------------------------
# Configuration
# -------------------------
@dataclass
class DBConfig:
    database_url: str = field(default_factory=lambda: os.environ.get("DB_PATH", r"PATH TO DB"))
    pool_size: int = int(os.environ.get("DB_POOL_SIZE", "3"))
    connect_timeout_s: float = float(os.environ.get("DB_CONNECT_TIMEOUT", "10.0"))
    retry_attempts: int = int(os.environ.get("DB_RETRY_ATTEMPTS", "3"))
    retry_delay_s: float = float(os.environ.get("DB_RETRY_DELAY_S", "0.2"))
    isolation_level: Optional[str] = None  # aiosqlite default is autocommit; we use explicit transactions
    enable_logging: bool = True
    log_level: int = logging.INFO


# -------------------------
# Logging
# -------------------------
def _setup_logger(cfg: DBConfig) -> logging.Logger:
    logger = logging.getLogger("DBEngine")
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = "[DBEngine] %(asctime)s %(levelname)s - %(message)s"
        h.setFormatter(logging.Formatter(fmt))
        logger.addHandler(h)
    logger.setLevel(cfg.log_level if cfg.enable_logging else logging.CRITICAL)
    return logger


# -------------------------
# Schema Definition
# -------------------------
_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

-- documents (Master Table)
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    document_type TEXT NOT NULL CHECK (document_type IN ('aadhar','pan','driving_license')),
    original_filename TEXT,
    upload_timestamp TEXT, -- ISO8601 TZ
    processing_status TEXT NOT NULL CHECK (processing_status IN ('uploaded','processing','completed','failed')) DEFAULT 'uploaded',
    validation_status TEXT CHECK (validation_status IN ('APPROVED','REVIEW_REQUIRED','REJECTED')),
    final_status TEXT CHECK (final_status IN ('approved','rejected','pending_review')) DEFAULT 'pending_review',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- aadhar_documents
CREATE TABLE IF NOT EXISTS aadhar_documents (
    aadhar_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    uid_number TEXT,
    name TEXT,
    date_of_birth TEXT,
    gender TEXT,
    address TEXT,
    pin_code TEXT,
    state TEXT,
    uid_confidence REAL,
    name_confidence REAL,
    dob_confidence REAL,
    gender_confidence REAL,
    address_confidence REAL,
    pin_confidence REAL,
    state_confidence REAL,
    uid_status TEXT,
    name_status TEXT,
    dob_status TEXT,
    gender_status TEXT,
    address_status TEXT,
    pin_status TEXT,
    state_status TEXT,
    validated_fields_json TEXT,
    original_extracted_json TEXT,
    last_edited_by TEXT,
    last_edited_at TEXT,
    is_edited INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
);

-- pan_documents
CREATE TABLE IF NOT EXISTS pan_documents (
    pan_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    pan_number TEXT,
    name TEXT,
    fathers_name TEXT,
    date_of_birth TEXT,
    signature_present INTEGER,
    pan_number_confidence REAL,
    name_confidence REAL,
    fathers_name_confidence REAL,
    dob_confidence REAL,
    signature_confidence REAL,
    pan_number_status TEXT,
    name_status TEXT,
    fathers_name_status TEXT,
    dob_status TEXT,
    signature_status TEXT,
    validated_fields_json TEXT,  -- JSON dump of validated_fields from validator
    original_extracted_json TEXT,
    last_edited_by TEXT,
    last_edited_at TEXT,
    is_edited INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
);

-- driving_license_documents
CREATE TABLE IF NOT EXISTS driving_license_documents (
    dl_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    dl_number TEXT,
    name TEXT,
    fathers_name TEXT,
    date_of_birth TEXT,
    address TEXT,
    issue_date TEXT,
    valid_until TEXT,
    dl_number_confidence REAL,
    name_confidence REAL,
    fathers_name_confidence REAL,
    dob_confidence REAL,
    address_confidence REAL,
    issue_date_confidence REAL,
    valid_until_confidence REAL,
    dl_number_status TEXT,
    name_status TEXT,
    fathers_name_status TEXT,
    dob_status TEXT,
    address_status TEXT,
    issue_date_status TEXT,
    valid_until_status TEXT,
    validated_fields_json TEXT,
    original_extracted_json TEXT,
    last_edited_by TEXT,
    last_edited_at TEXT,
    is_edited INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
);

-- processing_audit
CREATE TABLE IF NOT EXISTS processing_audit (
    audit_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    processing_stage TEXT NOT NULL CHECK (processing_stage IN ('preprocessor','classifier','extractor','validator')),
    input_data TEXT, -- JSON as TEXT
    output_data TEXT, -- JSON as TEXT
    processing_time_ms REAL,
    success INTEGER NOT NULL DEFAULT 1,
    error_message TEXT,
    timestamp TEXT NOT NULL,
    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
);

-- field_edit_history
CREATE TABLE IF NOT EXISTS field_edit_history (
    edit_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    table_name TEXT NOT NULL,
    field_name TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    edited_by TEXT NOT NULL,
    edit_timestamp TEXT NOT NULL,
    reason TEXT,
    confidence_override INTEGER DEFAULT 0,
    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
);

-- review_queue
CREATE TABLE IF NOT EXISTS review_queue (
    review_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    assigned_to TEXT,
    priority INTEGER DEFAULT 5,
    review_status TEXT NOT NULL CHECK (review_status IN ('pending','in_progress','completed')) DEFAULT 'pending',
    review_notes TEXT,
    missing_field TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_documents_validation_status ON documents(validation_status);
CREATE INDEX IF NOT EXISTS idx_processing_audit_doc_stage ON processing_audit(document_id, processing_stage);
CREATE INDEX IF NOT EXISTS idx_review_queue_status ON review_queue(review_status, priority);
"""


# -------------------------
# Helper utilities
# -------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def gen_uuid() -> str:
    return str(uuid.uuid4())


def id_from_filename(original_name: Optional[str], prefix: Optional[str] = None, max_suffix_chars: int = 10) -> str:
    """
    Deterministic, human-friendly id based on the original filename (stem).
    Produces: <prefix>_<safe-stem>_<hexhash>
    - If original_name is None/empty, falls back to UUID.
    - Keeps id length reasonable using sha1 hex truncated to max_suffix_chars.
    """
    try:
        if not original_name:
            return gen_uuid()
        from pathlib import Path
        stem = Path(original_name).stem
        # normalize stem to safe characters: letters, digits, underscore
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_")[:64] or "file"
        # short hash for uniqueness & determinism
        import hashlib
        h = hashlib.sha1(original_name.encode("utf-8")).hexdigest()[:max_suffix_chars]
        if prefix:
            return f"{prefix}_{safe}_{h}"
        return f"{safe}_{h}"
    except Exception:
        return gen_uuid()


# -------------------------
# Exceptions
# -------------------------
class DBEngineError(Exception): pass
class TransactionError(DBEngineError): pass
class MigrationError(DBEngineError): pass
class NotFoundError(DBEngineError): pass


# -------------------------
# Simple connection pool wrapper (aiosqlite doesn't include pool natively)
# We'll create N connections and reuse them with an asyncio.Queue.
# -------------------------
class AioSqlitePool:
    def __init__(self, db_path: str, size: int = 3, timeout: float = 10.0, loop: Optional[asyncio.AbstractEventLoop] = None):
        self._db_path = db_path
        self._size = max(1, size)
        self._timeout = timeout
        self._pool: Optional[asyncio.Queue] = None
        self._initialized = False
        self._loop = loop or asyncio.get_event_loop()

    async def init(self):
        if self._initialized:
            return
        self._pool = asyncio.Queue(maxsize=self._size)
        # aiosqlite accepts URI when "uri=True" passed to connect. Our DB path may be a URI like "file:...?"
        # We'll use aiosqlite.connect(uri=True) if path starts with "file:" or contains "mode=" or "cache=".
        uri_flag = isinstance(self._db_path, str) and (self._db_path.startswith("file:") or ("mode=" in self._db_path))
        for _ in range(self._size):
            conn = await aiosqlite.connect(self._db_path, timeout=self._timeout, uri=uri_flag)
            # Enable WAL for concurrency
            await conn.execute("PRAGMA journal_mode=WAL;")
            await conn.execute("PRAGMA foreign_keys=ON;")
            await conn.commit()
            await self._pool.put(conn)
        self._initialized = True

    async def acquire(self) -> aiosqlite.Connection:
        if not self._initialized:
            await self.init()
        assert self._pool is not None
        conn = await self._pool.get()
        return conn

    async def release(self, conn: aiosqlite.Connection):
        # attempt to reset if closed
        if conn and not conn._running:
            # closed or invalid; try to re-open
            conn = await aiosqlite.connect(self._db_path, timeout=self._timeout)
        await self._pool.put(conn)

    async def close(self):
        if not self._initialized or self._pool is None:
            return
        while not self._pool.empty():
            conn = await self._pool.get()
            try:
                await conn.close()
            except Exception:
                pass
        self._initialized = False


# -------------------------
# DatabaseManager
# -------------------------
class DatabaseManager:
    def __init__(self, cfg: Optional[DBConfig] = None):
        self.cfg = cfg or DBConfig()
        self.logger = _setup_logger(self.cfg)
        self._pool = AioSqlitePool(self.cfg.database_url, size=self.cfg.pool_size, timeout=self.cfg.connect_timeout_s)
        self._migrations_applied = False

    async def init(self):
        await self._pool.init()
        await self.apply_migrations()

    async def close(self):
        await self._pool.close()

    async def apply_migrations(self):
        """
        Apply baseline schema migration. For now we simply execute _SCHEMA_SQL inside a transaction.
        """
        if self._migrations_applied:
            return
        conn = await self._pool.acquire()
        try:
            async with conn.execute("BEGIN"):
                for stmt in _SCHEMA_SQL.split(";"):
                    s = stmt.strip()
                    if not s:
                        continue
                    await conn.execute(s)
                await conn.commit()
                        # ensure review_queue.missing_field exists (migration for older DBs)
            try:
                cur = await conn.execute("PRAGMA table_info(review_queue)")
                cols = await cur.fetchall()
                colnames = [c[1] for c in cols]  # pragma returns (cid, name, type, ...)
                if "missing_field" not in colnames:
                    await conn.execute("ALTER TABLE review_queue ADD COLUMN missing_field TEXT")
                    await conn.commit()
                    self.logger.info("Migrated review_queue: added missing_field column.")
            except Exception:
                # don't fail migration on this non-critical step; log only
                self.logger.warning("Failed to ensure review_queue.missing_field column: continuing. Error logged.")

            self._migrations_applied = True
            self.logger.info("Applied DB schema migrations.")
        except Exception as e:
            await conn.rollback()
            raise MigrationError(f"Failed to apply migrations: {e}") from e
        finally:
            await self._pool.release(conn)

    # Low-level helpers
    async def _execute_with_retry(self, func, *args, **kwargs):
        attempts = 0
        last_exc = None
        while attempts < max(1, self.cfg.retry_attempts):
            try:
                return await func(*args, **kwargs)
            except aiosqlite.OperationalError as e:
                last_exc = e
                attempts += 1
                self.logger.warning("OperationalError, retrying %d/%d: %s", attempts, self.cfg.retry_attempts, e)
                await asyncio.sleep(self.cfg.retry_delay_s)
            except Exception as e:
                # Non-transient - raise
                raise
        raise DBEngineError(f"Operation failed after retries: {last_exc}")

    # Context manager for transactions
    class _tx:
        def __init__(self, outer: "DatabaseManager"):
            self.outer = outer
            self.conn: Optional[aiosqlite.Connection] = None

        async def __aenter__(self):
            self.conn = await self.outer._pool.acquire()
            await self.conn.execute("BEGIN")
            return self.conn

        async def __aexit__(self, exc_type, exc, tb):
            try:
                if exc:
                    await self.conn.rollback()
                else:
                    await self.conn.commit()
            finally:
                await self.outer._pool.release(self.conn)

    def transaction(self):
        return DatabaseManager._tx(self)

    # Utility: simple insert helpers
    async def upsert_document_master(self, document_id: str, document_type: str, original_filename: Optional[str],
                                     upload_timestamp: Optional[str], processing_status: str,
                                     validation_status: Optional[str], final_status: Optional[str]):
        now = now_iso()
        conn = await self._pool.acquire()
        try:
            # Insert or update
            await conn.execute("""
                INSERT INTO documents(document_id, document_type, original_filename, upload_timestamp,
                                      processing_status, validation_status, final_status, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                  document_type=excluded.document_type,
                  original_filename=excluded.original_filename,
                  upload_timestamp=excluded.upload_timestamp,
                  processing_status=excluded.processing_status,
                  validation_status=excluded.validation_status,
                  final_status=excluded.final_status,
                  updated_at=excluded.updated_at
            """, (document_id, document_type, original_filename, upload_timestamp, processing_status, validation_status, final_status, now, now))
            await conn.commit()
        except Exception as e:
            await conn.rollback()
            raise
        finally:
            await self._pool.release(conn)

    async def insert_processing_audit(self, document_id: str, processing_stage: str,
                                      input_data: Dict[str, Any], output_data: Dict[str, Any],
                                      processing_time_ms: float, success: bool, error_message: Optional[str] = None):
        audit_id = None
        try:
            # attempt to fetch original_filename from documents table (non-blocking if not present)
            cur = await conn.execute("SELECT original_filename FROM documents WHERE document_id = ?", (document_id,))
            row = await cur.fetchone()
            orig_fn = row[0] if row and row[0] else None
            audit_id = id_from_filename(orig_fn or document_id, prefix="AUDIT")
        except Exception:
            audit_id = gen_uuid()
        ts = now_iso()
        conn = await self._pool.acquire()
        try:
            await conn.execute("""
                INSERT INTO processing_audit(audit_id, document_id, processing_stage, input_data, output_data,
                                            processing_time_ms, success, error_message, timestamp)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (audit_id, document_id, processing_stage, json.dumps(input_data, ensure_ascii=False), json.dumps(output_data, ensure_ascii=False),
                  float(processing_time_ms), 1 if success else 0, error_message, ts))
            await conn.commit()
            return audit_id
        except Exception as e:
            await conn.rollback()
            raise
        finally:
            await self._pool.release(conn)

    # -------------------------
    # Mapping validator output to tables
    # -------------------------
    async def store_validator_output(self, validator_json: Dict[str, Any], original_filename: Optional[str] = None) -> str:
        """
        Enhanced to handle all validation statuses and provide better error reporting.
        """
        if not isinstance(validator_json, dict):
            raise ValueError("validator_json must be a dict")

        validation_id = validator_json.get("validation_id", None)
        if not validation_id:
            if original_filename:
                validation_id = id_from_filename(original_filename, prefix="VALID")
            else:
                validation_id = gen_uuid()
        # If caller passed original_filename, make validation_id human-friendly: <filename_stem>_<uuid>
        if original_filename:
            try:
                from pathlib import Path
                stem = Path(original_filename).stem
                validation_id = f"{stem}_{validation_id}"
            except Exception:
                pass
        timestamp = validator_json.get("timestamp", now_iso())
        overall_status = validator_json.get("overall_status", "REJECTED")
        documents = validator_json.get("documents", []) or []

        # Validate document structures before transaction
        for doc in documents:
            if not isinstance(doc, dict):
                raise ValueError(f"Invalid document in validator output: {type(doc)}")
            
            required = ["document_id", "document_type", "validation_status", "validated_fields"]
            for key in required:
                if key not in doc:
                    raise ValueError(f"Document missing required key: {key}")

        # Process each document
        stored_docs = []
        async with self.transaction() as conn:
            try:
                for doc in documents:
                    doc_id = doc.get("document_id")
                    if not doc_id:
                        if original_filename:
                            doc_id = id_from_filename(original_filename, prefix=(doc.get("document_type") or "DOC"))
                        else:
                            doc_id = gen_uuid()
                    doc_type = doc.get("document_type")
                    validation_status = doc.get("validation_status")
                    confidence_score = float(doc.get("confidence_score", 0.0))
                    validated_fields = doc.get("validated_fields", {})
                    processing_recommendation = doc.get("processing_recommendation", "MANUAL_REVIEW")
                    original_extracted = doc.get("original_extracted", {})

                    # Insert master documents record
                    now_ts = now_iso()
                    await conn.execute("""
                        INSERT INTO documents(document_id, document_type, original_filename, upload_timestamp,
                                              processing_status, validation_status, final_status, created_at, updated_at)
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(document_id) DO UPDATE SET
                          document_type=excluded.document_type,
                          original_filename=excluded.original_filename,
                          updated_at=excluded.updated_at,
                          validation_status=excluded.validation_status
                    """, (
                        doc_id,
                        doc_type,
                        original_filename,
                        timestamp,
                        "completed" if validation_status == "APPROVED" else "processing",
                        validation_status,
                        "approved" if validation_status == "APPROVED" else ("pending_review" if validation_status == "REVIEW_REQUIRED" else "rejected"),
                        now_ts, now_ts
                    ))

                    # Choose per-type insertion
                    created_at = now_ts
                    if doc_type == "aadhar":
                        # Build column values - keep None when missing
                        # Map validated_fields keys to columns, confidences from nested info
                        uid_v = validated_fields.get("uid_number", {})
                        name_v = validated_fields.get("name", {})
                        dob_v = validated_fields.get("date_of_birth", {})
                        gender_v = validated_fields.get("gender", {})
                        addr_v = validated_fields.get("address", {})
                        pin_v = validated_fields.get("pin_code", {})
                        state_v = validated_fields.get("state", {})
                        aadhar_id = id_from_filename(original_filename or doc_id, prefix="Aadhar")
                        # Prepare JSON backups for auditing
                        validated_json = json.dumps(validated_fields, ensure_ascii=False)
                        original_extracted_json = json.dumps(original_extracted, ensure_ascii=False)

                        _values_tuple = (
                            aadhar_id, doc_id,
                            uid_v.get("normalized_value") if isinstance(uid_v, dict) else uid_v,
                            name_v.get("normalized_value") if isinstance(name_v, dict) else name_v,
                            dob_v.get("normalized_value") if isinstance(dob_v, dict) else dob_v,
                            gender_v.get("normalized_value") if isinstance(gender_v, dict) else gender_v,
                            addr_v.get("normalized_value") if isinstance(addr_v, dict) else addr_v,
                            pin_v.get("normalized_value") if isinstance(pin_v, dict) else pin_v,
                            state_v.get("normalized_value") if isinstance(state_v, dict) else state_v,
                            float(uid_v.get("confidence", 0.0) if isinstance(uid_v, dict) else 0.0),
                            float(name_v.get("confidence", 0.0) if isinstance(name_v, dict) else 0.0),
                            float(dob_v.get("confidence", 0.0) if isinstance(dob_v, dict) else 0.0),
                            float(gender_v.get("confidence", 0.0) if isinstance(gender_v, dict) else 0.0),
                            float(addr_v.get("confidence", 0.0) if isinstance(addr_v, dict) else 0.0),
                            float(pin_v.get("confidence", 0.0) if isinstance(pin_v, dict) else 0.0),
                            float(state_v.get("confidence", 0.0) if isinstance(state_v, dict) else 0.0),
                            (uid_v.get("status") if isinstance(uid_v, dict) else None),
                            (name_v.get("status") if isinstance(name_v, dict) else None),
                            (dob_v.get("status") if isinstance(dob_v, dict) else None),
                            (gender_v.get("status") if isinstance(gender_v, dict) else None),
                            (addr_v.get("status") if isinstance(addr_v, dict) else None),
                            (pin_v.get("status") if isinstance(pin_v, dict) else None),
                            (state_v.get("status") if isinstance(state_v, dict) else None),
                            validated_json,
                            original_extracted_json,
                            None,  # last_edited_by
                            None,  # last_edited_at
                            0,     # is_edited
                            created_at
                        )

                        # Aadhar table has 29 columns total
                        _expected_cols = 29
                        if len(_values_tuple) != _expected_cols:
                            raise RuntimeError(f"aadhar INSERT mismatch: values={len(_values_tuple)} expected_cols={_expected_cols}")

                        await conn.execute("""
                            INSERT INTO aadhar_documents(
                                aadhar_id, document_id, uid_number, name, date_of_birth, gender, address, pin_code, state,
                                uid_confidence, name_confidence, dob_confidence, gender_confidence,
                                address_confidence, pin_confidence, state_confidence,
                                uid_status, name_status, dob_status, gender_status, address_status, pin_status, state_status,
                                validated_fields_json, original_extracted_json,
                                last_edited_by, last_edited_at, is_edited, created_at
                            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, _values_tuple)

                    elif doc_type == "pan":
                        pan_v = validated_fields.get("pan_number", {})
                        name_v = validated_fields.get("name", {})
                        fathers_v = validated_fields.get("fathers_name", {})
                        dob_v = validated_fields.get("date_of_birth", {})
                        signature_v = validated_fields.get("signature_present", {})
                        pan_id = id_from_filename(original_filename or doc_id, prefix="PAN")
                        # precompute boolean normalization for signature_present
                        sig_norm = None
                        if isinstance(signature_v, dict):
                            sig_val = signature_v.get("normalized_value", signature_v.get("value"))
                        else:
                            sig_val = signature_v
                        if isinstance(sig_val, bool):
                            sig_norm = 1 if sig_val else 0
                        elif sig_val in (1, 0):
                            sig_norm = int(sig_val)
                        else:
                            sig_norm = None

                        validated_json = json.dumps(validated_fields, ensure_ascii=False)
                        original_extracted_json = json.dumps(original_extracted, ensure_ascii=False)

                        await conn.execute("""
                            INSERT INTO pan_documents(
                            pan_id, document_id, pan_number, name, fathers_name, date_of_birth, signature_present,
                            pan_number_confidence, name_confidence, fathers_name_confidence, dob_confidence, signature_confidence,
                            pan_number_status, name_status, fathers_name_status, dob_status, signature_status,
                            validated_fields_json, original_extracted_json,
                            last_edited_by, last_edited_at, is_edited, created_at
                            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            pan_id, doc_id,
                            pan_v.get("normalized_value") if isinstance(pan_v, dict) else pan_v,
                            name_v.get("normalized_value") if isinstance(name_v, dict) else name_v,
                            fathers_v.get("normalized_value") if isinstance(fathers_v, dict) else fathers_v,
                            dob_v.get("normalized_value") if isinstance(dob_v, dict) else dob_v,
                            sig_norm,
                            float(pan_v.get("confidence", 0.0) if isinstance(pan_v, dict) else 0.0),
                            float(name_v.get("confidence", 0.0) if isinstance(name_v, dict) else 0.0),
                            float(fathers_v.get("confidence", 0.0) if isinstance(fathers_v, dict) else 0.0),
                            float(dob_v.get("confidence", 0.0) if isinstance(dob_v, dict) else 0.0),
                            float(signature_v.get("confidence", 0.0) if isinstance(signature_v, dict) else 0.0),
                            (pan_v.get("status") if isinstance(pan_v, dict) else None),
                            (name_v.get("status") if isinstance(name_v, dict) else None),
                            (fathers_v.get("status") if isinstance(fathers_v, dict) else None),
                            (dob_v.get("status") if isinstance(dob_v, dict) else None),
                            (signature_v.get("status") if isinstance(signature_v, dict) else None),
                            validated_json,
                            original_extracted_json,
                            None, None, 0, created_at
                        ))

                    elif doc_type == "driving_license":
                        dl_v = validated_fields.get("dl_number", {})
                        name_v = validated_fields.get("name", {})
                        fathers_v = validated_fields.get("fathers_name", {})
                        dob_v = validated_fields.get("date_of_birth", {})
                        addr_v = validated_fields.get("address", {})
                        issue_v = validated_fields.get("issue_date", {})
                        valid_v = validated_fields.get("valid_until", {})
                        dl_id = id_from_filename(original_filename or doc_id, prefix="DL")
                        # Prepare JSON backups for auditing
                        validated_json = json.dumps(validated_fields, ensure_ascii=False)
                        original_extracted_json = json.dumps(original_extracted, ensure_ascii=False)

                        _values_tuple = (
                            dl_id, doc_id,
                            dl_v.get("normalized_value") if isinstance(dl_v, dict) else dl_v,
                            name_v.get("normalized_value") if isinstance(name_v, dict) else name_v,
                            fathers_v.get("normalized_value") if isinstance(fathers_v, dict) else fathers_v,
                            dob_v.get("normalized_value") if isinstance(dob_v, dict) else dob_v,
                            addr_v.get("normalized_value") if isinstance(addr_v, dict) else addr_v,
                            issue_v.get("normalized_value") if isinstance(issue_v, dict) else issue_v,
                            valid_v.get("normalized_value") if isinstance(valid_v, dict) else valid_v,
                            float(dl_v.get("confidence", 0.0) if isinstance(dl_v, dict) else 0.0),
                            float(name_v.get("confidence", 0.0) if isinstance(name_v, dict) else 0.0),
                            float(fathers_v.get("confidence", 0.0) if isinstance(fathers_v, dict) else 0.0),
                            float(dob_v.get("confidence", 0.0) if isinstance(dob_v, dict) else 0.0),
                            float(addr_v.get("confidence", 0.0) if isinstance(addr_v, dict) else 0.0),
                            float(issue_v.get("confidence", 0.0) if isinstance(issue_v, dict) else 0.0),
                            float(valid_v.get("confidence", 0.0) if isinstance(valid_v, dict) else 0.0),
                            (dl_v.get("status") if isinstance(dl_v, dict) else None),
                            (name_v.get("status") if isinstance(name_v, dict) else None),
                            (fathers_v.get("status") if isinstance(fathers_v, dict) else None),
                            (dob_v.get("status") if isinstance(dob_v, dict) else None),
                            (addr_v.get("status") if isinstance(addr_v, dict) else None),
                            (issue_v.get("status") if isinstance(issue_v, dict) else None),
                            (valid_v.get("status") if isinstance(valid_v, dict) else None),
                            validated_json,
                            original_extracted_json,
                            None,  # last_edited_by
                            None,  # last_edited_at
                            0,     # is_edited
                            created_at
                        )

                        _expected_cols = 29
                        if len(_values_tuple) != _expected_cols:
                            raise RuntimeError(f"driving_license INSERT mismatch: values={len(_values_tuple)} expected_cols={_expected_cols}")

                        await conn.execute("""
                            INSERT INTO driving_license_documents(
                                dl_id, document_id, dl_number, name, fathers_name, date_of_birth, address, issue_date, valid_until,
                                dl_number_confidence, name_confidence, fathers_name_confidence, dob_confidence, address_confidence,
                                issue_date_confidence, valid_until_confidence,
                                dl_number_status, name_status, fathers_name_status, dob_status, address_status, issue_date_status, valid_until_status,
                                validated_fields_json, original_extracted_json,
                                last_edited_by, last_edited_at, is_edited, created_at
                            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, _values_tuple)
                    else:
                        # Unknown type: create a document master only and an audit row for original_extracted
                        pass

                    # Build a richer audit output: include validated_fields JSON and original_extracted for traceability
                    audit_output = {
                        "validation_status": validation_status,
                        "recommendation": processing_recommendation,
                        "confidence_score": confidence_score,
                        "validated_fields": validated_fields,     # already in the structure passed to this function
                        "original_extracted": original_extracted
                    }
                    await conn.execute("""
                        INSERT INTO processing_audit(audit_id, document_id, processing_stage, input_data, output_data, processing_time_ms, success, error_message, timestamp)
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (gen_uuid(), doc_id, "validator", json.dumps(original_extracted, ensure_ascii=False),
                        json.dumps(audit_output, ensure_ascii=False),
                        0.0, 1 if validation_status != "REJECTED" else 0, None, now_iso()))

                    # utility local helper to pick best value for a field
                    def _pick_field(field_key: str):
                        # prefer validated_fields if present (dict with normalized/value/confidence/status)
                        vf = validated_fields.get(field_key, None)
                        val = None
                        conf = 0.0
                        status = None
                        if isinstance(vf, dict):
                            # typical validator format
                            val = vf.get("normalized_value") or vf.get("value") or None
                            # validator may have set a confidence; use it when present
                            try:
                                conf = float(vf.get("confidence", 0.0) or 0.0)
                            except Exception:
                                conf = 0.0
                            status = vf.get("status")
                        elif vf is not None:
                            # could be raw string from validated_fields structure
                            val = vf
                            conf = 0.0
                            status = None

                        # fallback to original_extracted (structure: original_extracted['extracted_data']['fields'])
                        if (val is None or val == "") and isinstance(original_extracted, dict):
                            try:
                                val = original_extracted.get("extracted_data", {}).get("fields", {}).get(field_key, val)
                            except Exception:
                                pass

                        # NEW: fallback to extractor confidence scores if validator didn't provide one
                        try:
                            # note: original_extracted may have confidence_scores map
                            ext_conf_map = original_extracted.get("extracted_data", {}).get("confidence_scores", {}) if isinstance(original_extracted, dict) else {}
                            if (conf is None or float(conf) == 0.0) and isinstance(ext_conf_map, dict):
                                # attempt to coerce to float, otherwise leave as 0.0
                                try:
                                    conf = float(ext_conf_map.get(field_key, conf or 0.0))
                                except Exception:
                                    conf = float(conf or 0.0)
                        except Exception:
                            # be defensive; don't crash on unexpected structures
                            try:
                                conf = float(conf or 0.0)
                            except Exception:
                                conf = 0.0

                        # Normalize booleans (signature_present) to ints for DB insertion when needed
                        return val, float(conf or 0.0), status

                    # Compute missing/low-confidence fields to populate review_queue.missing_field
                    missing_fields = []
                    _CONF_THRESHOLD = 0.4
                    # Determine schema for doc_type
                    schema_keys = []
                    if doc_type == "aadhar":
                        schema_keys = ["uid_number", "name", "date_of_birth", "gender", "address", "pin_code", "state"]
                    elif doc_type == "pan":
                        schema_keys = ["pan_number", "name", "fathers_name", "date_of_birth", "signature_present"]
                    elif doc_type == "driving_license":
                        schema_keys = ["dl_number", "name", "fathers_name", "date_of_birth", "address", "issue_date", "valid_until"]
                    self.logger.debug("Validated fields keys: %s; original_extracted fields: %s", list(validated_fields.keys()), list(original_extracted.get("extracted_data", {}).get("fields", {}).keys() if isinstance(original_extracted, dict) else []))
                    for fk in schema_keys:
                        v_norm, v_conf, v_status = _pick_field(fk)
                        # missing if normalized/parsed value is None or empty
                        if v_norm in (None, "", "NULL"):
                            missing_fields.append({"field": fk, "reason": "missing", "value": v_norm, "confidence": v_conf, "status": v_status})
                        else:
                            # low confidence fallback
                            try:
                                if float(v_conf) < _CONF_THRESHOLD:
                                    missing_fields.append({"field": fk, "reason": "low_confidence", "value": v_norm, "confidence": v_conf, "status": v_status})
                            except Exception:
                                pass

                    # If review required, add to review_queue and include missing_field JSON
                    if processing_recommendation in ("MANUAL_REVIEW",) or validation_status == "REVIEW_REQUIRED" or missing_fields:
                        review_id = review_id = id_from_filename(original_filename or doc_id, prefix="REVIEW")
                        created_at = now_iso()
                        missing_field_json = json.dumps(missing_fields, ensure_ascii=False)
                        await conn.execute("""
                            INSERT INTO review_queue(review_id, document_id, assigned_to, priority, review_status, review_notes, missing_field, created_at)
                            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                        """, (review_id, doc_id, None, 5, "pending", "Auto-added by validator", missing_field_json, created_at))

                    stored_docs.append(doc_id)
                # commit done by outer context manager
            except Exception:
                await conn.rollback()
                raise

        # return validation id for reference
        return validation_id

    # -------------------------
    # Query & Retrieval functions
    # -------------------------
    async def get_document_for_review(self, document_id: str) -> Dict[str, Any]:
        """
        Returns the document master row, per-type fields, processing_audit entries and field_edit_history.
        Highlights fields needing attention (INVALID, low confidence).
        """
        conn = await self._pool.acquire()
        try:
            cur = await conn.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,))
            master = await cur.fetchone()
            if not master:
                raise NotFoundError(f"Document not found: {document_id}")
            colnames = [d[0] for d in cur.description] if cur.description else []
            master_dict = dict(zip(colnames, master))

            # fetch processing_audit
            cur = await conn.execute("SELECT * FROM processing_audit WHERE document_id = ? ORDER BY timestamp DESC", (document_id,))
            audits = await cur.fetchall()
            audit_cols = [d[0] for d in cur.description] if cur.description else []
            audits_list = [dict(zip(audit_cols, a)) for a in audits]

            # fetch edit history
            cur = await conn.execute("SELECT * FROM field_edit_history WHERE document_id = ? ORDER BY edit_timestamp DESC", (document_id,))
            edits = await cur.fetchall()
            edit_cols = [d[0] for d in cur.description] if cur.description else []
            edits_list = [dict(zip(edit_cols, e)) for e in edits]

            # fetch per-type details
            per_type = {}
            if master_dict.get("document_type") == "aadhar":
                cur = await conn.execute("SELECT * FROM aadhar_documents WHERE document_id = ?", (document_id,))
                row = await cur.fetchone()
                if row:
                    cols = [d[0] for d in cur.description]
                    per_type['aadhar'] = dict(zip(cols, row))
            elif master_dict.get("document_type") == "pan":
                cur = await conn.execute("SELECT * FROM pan_documents WHERE document_id = ?", (document_id,))
                row = await cur.fetchone()
                if row:
                    cols = [d[0] for d in cur.description]
                    per_type['pan'] = dict(zip(cols, row))
            elif master_dict.get("document_type") == "driving_license":
                cur = await conn.execute("SELECT * FROM driving_license_documents WHERE document_id = ?", (document_id,))
                row = await cur.fetchone()
                if row:
                    cols = [d[0] for d in cur.description]
                    per_type['driving_license'] = dict(zip(cols, row))

            # Highlight fields requiring attention
            attention = []
            # Look for _status columns with INVALID or MANUAL_OVERRIDE, and confidence < 0.6
            # Gather from per_type row
            for tname, pdata in per_type.items():
                for k, v in pdata.items():
                    if k.endswith("_status") and v and str(v).upper() in ("INVALID", "REQUIRES_REVIEW", "MANUAL_OVERRIDE"):
                        base_field = k[:-7]
                        attention.append({"field": base_field, "issue": v})
                # confidences
                for k, v in pdata.items():
                    if k.endswith("_confidence") and v is not None:
                        try:
                            vv = float(v)
                            if vv < 0.6:
                                base_field = k[:-11]
                                attention.append({"field": base_field, "issue": f"low_confidence_{vv:.3f}"})
                        except Exception:
                            continue

            return {
                "master": master_dict,
                "per_type": per_type,
                "processing_audit": audits_list,
                "edit_history": edits_list,
                "attention": attention
            }
        finally:
            await self._pool.release(conn)

    async def count_documents_by_status(self) -> Dict[str, Any]:
        conn = await self._pool.acquire()
        try:
            cur = await conn.execute("SELECT document_type, validation_status, COUNT(*) as cnt FROM documents GROUP BY document_type, validation_status")
            rows = await cur.fetchall()
            res = {}
            for r in rows:
                dt, vs, cnt = r
                res.setdefault(dt, {})[vs] = cnt
            return res
        finally:
            await self._pool.release(conn)

    async def average_confidence_by_type(self) -> Dict[str, float]:
        conn = await self._pool.acquire()
        try:
            # For each type, average confidence across per-type confidence columns (best-effort)
            out = {}
            # Aadhaar
            cur = await conn.execute("""
                SELECT AVG((uid_confidence + name_confidence + dob_confidence + gender_confidence + IFNULL(address_confidence,0) + IFNULL(pin_confidence,0) + IFNULL(state_confidence,0)) / 7.0)
                FROM aadhar_documents
            """)
            r = await cur.fetchone()
            out['aadhar'] = float(r[0]) if r and r[0] is not None else 0.0
            # PAN
            cur = await conn.execute("""
                SELECT AVG((pan_number_confidence + name_confidence + fathers_name_confidence + dob_confidence + signature_confidence) / 5.0) FROM pan_documents
            """)
            r = await cur.fetchone()
            out['pan'] = float(r[0]) if r and r[0] is not None else 0.0
            # DL
            cur = await conn.execute("""
                SELECT AVG((dl_number_confidence + name_confidence + fathers_name_confidence + dob_confidence + IFNULL(address_confidence,0) + IFNULL(issue_date_confidence,0) + IFNULL(valid_until_confidence,0)) / 7.0)
                FROM driving_license_documents
            """)
            r = await cur.fetchone()
            out['driving_license'] = float(r[0]) if r and r[0] is not None else 0.0
            return out
        finally:
            await self._pool.release(conn)


# -------------------------
# FieldEditor
# -------------------------
class FieldEditor:
    """
    Supports manual editing of any field in per-type tables and records edit history.
    Ensures transactional safety: edits and history writes occur in a single transaction.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db
        self.logger = db.logger

    async def edit_field(self, document_id: str, table_name: str, field_name: str,
                         new_value: Any, edited_by: str, reason: Optional[str] = None,
                         confidence_override: Optional[bool] = True) -> Dict[str, Any]:
        """
        Edits a single field in given table (aadhar_documents, pan_documents, driving_license_documents).
        - Stores old value in field_edit_history
        - Sets is_edited=1, last_edited_by, last_edited_at in target row
        - If confidence_override True -> set corresponding _confidence to NULL or leave as-is (we set to NULL to indicate override)
        - Set field status to "MANUAL_OVERRIDE"
        Returns details about the edit performed.
        """
        allowed_tables = {"aadhar_documents", "pan_documents", "driving_license_documents"}
        if table_name not in allowed_tables:
            raise ValueError(f"Unsupported table for editing: {table_name}")

        edit_id = gen_uuid()
        ts = now_iso()

        async with self.db.transaction() as conn:
            # fetch existing row
            cur = await conn.execute(f"SELECT * FROM {table_name} WHERE document_id = ?", (document_id,))
            row = await cur.fetchone()
            if not row:
                raise NotFoundError(f"No record in {table_name} for document_id {document_id}")
            cols = [d[0] for d in cur.description]
            row_map = dict(zip(cols, row))
            old_value = row_map.get(field_name)

            # Prepare update statements
            # 1) update the field
            await conn.execute(f"UPDATE {table_name} SET {field_name} = ? WHERE document_id = ?", (json.dumps(new_value, ensure_ascii=False) if isinstance(new_value, (dict,list)) else new_value, document_id))
            # 2) mark status field to MANUAL_OVERRIDE if exists
            status_col = f"{field_name}_status"
            try:
                if status_col in cols:
                    await conn.execute(f"UPDATE {table_name} SET {status_col} = ? WHERE document_id = ?", ("MANUAL_OVERRIDE", document_id))
            except Exception:
                # ignore if status col doesn't exist or update fails
                pass
            # 3) if confidence_override requested and confidence column exists -> set to null or an override marker
            conf_col = f"{field_name}_confidence"
            if confidence_override and conf_col in cols:
                await conn.execute(f"UPDATE {table_name} SET {conf_col} = NULL WHERE document_id = ?", (document_id,))

            # 4) update edit-tracking columns
            await conn.execute(f"UPDATE {table_name} SET is_edited = 1, last_edited_by = ?, last_edited_at = ? WHERE document_id = ?", (edited_by, ts, document_id))

            # insert edit history row
            await conn.execute("""
                INSERT INTO field_edit_history(edit_id, document_id, table_name, field_name, old_value, new_value, edited_by, edit_timestamp, reason, confidence_override)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                edit_id, document_id, table_name, field_name,
                json.dumps(old_value, ensure_ascii=False) if old_value is not None else None,
                json.dumps(new_value, ensure_ascii=False) if new_value is not None else None,
                edited_by, ts, reason, 1 if confidence_override else 0
            ))

        return {
            "edit_id": edit_id,
            "document_id": document_id,
            "table_name": table_name,
            "field_name": field_name,
            "old_value": old_value,
            "new_value": new_value,
            "edited_by": edited_by,
            "edit_timestamp": ts,
            "confidence_override": bool(confidence_override)
        }


# -------------------------
# ReviewManager
# -------------------------
class ReviewManager:
    """
    Manage review queue: assign, start, complete reviews.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db
        self.logger = db.logger

    async def assign_review(self, review_id: str, assigned_to: str) -> None:
        conn = await self.db._pool.acquire()
        try:
            await conn.execute("UPDATE review_queue SET assigned_to = ?, review_status = 'in_progress' WHERE review_id = ?", (assigned_to, review_id))
            await conn.commit()
        finally:
            await self.db._pool.release(conn)

    async def complete_review(self, review_id: str, completed_by: str, review_notes: Optional[str] = None, final_status: Optional[str] = None) -> None:
        conn = await self.db._pool.acquire()
        try:
            completed_at = now_iso()
            await conn.execute("UPDATE review_queue SET review_status = 'completed', completed_at = ?, review_notes = coalesce(review_notes, '') || ? WHERE review_id = ?", (completed_at, ("\n" + review_notes) if review_notes else "", review_id))
            # Optionally update documents.final_status to approved/rejected
            if final_status:
                # fetch document id
                cur = await conn.execute("SELECT document_id FROM review_queue WHERE review_id = ?", (review_id,))
                row = await cur.fetchone()
                if row:
                    document_id = row[0]
                    await conn.execute("UPDATE documents SET final_status = ? WHERE document_id = ?", (final_status, document_id))
            await conn.commit()
        finally:
            await self.db._pool.release(conn)

    async def fetch_next_pending(self, priority_threshold: Optional[int] = None) -> Optional[Dict[str, Any]]:
        conn = await self.db._pool.acquire()
        try:
            if priority_threshold is None:
                cur = await conn.execute("SELECT * FROM review_queue WHERE review_status = 'pending' ORDER BY priority ASC, created_at ASC LIMIT 1")
            else:
                cur = await conn.execute("SELECT * FROM review_queue WHERE review_status = 'pending' AND priority <= ? ORDER BY priority ASC, created_at ASC LIMIT 1", (priority_threshold,))
            row = await cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))
        finally:
            await self.db._pool.release(conn)


# -------------------------
# Example usage & simple async tests
# -------------------------
async def _example_flow():
    cfg = DBConfig()
    db = DatabaseManager(cfg)
    await db.init()

    # Example: store a validator output (minimal)
    example_validator = {
        "validation_id": gen_uuid(),
        "timestamp": now_iso(),
        "overall_status": "REVIEW_REQUIRED",
        "documents": [
            {
                "document_id": "DOC_EX_1",
                "document_type": "pan",
                "validation_status": "REVIEW_REQUIRED",
                "confidence_score": 0.65,
                "validated_fields": {
                    "pan_number": {"value": "ABCDE1234F", "normalized_value": "ABCDE1234F", "status": "VALID", "confidence": 0.85, "validation_rules": [], "errors": []},
                    "name": {"value": "Anurag Kaushik", "normalized_value": "Anurag Kaushik", "status": "VALID", "confidence": 0.9, "validation_rules": [], "errors": []},
                    "fathers_name": {"value": "Ramesh Kaushik", "normalized_value": "Ramesh Kaushik", "status": "VALID", "confidence": 0.88, "validation_rules": [], "errors": []},
                    "date_of_birth": {"value": "17/03/2003", "normalized_value": "17/03/2003", "status": "VALID", "confidence": 0.8, "validation_rules": [], "errors": []},
                    "signature_present": {"value": True, "normalized_value": True, "status": "VALID", "confidence": 0.7, "validation_rules": [], "errors": []}
                },
                "processing_recommendation": "MANUAL_REVIEW",
                "original_extracted": {"fields": {"pan_number": "ABCDE1234F"}}
            }
        ]
    }
    validation_id = await db.store_validator_output(example_validator, original_filename="sample_pan.pdf")
    print("Stored validation_id:", validation_id)

    # Retrieve document for review
    doc_review = await db.get_document_for_review("DOC_EX_1")
    print("Document for review:", json.dumps(doc_review, indent=2, default=str)[:2000])

    # Edit a field manually
    editor = FieldEditor(db)
    edit_res = await editor.edit_field(document_id="DOC_EX_1", table_name="pan_documents", field_name="name", new_value="Anurag K.", edited_by="user-123", reason="Shorten name", confidence_override=True)
    print("Edit result:", edit_res)

    # Fetch next pending review
    rm = ReviewManager(db)
    next_pending = await rm.fetch_next_pending()
    print("Next pending review:", next_pending)

    # Basic reporting
    counts = await db.count_documents_by_status()
    avg_conf = await db.average_confidence_by_type()
    print("Counts:", counts)
    print("Average confidences:", avg_conf)

    await db.close()

# Run example when executed as script
if __name__ == "__main__":
    import asyncio
    asyncio.run(_example_flow())
