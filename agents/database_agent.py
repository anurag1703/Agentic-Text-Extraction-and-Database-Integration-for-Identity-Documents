"""
database_agent.py

"""

from __future__ import annotations

import os
import sys
import time
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import asyncio

# Ensure project root on sys.path
HERE = os.path.dirname(os.path.abspath(__file__))          # agents/
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))   # project root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Bootstrap logger
_boot_logger = logging.getLogger("DatabaseAgentBootstrap")
if not _boot_logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[DatabaseAgentBootstrap] %(asctime)s %(levelname)s - %(message)s"))
    _boot_logger.addHandler(ch)
_boot_logger.setLevel(logging.INFO)

# Try to import Autogen AssistantAgent; fallback to stub
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

# Attempt to import DatabaseManager & DBConfig from user's db_engine
try:
    from tools.db_engine import DatabaseManager, DBConfig  # type: ignore
except Exception:
    try:
        from db_engine import DatabaseManager, DBConfig  # type: ignore
    except Exception as e:
        raise ImportError(
            "Could not import required symbols from db_engine.py. "
            "Make sure db_engine.py exists and exports DatabaseManager and DBConfig."
        ) from e

# Module logger
logger = logging.getLogger("DatabaseAgent")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[DatabaseAgent] %(asctime)s %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# ---------------------------
# Local registry for tool functions (deferred registration)
# ---------------------------
_GLOBAL_TOOL_REGISTRY: List[Dict[str, Any]] = []

def register_function(func=None, *, name: Optional[str] = None, description: Optional[str] = None):
    """
    Safe decorator: collects function metadata. Actual Autogen registration is deferred.
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
class DatabaseAgentConfig:
    db_path: str = r"PATH TO DB"   # default SQLite file (relative)
    sqlite_journal_mode: Optional[str] = None
    ensure_tables: bool = True
    max_retries: int = 1
    retry_delay_s: float = 0.2
    verbose: bool = True


# ---------------------------
# Utilities: DB manager factory & sanitizers
# ---------------------------
def _build_db_config(agent_cfg: DatabaseAgentConfig, overrides: Optional[Dict[str, Any]]) -> DBConfig:
    db_cfg = DBConfig()
    try:
        # apply db_path if attribute exists
        if hasattr(db_cfg, "db_path"):
            setattr(db_cfg, "db_path", agent_cfg.db_path)
    except Exception:
        pass

    # apply overrides
    if overrides:
        for k, v in overrides.items():
            if hasattr(db_cfg, k):
                try:
                    setattr(db_cfg, k, v)
                except Exception:
                    logger.debug("Failed to apply DBConfig override %s=%s", k, v)
    return db_cfg

def _get_db_manager(agent_cfg: DatabaseAgentConfig, overrides: Optional[Dict[str, Any]] = None) -> Tuple[DatabaseManager, DBConfig]:
    db_cfg = _build_db_config(agent_cfg, overrides)
    dbm = DatabaseManager(db_cfg)
    # ensure tables if requested and supported
    try:
        if agent_cfg.ensure_tables and hasattr(dbm, "ensure_tables"):
            dbm.ensure_tables()
    except Exception:
        # not critical; continue
        logger.debug("ensure_tables call failed or not supported by DatabaseManager")
    return dbm, db_cfg

def _sanitize_for_storage(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert numpy types / lists to JSON-friendly primitives for DB insertion.
    Keeps record small by removing raw image arrays if present.
    """
    safe: Dict[str, Any] = {}
    for k, v in (record.items() if isinstance(record, dict) else []):
        try:
            # remove heavy image arrays
            if k in ("image_array", "image_bytes", "raw_image"):
                continue
            # basic JSON serializable types pass through
            if isinstance(v, (str, int, float, bool)) or v is None:
                safe[k] = v
            else:
                # try JSON serialization
                safe[k] = json.loads(json.dumps(v, default=str))
        except Exception:
            # fallback to string
            safe[k] = str(v)
    return safe


# ---------------------------
# Tool functions (decorated)
# ---------------------------
# --- Replace the existing store_documents function with this ---
@register_function(name="store_documents", description="Store validated documents/records in the database.")
def store_documents(records: List[Dict[str, Any]], agent_config: Optional[Dict[str, Any]] = None
                    ) -> Dict[str, Any]:
    """
    Persist a list of document records to the DB.

    This implementation constructs DatabaseManager inside an asyncio coroutine
    (via asyncio.run) so that any pool initialization that requires a running
    event loop succeeds.
    """
    import asyncio
    from typing import List, Dict, Any, Optional

    start = time.perf_counter()
    agent_cfg = DatabaseAgentConfig()
    if agent_config and isinstance(agent_config, dict):
        for k, v in agent_config.items():
            if hasattr(agent_cfg, k):
                try:
                    setattr(agent_cfg, k, v)
                except Exception:
                    logger.debug("Ignoring agent_config override %s=%s", k, v)

    # Prepare DB config
    try:
        db_cfg = _build_db_config(agent_cfg, None)
    except Exception as ex:
        logger.exception("Failed to build DBConfig: %s", ex)
        return {"status": "failed", "error": f"db_config_build_failed: {ex}"}
    
    # --- Normalize incoming payloads to common shapes ---
    # Accept:
    #   - list of records
    #   - dict with top-level "documents" -> treat as validator output
    #   - dict with "result": { "documents": [...] } -> unwrap to result
    #   - wrapper with "status" + "documents" -> keep as-is
    # This ensures orchestrator wrappers like {"status":"success","result":{...}} are handled.
    try:
        normalized_records = records
        if isinstance(records, dict):
            # common wrapper: {"status": "...", "result": {...}}
            if "result" in records and isinstance(records["result"], dict):
                normalized_records = records["result"]
            # sometimes validators return {"status": "...", "documents": [...]}
            elif "documents" in records:
                normalized_records = records
        # else if it's a list, leave as-is
    except Exception:
        normalized_records = records

    logger.info("store_documents: normalized_records type=%s keys=%s", type(normalized_records).__name__,
            list(normalized_records.keys())[:10] if isinstance(normalized_records, dict) else f"len={len(normalized_records) if hasattr(normalized_records,'__len__') else 'unknown'}")

    # Define async worker that constructs DatabaseManager inside running loop
    async def _store_async(db_cfg, recs: List[Dict[str, Any]]):
        dbm = DatabaseManager(db_cfg)  # constructed **inside** the running loop
        try:
            # initialize pool/migrations if provided
            if hasattr(dbm, "init"):
                await dbm.init()

            stored = 0
            errors = []

            # If records is a dict with 'documents', call store_validator_output (preferred)
            if isinstance(recs, dict) and "documents" in recs:
                try:
                    vid = await dbm.store_validator_output(recs, original_filename=recs.get("original_filename"))
                    return {"status": "success", "stored": 1 if vid else 0, "validation_id": vid, "errors": []}
                except Exception as e:
                    return {"status": "failed", "stored": 0, "errors": [str(e)]}

            # If list of records, attempt upsert or store per record
            if isinstance(recs, list):
                for rec in recs:
                    try:
                        # prefer upsert_document_master if available
                        if hasattr(dbm, "upsert_document_master"):
                            did = rec.get("document_id") or rec.get("id") or None
                            dtype = rec.get("document_type") or rec.get("type") or "unknown"
                            original_filename = rec.get("original_filename") or rec.get("source")
                            upload_ts = rec.get("upload_timestamp")
                            processing_status = rec.get("processing_status", "processing")
                            validation_status = rec.get("validation_status")
                            final_status = rec.get("final_status")
                            # ensure we have an id
                            if not did:
                                # try db to generate or fall back to simple uuid
                                try:
                                    from uuid import uuid4
                                    did = "doc_" + uuid4().hex[:12]
                                except Exception:
                                    did = f"doc_{int(time.time())}"
                            await dbm.upsert_document_master(did, dtype, original_filename, upload_ts, processing_status, validation_status, final_status)
                            # optional audit insert if supported
                            if hasattr(dbm, "insert_processing_audit"):
                                try:
                                    input_data = rec.get("input_data", {})
                                    output_data = rec.get("output_data", {})
                                    proc_ms = float(rec.get("processing_time_ms", 0.0))
                                    success = bool(rec.get("success", True))
                                    await dbm.insert_processing_audit(did, rec.get("processing_stage", "agent_store"), input_data, output_data, proc_ms, success, rec.get("error_message"))
                                except Exception:
                                    pass
                            stored += 1
                        elif hasattr(dbm, "store_validator_output") and isinstance(rec, dict) and "validated_fields" in rec:
                            # convert single rec into validator-like wrapper
                            fake = {"validation_id": rec.get("validation_id") or f"val_{int(time.time())}", "documents": [rec]}
                            await dbm.store_validator_output(fake, original_filename=rec.get("original_filename"))
                            stored += 1
                        else:
                            raise RuntimeError("No supported insertion method on DatabaseManager for this record")
                    except Exception as e:
                        errors.append(str(e))
                return {"status": ("success" if stored > 0 and not errors else ("success" if stored>0 else "failed")), "stored": stored, "errors": errors}
            return {"status": "failed", "error": "unsupported_records_type"}
        finally:
            # close/cleanup
            try:
                if hasattr(dbm, "close"):
                    await dbm.close()
            except Exception:
                pass

    # Run the async worker
    try:
        result = asyncio.run(_store_async(db_cfg, normalized_records))
        if isinstance(result, dict):
            # add timing
            elapsed = time.perf_counter() - start
            result.setdefault("timing", {})["elapsed_s"] = round(elapsed, 3)
        return result
    except Exception as ex:
        logger.exception("store_documents failed when running async worker: %s", ex)
        return {"status": "failed", "error": str(ex)}


# --- Replace the existing fetch_document function with this ---
@register_function(name="fetch_document", description="Fetch a stored document by document_id.")
def fetch_document(document_id: str, agent_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Uses DatabaseManager.get_document_for_review(document_id) (async) to return
    master + per-type rows + processing_audit + edit_history.

    Constructs DatabaseManager inside an asyncio event loop via asyncio.run to avoid
    'no current event loop' errors.
    """
    import asyncio

    agent_cfg = DatabaseAgentConfig()
    if agent_config and isinstance(agent_config, dict):
        for k, v in agent_config.items():
            if hasattr(agent_cfg, k):
                try:
                    setattr(agent_cfg, k, v)
                except Exception:
                    logger.debug("Ignoring agent_config override %s=%s", k, v)

    async def _fetch_async(db_cfg, doc_id: str):
        dbm = DatabaseManager(db_cfg)
        # initialize (pool/connect)
        if hasattr(dbm, "init"):
            await dbm.init()
        # preferred async getter
        if hasattr(dbm, "get_document_for_review"):
            res = await dbm.get_document_for_review(doc_id)
            # close pool if supported
            if hasattr(dbm, "close"):
                try:
                    await dbm.close()
                except Exception:
                    pass
            return res
        # fallback: try other async fetch methods
        if hasattr(dbm, "get_document_by_id"):
            res = await dbm.get_document_by_id(doc_id)
            if hasattr(dbm, "close"):
                try:
                    await dbm.close()
                except Exception:
                    pass
            return res
        if hasattr(dbm, "fetch_record"):
            res = await dbm.fetch_record(doc_id)
            if hasattr(dbm, "close"):
                try:
                    await dbm.close()
                except Exception:
                    pass
            return res
        # close before returning
        if hasattr(dbm, "close"):
            try:
                await dbm.close()
            except Exception:
                pass
        return None

    try:
        db_cfg = _build_db_config(DatabaseAgentConfig(), None)
        doc = asyncio.run(_fetch_async(db_cfg, document_id))
        if doc is None:
            return {"status": "failed", "error": "fetch_method_unavailable_or_no_record"}
        return {"status": "success", "document": doc}
    except Exception as ex:
        logger.exception("fetch_document failed: %s", ex)
        return {"status": "failed", "error": str(ex)}

@register_function(name="query_documents", description="Query documents by simple filters. Query is dict of column->value equality.")
def query_documents(query: Optional[Dict[str, Any]] = None, agent_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Safe, parameterized query implementation:
      - If query contains 'document_id' -> returns get_document_for_review(document_id)
      - Otherwise allows equality filters only on a small whitelist of columns
        (document_type, validation_status, final_status, upload_timestamp).
    Runs DB work inside asyncio.run to construct DatabaseManager inside a running event loop.
    """
    if not query:
        return {"status": "failed", "error": "empty_query_not_supported"}

    # If user asks for a document_id, reuse fetch_document (which already uses asyncio.run)
    doc_id = query.get("document_id") or query.get("id")
    if doc_id:
        return fetch_document(doc_id, agent_config=agent_config)

    # Allowed columns for safe querying
    allowed_cols = {"document_type", "validation_status", "final_status", "upload_timestamp"}

    # Build predicates from query but only for allowed columns
    predicates = []
    params = []
    for k, v in (query.items() if isinstance(query, dict) else []):
        if k in allowed_cols:
            predicates.append(f"{k} = ?")
            params.append(v)

    if not predicates:
        return {"status": "failed", "error": "no_allowed_query_keys. Supported keys: " + ", ".join(sorted(allowed_cols))}

    import asyncio

    agent_cfg = DatabaseAgentConfig()
    if agent_config and isinstance(agent_config, dict):
        for k, v in agent_config.items():
            if hasattr(agent_cfg, k):
                try:
                    setattr(agent_cfg, k, v)
                except Exception:
                    logger.debug("Ignoring agent_config override %s=%s", k, v)

    async def _query_async(db_cfg: DBConfig, where_clause: str, params_tuple: Tuple[Any, ...]):
        dbm = DatabaseManager(db_cfg)
        # init pool and migrations
        if hasattr(dbm, "init"):
            await dbm.init()
        # Acquire a raw connection from the pool and run a parameterized SELECT
        conn = await dbm._pool.acquire()
        try:
            sql = f"SELECT * FROM documents WHERE {where_clause}"
            cur = await conn.execute(sql, params_tuple)
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            result = [dict(zip(cols, r)) for r in rows]
            return result
        finally:
            # ensure connection is returned to pool and pool closed by caller
            await dbm._pool.release(conn)
            if hasattr(dbm, "close"):
                try:
                    await dbm.close()
                except Exception:
                    pass

    try:
        db_cfg = _build_db_config(DatabaseAgentConfig(), None)
        where_clause = " AND ".join(predicates)
        rows = asyncio.run(_query_async(db_cfg, where_clause, tuple(params)))
        return {"status": "success", "rows": rows, "count": len(rows)}
    except Exception as ex:
        logger.exception("query_documents failed: %s", ex)
        return {"status": "failed", "error": str(ex)}


@register_function(name="db_health_check", description="Check DB file and DatabaseManager availability.")
def db_health_check(agent_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Verifies DatabaseManager instantiation, connection pool creation, and path health.
    Returns {"ok": bool, "messages": [str, ...]}.
    """
    import asyncio

    msgs: List[str] = []
    ok = True
    agent_cfg = DatabaseAgentConfig()
    if agent_config and isinstance(agent_config, dict):
        for k, v in agent_config.items():
            if hasattr(agent_cfg, k):
                try:
                    setattr(agent_cfg, k, v)
                except Exception:
                    logger.debug("Ignoring agent_config override %s=%s", k, v)

    async def _health_async(db_cfg):
        dbm = DatabaseManager(db_cfg)
        try:
            if hasattr(dbm, "init"):
                await dbm.init()
                msgs.append("DatabaseManager_instantiated_and_initialized")
            else:
                msgs.append("DatabaseManager_instantiated (no init method)")
            # Test simple pool acquire/release
            if hasattr(dbm, "_pool"):
                conn = await dbm._pool.acquire()
                msgs.append("pool_acquire_success")
                await dbm._pool.release(conn)
            if hasattr(dbm, "close"):
                await dbm.close()
            return True
        except Exception as e:
            msgs.append(f"DatabaseManager_error: {e}")
            return False

    try:
        db_cfg = _build_db_config(agent_cfg, None)
        ok = asyncio.run(_health_async(db_cfg))
        if hasattr(db_cfg, "db_path"):
            msgs.append(f"db_path: {getattr(db_cfg, 'db_path')}")
    except Exception as ex:
        msgs.append(f"health_check_exception: {ex}")
        ok = False

    return {"ok": ok, "messages": msgs}

# ---------------------------
# Agent factory: attach functions from local registry
# ---------------------------
def create_database_agent(agent_name: str = "DatabaseAgent", agent_cfg: Optional[DatabaseAgentConfig] = None) -> AssistantAgent:
    agent_cfg = agent_cfg or DatabaseAgentConfig()
    system_prompt = (
        "DatabaseAgent: provides deterministic storage and retrieval tools for validated document records. "
        "Exposes functions: store_documents, fetch_document, query_documents, db_health_check. "
        "Tool-only agent: no LLM reasoning."
    )

    agent = AssistantAgent(name=agent_name, role="Stores documents in DB", system_prompt=system_prompt)

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

    logger.info("DatabaseAgent created and functions registered.")
    return agent


# ---------------------------
# Local basic test harness (calls the tools directly)
# ---------------------------
def basic_test() -> None:
    """
    Basic test flow:
      - constructs a sample record
      - runs db_health_check()
      - calls store_documents(...) and then fetch_document and query_documents
    """
    print("=== DatabaseAgent basic_test ===")
    agent_cfg = DatabaseAgentConfig()
    print("Running db_health_check() ...")
    print(json.dumps(db_health_check(), indent=2))

    # Sample record
    sample = {
        "document_id": f"doc_test_{int(time.time())}",
        "document_type": "pan",
        "fields": {"pan_number": "ABCDE1234F", "name": "TEST USER"},
        "source": "unit_test"
    }

    print("Storing sample record ...")
    res = store_documents([sample])
    print("store_documents result:", json.dumps(res, indent=2))

    if res.get("status") == "success" and res.get("stored", 0) > 0:
        did = sample["document_id"]
        print("Fetching stored document ...")
        f = fetch_document(did)
        print("fetch_document:", json.dumps(f, indent=2))
        print("Querying by document_type ...")
        q = query_documents({"document_type": "pan"})
        print("query_documents:", json.dumps({"status": q.get("status"), "rows_count": len(q.get("rows", []) if q.get("rows") else [])}, indent=2))
    else:
        print("Store failed; skipping fetch/query.")


# ---------------------------
# CLI entrypoint
# ---------------------------
if __name__ == "__main__":
    # Run the basic test when module executed
    basic_test()
