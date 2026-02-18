
from __future__ import annotations
import os
import json
import time
import logging
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
import concurrent.futures
import asyncio

# IO for saving images
from PIL import Image
import numpy as np

# Imports from your modules
try:
    from document_preprocessor import process_pdf_file, PreprocessorConfig, DocumentPreprocessorError  # type: ignore
except Exception as e:
    raise ImportError("Failed to import document_preprocessor. Make sure document_preprocessor.py is on PYTHONPATH.") from e

try:
    from document_classifier import DocumentClassifier, ClassifierConfig, DocumentClassifierError, ClassificationConfidenceError  # type: ignore
except Exception as e:
    raise ImportError("Failed to import document_classifier. Ensure document_classifier.py is present and importable.") from e

# Vision extractor import (integration)
try:
    from vision_extractor import VisionExtractor, ExtractorConfig, VisionExtractorError  # type: ignore
except Exception as e:
    # Allow orchestrator to run even if vision_extractor is not present; surface clear message
    VisionExtractor = None  # type: ignore
    ExtractorConfig = None  # type: ignore
    VisionExtractorError = Exception  # type: ignore

try:
    from data_validator import validate_batch, ValidatorConfig  # type: ignore
except Exception as e:
    raise ImportError("Failed to import data_validator. Ensure data_validator.py is present and importable.") from e

try:
    from db_engine import DatabaseManager, DBConfig  # type: ignore
except Exception as e:  # type: ignore
    raise ImportError("Failed to import database_manager. Ensure database_manager.py is present and importable.") from e

logger = logging.getLogger("OrchestratorWrapper")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[OrchestratorWrapper] %(asctime)s %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# -------------------------
# Helpers
# -------------------------
def _ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_json_serializable(obj):
    """
    Convert common non-serializables (numpy types, numpy arrays) to serializable types.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, dict, str, int, float, bool)) or obj is None:
        return obj
    try:
        return str(obj)
    except Exception:
        return None


# -------------------------
# Sanitizer: remove heavy image arrays before JSON dump
# -------------------------
def _sanitize_for_json(obj, keys_to_remove=("image_array",), max_list_length=10000):
    """
    Recursively remove large image arrays and convert numpy types to standard Python types
    - keys_to_remove: tuple of key names to remove entirely
    - returns a deep-copied sanitized structure
    """
    # Guard for None or primitive types
    import numpy as _np

    if obj is None:
        return None
    if isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, _np.generic):
        # convert numpy scalar to native
        return obj.item()
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and k in keys_to_remove:
                # skip this key entirely
                continue
            # sanitize nested
            out[k] = _sanitize_for_json(v, keys_to_remove=keys_to_remove, max_list_length=max_list_length)
        return out
    if isinstance(obj, (list, tuple, set)):
        # If list is extremely large, truncate for safety (but preserve small lists)
        if isinstance(obj, (list, tuple)) and len(obj) > max_list_length:
            # return a short marker instead of full list
            return f"<TRUNCATED_LIST length={len(obj)}>"
        return [_sanitize_for_json(x, keys_to_remove=keys_to_remove, max_list_length=max_list_length) for x in obj]
    # Numpy arrays: remove (these are large)
    if hasattr(obj, "dtype") and hasattr(obj, "shape"):
        # likely a numpy array — do not include
        return "<IMAGE_ARRAY_REMOVED>"
    # fallback - try to convert to string
    try:
        return str(obj)
    except Exception:
        return None


# -------------------------
# Orchestration function
# -------------------------
def orchestrate_pdf_processing(
    pdf_path: str,
    preprocessor_cfg: Optional[PreprocessorConfig] = None,
    classifier_cfg: Optional[ClassifierConfig] = None,
    extractor_cfg: Optional[ExtractorConfig] = None,
    validator_cfg: Optional[ValidatorConfig] = None,
    db_cfg: Optional[DBConfig] = None,
    model_path_override: Optional[str] = None,
    save_json_path: Optional[str] = None,
    save_images_dir: Optional[str] = None,
    strict_confidence: bool = False,
    raise_on_preprocess_error: bool = True,
    run_async: bool = False,
    run_extraction: bool = True,
    run_validation: bool = True,
    run_db_storage: bool = True,
    enable_health_checks: bool = True
) -> Dict[str, Any]:
    """
    COMPLETE pipeline: PDF → Preprocess → Classify → Extract → Validate → Store
    
    Each stage consumes exact output of previous stage without transformation.
    """
    start_total = time.perf_counter()
    pdf_path = str(pdf_path)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # HEALTH CHECKS (Optional)
    if enable_health_checks:
        if run_extraction and VisionExtractor:
            extractor_config = extractor_cfg or ExtractorConfig()
            extractor = VisionExtractor(extractor_config)
            try:
                health_ok = asyncio.run(extractor.health_check())
                if not health_ok:
                    logger.warning("Ollama health check failed - extraction may fail")
            except Exception as e:
                logger.warning(f"Health check failed: {e}")

    # STAGE 1: PDF PREPROCESSING
    pre_start = time.perf_counter()
    try:
        preprocessed_pages = process_pdf_file(pdf_path, preprocessor_cfg)
    except DocumentPreprocessorError as e:
        logger.exception("Preprocessor failed")
        if raise_on_preprocess_error:
            raise
        else:
            return {"error": {"stage": "preprocessing", "message": str(e)}}
    pre_elapsed = time.perf_counter() - pre_start

    num_pages = len(preprocessed_pages)
    logger.info(f"✓ Preprocessed {num_pages} pages in {pre_elapsed:.3f}s")

    # Save preprocessed images if requested
    if save_images_dir:
        _save_preprocessed_images(preprocessed_pages, save_images_dir)

    # STAGE 2: DOCUMENT CLASSIFICATION
    cls_start = time.perf_counter()
    try:
        classifier_config = classifier_cfg or ClassifierConfig()
        if model_path_override:
            classifier_config.model_path = model_path_override
        
        classifier = DocumentClassifier(classifier_config)
        classification_result = classifier.classify_and_aggregate(preprocessed_pages)
        
        # Handle classifier rejection
        if classification_result.get("status") == "rejected":
            rejection_data = _handle_classifier_rejection(classification_result, pdf_path, save_json_path)
            return rejection_data
            
    except DocumentClassifierError as e:
        logger.exception("Classification failed")
        raise
    cls_elapsed = time.perf_counter() - cls_start

    logger.info(f"✓ Classified {len(classification_result.get('pages', []))} pages into {len(classification_result.get('documents', []))} documents in {cls_elapsed:.3f}s")

    # Strict confidence enforcement
    if strict_confidence:
        _enforce_confidence_threshold(classification_result, classifier_config.confidence_threshold)

    # INITIAL RESULT STRUCTURE
    result = {
        "pipeline_stages": ["preprocessing", "classification"],
        "classification": classification_result,
        "preprocessor_info": {"num_pages": num_pages, "elapsed_s": round(pre_elapsed, 4)},
        "runtime": {"total_elapsed_s": round(time.perf_counter() - start_total, 4)}
    }

    # STAGE 3: VISION EXTRACTION
    extraction_result = None
    if run_extraction and VisionExtractor is not None:
        try:
            extract_start = time.perf_counter()
            extractor_config = extractor_cfg or ExtractorConfig()
            extractor = VisionExtractor(extractor_config)
            
            # DIRECT USAGE: classifier output → extractor input
            extraction_result = extractor.extract_documents(classification_result)
            extract_elapsed = time.perf_counter() - extract_start
            
            result["extraction"] = extraction_result
            result["pipeline_stages"].append("extraction")
            result["runtime"]["extraction_elapsed_s"] = round(extract_elapsed, 4)
            
            logger.info(f"✓ Extracted data from {len(extraction_result.get('documents', []))} documents in {extract_elapsed:.3f}s")
            
        except VisionExtractorError as e:
            logger.exception("Vision extraction failed")
            result["extraction_error"] = {
                "stage": "extraction", 
                "type": e.__class__.__name__, 
                "message": str(e)
            }

    # STAGE 4: DATA VALIDATION
    validation_result = None
    if run_validation and extraction_result and extraction_result.get("success"):
        try:
            validate_start = time.perf_counter()
            validator_config = validator_cfg or ValidatorConfig()
            
            # DIRECT USAGE: extractor output → validator input
            validation_result = validate_batch(extraction_result, validator_config)
            validate_elapsed = time.perf_counter() - validate_start
            
            result["validation"] = validation_result
            result["pipeline_stages"].append("validation")
            result["runtime"]["validation_elapsed_s"] = round(validate_elapsed, 4)
            
            logger.info(f"✓ Validated {len(validation_result.get('documents', []))} documents in {validate_elapsed:.3f}s")
            
        except Exception as e:
            logger.exception("Validation failed")
            result["validation_error"] = {
                "stage": "validation",
                "type": e.__class__.__name__,
                "message": str(e)
            }

    # STAGE 5: DATABASE STORAGE
    storage_result = None
    if run_db_storage and validation_result and db_cfg:
        try:
            storage_start = time.perf_counter()
            db_manager = DatabaseManager(db_cfg)
            
            # Initialize database connection
            asyncio.run(db_manager.init())
            
            # DIRECT USAGE: validator output → database input
            validation_id = asyncio.run(
                db_manager.store_validator_output(
                    validation_result, 
                    original_filename=os.path.basename(pdf_path)
                )
            )
            
            storage_elapsed = time.perf_counter() - storage_start
            storage_result = {"validation_id": validation_id, "success": True}
            result["database_storage"] = storage_result
            result["pipeline_stages"].append("database_storage")
            result["runtime"]["storage_elapsed_s"] = round(storage_elapsed, 4)
            
            logger.info(f"✓ Stored results in database with validation_id: {validation_id}")
            
            # Close database connection
            asyncio.run(db_manager.close())
            
        except Exception as e:
            logger.exception("Database storage failed")
            result["storage_error"] = {
                "stage": "database_storage",
                "type": e.__class__.__name__,
                "message": str(e)
            }

    # SAVE RESULTS TO JSON
    if save_json_path:
        _save_pipeline_results(result, pdf_path, save_json_path)

    # FINAL METRICS
    result["runtime"]["total_elapsed_s"] = round(time.perf_counter() - start_total, 4)
    result["pipeline_status"] = "complete"
    
    logger.info(f"✓ Pipeline completed in {result['runtime']['total_elapsed_s']:.3f}s")
    
    return result

def _save_preprocessed_images(preprocessed_pages: List[Dict], save_dir: str):
    """Save preprocessed images to directory."""
    try:
        savedir = _ensure_dir(save_dir)
        for i, page in enumerate(preprocessed_pages):
            md = page.get("metadata", {})
            pgno = md.get("page_number", i + 1)
            img_arr = page.get("image_array")
            if img_arr is not None:
                try:
                    # Ensure the array is properly formatted for PIL
                    if isinstance(img_arr, np.ndarray):
                        img = Image.fromarray(np.uint8(img_arr))
                    else:
                        # Convert to numpy array first if needed
                        img_arr_np = np.array(img_arr, dtype=np.uint8)
                        img = Image.fromarray(img_arr_np)
                    
                    outpath = savedir / f"page_{pgno:03d}.png"
                    img.save(outpath)
                    logger.debug(f"Saved preprocessed image: {outpath}")
                except Exception as e:
                    logger.warning(f"Failed saving image for page {pgno}: {e}")
    except Exception as e:
        logger.error(f"Failed to save preprocessed images to {save_dir}: {e}")


def _handle_classifier_rejection(classification_result: Dict, pdf_path: str, save_json_path: Optional[str]):
    """Handle classifier rejection with proper error reporting."""
    reason = classification_result.get("reason", "rejected")
    issues = classification_result.get("issues", [])
    logger.error(f"Classification rejected: {reason} - {issues}")

    if save_json_path:
        try:
            outp = Path(save_json_path)
            if outp.is_dir():
                pdf_base = Path(pdf_path).stem
                outp = outp / f"{pdf_base}.rejected.json"
            outp.parent.mkdir(parents=True, exist_ok=True)
            sanitized = _sanitize_for_json(classification_result)
            with open(outp, "w", encoding="utf-8") as f:
                json.dump(sanitized, f, default=_to_json_serializable, indent=2)
        except Exception:
            logger.exception("Failed to save rejection JSON")

    return {
        "pipeline_status": "rejected",
        "rejection_reason": reason,
        "rejection_issues": issues,
        "classification_result": classification_result
    }

def _enforce_confidence_threshold(classification_result: Dict, threshold: float):
    """Enforce strict confidence threshold."""
    low_conf_pages = [
        p for p in classification_result.get("pages", []) 
        if p.get("confidence", 0.0) < threshold
    ]
    if low_conf_pages:
        page_nos = [p.get("page_number") for p in low_conf_pages]
        raise ClassificationConfidenceError(
            f"{len(low_conf_pages)} pages below confidence threshold {threshold:.3f} - pages: {page_nos}"
        )


def _save_pipeline_results(result: Dict, pdf_path: str, save_path: str):
    """Save complete pipeline results to JSON (sanitized of image arrays)."""
    try:
        outp = Path(save_path)
        if outp.is_dir():
            pdf_base = Path(pdf_path).stem
            outp = outp / f"{pdf_base}.pipeline_results.json"
        outp.parent.mkdir(parents=True, exist_ok=True)

        # Sanitize result to remove image arrays & large binary data
        sanitized = _sanitize_for_json(result)

        with open(outp, "w", encoding="utf-8") as f:
            json.dump(sanitized, f, default=_to_json_serializable, indent=2)
        logger.info(f"✓ Saved pipeline results to {outp}")
    except Exception:
        logger.exception("Failed to save pipeline results")


# -------------------------
# Async wrapper
# -------------------------
def orchestrate_pdf_classification_async(*args, **kwargs) -> concurrent.futures.Future:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(orchestrate_pdf_classification, *args, **kwargs)
    return future


# -------------------------
# CLI usage example - UPDATED FOR COMPLETE PIPELINE
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog="document_pipeline", 
        description="Complete document processing pipeline: PDF → Preprocess → Classify → Extract → Validate → Store"
    )
    
    # Required arguments
    parser.add_argument("pdf_path", help="Path to PDF file to process")
    
    # Output options
    parser.add_argument("--save-json", help="Path to save pipeline JSON output (file or directory)")
    parser.add_argument("--save-images", help="Directory to save preprocessed page images (PNG)")
    
    # Configuration overrides
    parser.add_argument("--model", help="Override model path for classifier")
    parser.add_argument("--ollama-endpoint", help="Ollama endpoint for extraction (default: http://localhost:11434)")
    parser.add_argument("--ollama-model", help="Ollama model name for extraction (default: qwen2.5vl:3b)")
    
    # Pipeline stage control
    parser.add_argument("--no-extract", help="Skip vision extraction stage", action="store_true")
    parser.add_argument("--no-validate", help="Skip validation stage", action="store_true") 
    parser.add_argument("--no-store", help="Skip database storage stage", action="store_true")
    parser.add_argument("--stages", help="Specify which stages to run (comma-separated: preprocess,classify,extract,validate,store)")
    
    # Processing options
    parser.add_argument("--strict", help="Enable strict confidence enforcement (raises on low-confidence pages)", action="store_true")
    parser.add_argument("--max-pages", type=int, help="Maximum number of pages to process from PDF", default=10)
    parser.add_argument("--timeout", type=int, help="Processing timeout in seconds", default=420)
    
    # Database options
    parser.add_argument("--db-path", help="Database file path for storage")
    parser.add_argument("--no-health-check", help="Skip Ollama health check", action="store_true")
    
    args = parser.parse_args()

    try:
        # Build configurations based on CLI arguments
        preprocessor_cfg = PreprocessorConfig(
            max_pages=args.max_pages,
            timeout_seconds=args.timeout
        )
        
        classifier_cfg = ClassifierConfig()
        if args.model:
            classifier_cfg.model_path = args.model
        
        extractor_cfg = ExtractorConfig()
        if args.ollama_endpoint:
            extractor_cfg.ollama_endpoint = args.ollama_endpoint
        if args.ollama_model:
            extractor_cfg.model_name = args.ollama_model
        
        validator_cfg = ValidatorConfig()
        
        db_cfg = DBConfig()
        if args.db_path:
            db_cfg.database_url = args.db_path
        
        # Determine which stages to run
        run_extraction = not args.no_extract
        run_validation = not args.no_validate  
        run_db_storage = not args.no_store
        enable_health_checks = not args.no_health_check
        
        # If specific stages are specified, override the boolean flags
        if args.stages:
            stages_list = [s.strip().lower() for s in args.stages.split(',')]
            run_extraction = 'extract' in stages_list
            run_validation = 'validate' in stages_list  
            run_db_storage = 'store' in stages_list
        
        print(f"Starting document pipeline for: {args.pdf_path}")
        print(f"Stages: Preprocess → Classify {'→ Extract' if run_extraction else ''} {'→ Validate' if run_validation else ''} {'→ Store' if run_db_storage else ''}")
        
        # Execute complete pipeline
        out = orchestrate_pdf_processing(
            pdf_path=args.pdf_path,
            preprocessor_cfg=preprocessor_cfg,
            classifier_cfg=classifier_cfg,
            extractor_cfg=extractor_cfg, 
            validator_cfg=validator_cfg,
            db_cfg=db_cfg,
            model_path_override=args.model,
            save_json_path=args.save_json,
            save_images_dir=args.save_images,
            strict_confidence=args.strict,
            run_extraction=run_extraction,
            run_validation=run_validation,
            run_db_storage=run_db_storage,
            enable_health_checks=enable_health_checks
        )
        
        # Display results
        if out:
            pipeline_status = out.get("pipeline_status", "unknown")
            print(f"\n=== PIPELINE COMPLETED: {pipeline_status.upper()} ===")
            
            # Show basic metrics
            runtime = out.get("runtime", {})
            total_time = runtime.get("total_elapsed_s", 0)
            print(f"Total processing time: {total_time:.2f}s")
            
            # Show stage-specific results
            if "classification" in out:
                cls_result = out["classification"]
                pages = len(cls_result.get("pages", []))
                docs = len(cls_result.get("documents", []))
                print(f"Classification: {pages} pages → {docs} documents")
            
            if "extraction" in out:
                ext_result = out["extraction"]
                success = ext_result.get("success", False)
                docs_processed = len(ext_result.get("documents", []))
                print(f"Extraction: {'SUCCESS' if success else 'FAILED'} - {docs_processed} documents processed")
            
            if "validation" in out:
                val_result = out["validation"]
                overall_status = val_result.get("overall_status", "UNKNOWN")
                val_metrics = val_result.get("summary_metrics", {})
                approved = val_metrics.get("approved_count", 0)
                review = val_metrics.get("review_required_count", 0)
                rejected = val_metrics.get("rejected_count", 0)
                print(f"Validation: {overall_status} - Approved: {approved}, Review: {review}, Rejected: {rejected}")
            
            if "database_storage" in out:
                storage_result = out["database_storage"]
                if storage_result.get("success"):
                    validation_id = storage_result.get("validation_id")
                    print(f"Storage: SUCCESS - Validation ID: {validation_id}")
                else:
                    print("Storage: FAILED")
            
            # Show errors if any
            errors = []
            for key in out:
                if key.endswith('_error'):
                    errors.append(f"{key}: {out[key].get('message')}")
            
            if errors:
                print(f"\nErrors encountered: {len(errors)}")
                for err in errors:
                    print(f"  - {err}")
                    
            # Show output file locations
            if args.save_json:
                print(f"\nOutput JSON saved to: {args.save_json}")
            if args.save_images:
                print(f"Preprocessed images saved to: {args.save_images}")
                
        else:
            print("Pipeline completed but no output returned - check logs for details.")
            
    except DocumentPreprocessorError as e:
        print(f"❌ PREPROCESSING FAILED: {e}")
        exit(1)
    except ClassificationConfidenceError as e:
        print(f"❌ CONFIDENCE THRESHOLD FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ PROCESSING FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
