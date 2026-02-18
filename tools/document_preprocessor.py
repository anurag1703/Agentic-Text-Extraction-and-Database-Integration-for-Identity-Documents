from __future__ import annotations

import os
import io
import math
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import concurrent.futures

from pdf2image import convert_from_path
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2


class DocumentPreprocessorError(Exception):
    """Base class for preprocessor exceptions."""


class FileTooLargeError(DocumentPreprocessorError):
    """Raised when file exceeds max allowed size."""


class InvalidPDFError(DocumentPreprocessorError):
    """Raised when PDF is invalid or corrupted."""


class ProcessingTimeoutError(DocumentPreprocessorError):
    """Raised when processing exceeds timeout limit."""


class QualityThresholdError(DocumentPreprocessorError):
    """Raised when image quality is below acceptable threshold."""


class PDFReadError(DocumentPreprocessorError):
    """Raised for generic PDF read errors."""


@dataclass
class PreprocessorConfig:
    # File & PDF handling
    max_file_size_bytes: int = 10 * 1024 * 1024
    max_pages: int = 5
    dpi: int = 280
    poppler_path: Optional[str] = r"PATH TO POPPLER"

    # Image quality thresholds
    min_dimension_px: int = 600
    max_dimension_px: int = 1024
    blur_threshold: float = 120.0  # Laplacian variance
    brightness_min: float = 50.0  # mean brightness
    brightness_max: float = 205.0  # avoid overexposed pages
    contrast_min: float = 30.0  # simple measure via std dev
    quality_score_min: float = 0.25  # global threshold to decide QualityThresholdError

    # File output / preview options (new)
    save_images: bool = False              # if True, write per-page PNG/JPEG to save_dir
    save_dir: Optional[str] = None         # target dir for saved page images; if None will default to outputs/preprocessed
    return_preview_bytes: bool = False     # if True, attach small JPEG bytes in 'preview_bytes' on each page dict
    preview_format: str = "jpg"            # "jpg" or "png"
    preview_quality: int = 90              # JPEG quality

    # Enhancement toggles / params
    enable_orientation: bool = True
    enable_resize: bool = True
    enable_sharpen: bool = True
    enable_clahe: bool = True
    enable_denoise: bool = True

    # Sharpening params (Unsharp mask)
    unsharp_radius: float = 0.5
    unsharp_percent: int = 150
    unsharp_threshold: int = 3

    # Denoising params
    denoise_h: int = 10  # parameter for cv2.fastNlMeansDenoisingColored

    # Timeouts
    timeout_seconds: int = 180

    # Memory/performance
    page_batch_size: int = 1  # process page-by-page to reduce memory footprint

    # Logging options
    verbose: bool = True


def _log(cfg: PreprocessorConfig, *args, **kwargs):
    if cfg.verbose:
        print("[DocumentPreprocessor]", *args, **kwargs)


def _check_magic_pdf(path: str) -> bool:
    """Check magic bytes for PDF header '%PDF-' within first 8 bytes."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        return head.startswith(b"%PDF-")
    except Exception:
        return False


def _pil_to_numpy_rgb(img: Image.Image) -> np.ndarray:
    """Convert PIL.Image (RGB) to numpy array H,W,3 uint8 (RGB)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img)
    # Ensure uint8
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _numpy_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    """Convert numpy array H,W,3 (RGB) to PIL Image."""
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return Image.fromarray(arr)


def _measure_blur_cv2(np_rgb: np.ndarray) -> float:
    """Compute Laplacian variance as blur measure (higher is sharper)."""
    gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var())
    return var


def _measure_brightness_contrast(np_rgb: np.ndarray) -> Tuple[float, float]:
    """Return (brightness_mean, contrast_stddev) from grayscale."""
    gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    return mean, std

def remove_padding_from_pil(pil_img: Image.Image,
                            bg_color_sample_size: int = 10,
                            tolerance: int = 12,
                            min_crop_ratio: float = 0.01,
                            margin_px: int = 8) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Remove uniform/near-uniform padding (border) from a PIL RGB image.

    Parameters
    ----------
    pil_img : PIL.Image.Image
        Input image (will be converted to RGB if needed).
    bg_color_sample_size : int
        Pixel region size taken from each corner to estimate background color.
    tolerance : int
        Per-channel tolerance (0-255). Pixels within tolerance of bg color are considered background.
    min_crop_ratio : float
        Minimum fraction of image area that must be cropped to actually perform cropping.
        Prevents accidental cropping when no border present.
    margin_px : int
        Keep `margin_px` pixels around the detected content (safety margin).

    Returns
    -------
    cropped_img : PIL.Image.Image
        Cropped image (or original if no cropping applied).
    info : dict
        { 'cropped': bool, 'crop_bbox': (left, upper, right, lower), 'bg_color_est': (r,g,b) }
    """
    # ensure RGB
    if pil_img.mode != "RGB":
        pil = pil_img.convert("RGB")
    else:
        pil = pil_img

    arr = np.asarray(pil)  # H,W,3 uint8
    h, w = arr.shape[:2]

    # sample small corner patches to estimate background color (median makes it robust)
    s = max(1, min(bg_color_sample_size, min(h//10, w//10)))
    corners = [
        arr[0:s, 0:s],                # top-left
        arr[0:s, w-s:w],              # top-right
        arr[h-s:h, 0:s],              # bottom-left
        arr[h-s:h, w-s:w],            # bottom-right
    ]
    # compute median color across corners
    samples = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
    bg_color = tuple(np.median(samples, axis=0).astype(int).tolist())  # (r,g,b)

    # compute per-pixel absolute distance from bg_color
    diff = np.abs(arr.astype(int) - np.array(bg_color, dtype=int).reshape((1,1,3)))
    # any channel above tolerance -> considered foreground
    fg_mask = np.any(diff > tolerance, axis=2).astype(np.uint8) * 255  # 0/255

    # morphological ops to remove tiny specks and fill holes
    kernel_size = max(3, min(31, int(min(h, w) * 0.01)))  # adapt kernel to image size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # find bounding box of foreground
    ys, xs = np.where(fg_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        # no foreground found -> return original
        return pil_img, {"cropped": False, "crop_bbox": (0,0,w,h), "bg_color_est": bg_color}

    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())

    # add safety margin
    left = max(0, left - margin_px)
    top = max(0, top - margin_px)
    right = min(w-1, right + margin_px)
    bottom = min(h-1, bottom + margin_px)

    # compute crop area and check it is meaningful (avoid cropping tiny amounts)
    crop_w = right - left + 1
    crop_h = bottom - top + 1
    crop_area = crop_w * crop_h
    orig_area = w * h

    if (orig_area - crop_area) / orig_area < min_crop_ratio:
        # not enough padding to crop (less than min_crop_ratio of area), skip cropping
        return pil_img, {"cropped": False, "crop_bbox": (0,0,w,h), "bg_color_est": bg_color}

    cropped = pil.crop((left, top, right + 1, bottom + 1))
    info = {"cropped": True, "crop_bbox": (left, top, right, bottom), "bg_color_est": bg_color}
    return cropped, info


def _rotate_if_vertical(pil_img: Image.Image) -> Tuple[Image.Image, bool]:
    """
    Check orientation: if height > width (vertical), rotate 90 degrees CCW
    to make it landscape. Returns (image, rotated_flag).
    """
    w, h = pil_img.size
    if h > w:
        rotated = pil_img.rotate(90, expand=True)
        return rotated, True
    return pil_img, False


def _resize_preserve_aspect(pil_img: Image.Image, min_px: int, max_px: int) -> Tuple[Image.Image, Tuple[int, int], bool]:
    """
    Resize such that the longer side is within [min_px, max_px].
    Maintains aspect ratio.
    Returns (resized_image, new_dims, resized_flag).
    """
    w, h = pil_img.size
    longer = max(w, h)
    if longer < min_px or longer > max_px:
        if longer < min_px:
            scale = min_px / longer
        else:
            scale = max_px / longer
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        return resized_img, (new_w, new_h), True
    return pil_img, (w, h), False


def _apply_unsharp_mask(pil_img: Image.Image, radius: float, percent: int, threshold: int) -> Image.Image:
    """
    Implement unsharp mask using PIL's filter as approximation.
    """
    try:
        # Use PIL's UnsharpMask if available
        return pil_img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
    except Exception:
        # Fallback: simple sharpening via filter
        return pil_img.filter(ImageFilter.SHARPEN)


def _apply_clahe_on_rgb(np_rgb: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE on the L channel of LAB colorspace for contrast-limited histogram equalization.
    Returns RGB uint8 array.
    """
    lab = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    rgb2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return rgb2


def _apply_denoise(np_rgb: np.ndarray, h: int) -> np.ndarray:
    """
    Use OpenCV's fastNlMeansDenoisingColored - works on BGR; convert appropriately.
    """
    bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)
    rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    return rgb


def _normalize_dimensions(np_rgb: np.ndarray, target_dims: Tuple[int, int]) -> np.ndarray:
    """Resize numpy array to target dims (width, height) using cv2.INTER_AREA for downscale."""
    h, w = np_rgb.shape[:2]
    target_w, target_h = target_dims
    if (w, h) == (target_w, target_h):
        return np_rgb
    return cv2.resize(np_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _compute_quality_score(blur_var: float, brightness: float, contrast: float,
                           cfg: PreprocessorConfig) -> float:
    """
    Compute a normalized quality score in [0,1].
    Heuristics:
    - blur contributes positively with sigmoid scaling (higher is better)
    - brightness penalizes very dark or very bright images
    - contrast additively helps
    """
    # Blur score (normalize around blur threshold)
    b_score = 1.0 - math.exp(-max(0.0, blur_var) / (cfg.blur_threshold * 2.0))
    # Brightness penalty: ideal range between brightness_min and brightness_max
    if brightness < cfg.brightness_min:
        bright_score = brightness / cfg.brightness_min  # 0..1
    elif brightness > cfg.brightness_max:
        # penalize overexposure
        bright_score = max(0.0, 1.0 - (brightness - cfg.brightness_max) / (255 - cfg.brightness_max))
    else:
        bright_score = 1.0
    # Contrast score normalized roughly by expected std-> consider 0..100 mapping
    contrast_score = min(1.0, contrast / 80.0)

    # Weighted aggregation
    score = (0.5 * b_score) + (0.3 * bright_score) + (0.2 * contrast_score)
    # clamp
    score = min(1.0, max(0.0, score))
    return score


class DocumentPreprocessor:
    def __init__(self, cfg: Optional[PreprocessorConfig] = None):
        self.cfg = cfg or PreprocessorConfig()
        # Validate config sensible values
        if self.cfg.max_pages <= 0:
            raise ValueError("max_pages must be positive")
        if self.cfg.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    def _validate_input_pdf(self, path: str) -> None:
        # Accept only PDF by extension and magic bytes
        if not isinstance(path, str):
            raise InvalidPDFError("Path must be a string")
        if not os.path.exists(path):
            raise InvalidPDFError(f"File does not exist: {path}")
        if not os.path.isfile(path):
            raise InvalidPDFError(f"Not a file: {path}")
        # extension check
        ext = os.path.splitext(path)[1].lower()
        if ext != ".pdf":
            raise InvalidPDFError("Only PDF files are accepted")
        # size check
        size = os.path.getsize(path)
        if size > self.cfg.max_file_size_bytes:
            raise FileTooLargeError(f"File too large: {size} bytes > {self.cfg.max_file_size_bytes}")
        # magic bytes
        if not _check_magic_pdf(path):
            raise InvalidPDFError("File does not have valid PDF magic bytes (corrupted or not PDF)")

    def _convert_pdf_pages(self, path: str, max_pages: int) -> List[Image.Image]:
        """
        Convert PDF to PIL images using pdf2image.convert_from_path.
        Process pages up to max_pages and return PIL.Image list.
        This method streams pages by using first_page/last_page if needed.

        Robustness additions:
         - Handles PIL.Image.DecompressionBombError by retrying with lower DPI.
         - As a last resort temporarily increases Image.MAX_IMAGE_PIXELS for a single retry
           (logged with a warning) before failing.
        """
        try:
            images: List[Image.Image] = []
            for page_no in range(1, max_pages + 1):
                try:
                    pil_pages = convert_from_path(
                        path,
                        dpi=self.cfg.dpi,
                        first_page=page_no,
                        last_page=page_no,
                        fmt="ppm",
                        poppler_path=self.cfg.poppler_path,
                    )
                except Exception as e:
                    # If this is a DecompressionBombError, attempt controlled fallbacks
                    from PIL import Image as PILImage
                    # Detect DecompressionBombError robustly (string match as well as class)
                    decompression_error = False
                    try:
                        from PIL import Image as _PILImage
                        # The specific exception class
                        DecompressionBombError = getattr(_PILImage, "DecompressionBombError", None)
                        if DecompressionBombError is not None and isinstance(e, DecompressionBombError):
                            decompression_error = True
                    except Exception:
                        decompression_error = False

                    # Additionally allow detection by exception name (some environments)
                    if not decompression_error and e.__class__.__name__ == "DecompressionBombError":
                        decompression_error = True

                    if decompression_error:
                        # 1) Retry with a lower DPI (half of configured, min 72)
                        lower_dpi = max(72, int(max(1, self.cfg.dpi) // 2))
                        try:
                            _log(self.cfg, f"DecompressionBombError rendering page {page_no} at dpi={self.cfg.dpi}; retrying at lower dpi={lower_dpi}")
                            pil_pages = convert_from_path(
                                path,
                                dpi=lower_dpi,
                                first_page=page_no,
                                last_page=page_no,
                                fmt="ppm",
                                poppler_path=self.cfg.poppler_path,
                            )
                        except Exception as e2:
                            # If still decompression bomb, attempt a single retry after temporarily increasing limit
                            try:
                                # Temporarily increase MAX_IMAGE_PIXELS only for this retry
                                orig_limit = getattr(PILImage, "MAX_IMAGE_PIXELS", None)
                                # set to a larger but bounded value (example: 1e9 pixels) rather than None
                                safe_increase = 1_000_000_000
                                PILImage.MAX_IMAGE_PIXELS = safe_increase
                                _log(self.cfg, f"Retrying page {page_no} by temporarily increasing PIL.Image.MAX_IMAGE_PIXELS to {safe_increase}")
                                pil_pages = convert_from_path(
                                    path,
                                    dpi=lower_dpi,
                                    first_page=page_no,
                                    last_page=page_no,
                                    fmt="ppm",
                                    poppler_path=self.cfg.poppler_path,
                                )
                            except Exception as e3:
                                # restore original limit before re-raising
                                try:
                                    PILImage.MAX_IMAGE_PIXELS = orig_limit
                                except Exception:
                                    pass
                                # raise an informative InvalidPDFError
                                raise InvalidPDFError(f"Failed to convert PDF to images (DecompressionBomb retries failed): {e3}")
                            else:
                                # restore original limit
                                try:
                                    PILImage.MAX_IMAGE_PIXELS = orig_limit
                                except Exception:
                                    pass
                        # end second-level error handling
                    else:
                        # Not a decompression bomb — raise wrapped error
                        raise InvalidPDFError(f"Failed to convert PDF to images: {e}")

                # if pil_pages was set successfully by any attempt, append first page
                if not pil_pages:
                    break
                images.append(pil_pages[0])
            return images
        except InvalidPDFError:
            # re-raise our meaningful error type
            raise
        except Exception as e:
            raise InvalidPDFError(f"Failed to convert PDF to images: {e}")

    def _process_single_page(self, pil_img: Image.Image, page_number: int) -> Dict[str, Any]:
        """
        Process a single PIL image with quality checks and conditional enhancements.
        Returns dict with 'image_array' and 'metadata'.
        """
        start_time = time.perf_counter()
        enhancements_applied: List[str] = []
        original_format = pil_img.mode or "UNKNOWN"
        metadata: Dict[str, Any] = {
            "original_format": original_format,
            "processed_format": "RGB",
            "dimensions": pil_img.size,  # (width, height)
            "quality_score": None,
            "enhancements_applied": [],
            "processing_time": None,
            "page_number": page_number,
        }

        # Ensure RGB
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        cropped_img, crop_info = remove_padding_from_pil(pil_img,
                                                        bg_color_sample_size=10,
                                                        tolerance=12,
                                                        min_crop_ratio=0.01,
                                                        margin_px=8)
        if crop_info.get("cropped"):
            pil_img = cropped_img
            enhancements_applied.append(f"cropped_padding_{crop_info['crop_bbox']}")
        # === End padding removal ===

        # Orientation (run after cropping)
        if self.cfg.enable_orientation:
            pil_img, rotated = _rotate_if_vertical(pil_img)
            if rotated:
                enhancements_applied.append("rotated_90_ccw")

        # Dimension check and resize if needed
        pil_img, (new_w, new_h), resized_flag = _resize_preserve_aspect(
            pil_img, self.cfg.min_dimension_px, self.cfg.max_dimension_px
        )
        if resized_flag:
            enhancements_applied.append(f"resized_to_{new_w}x{new_h}")

        # Convert to numpy for OpenCV operations
        np_rgb = _pil_to_numpy_rgb(pil_img)

        # Quality assessments
        blur_var = _measure_blur_cv2(np_rgb)
        brightness, contrast = _measure_brightness_contrast(np_rgb)
        # Compute quality score
        quality_score = _compute_quality_score(blur_var, brightness, contrast, self.cfg)

        # Track base metadata
        metadata["dimensions"] = (np_rgb.shape[1], np_rgb.shape[0])  # (w, h)
        metadata["raw_quality_metrics"] = {
            "blur_variance": blur_var,
            "brightness_mean": brightness,
            "contrast_std": contrast,
        }
        metadata["quality_score"] = round(float(quality_score), 4)

        # Enhancement pipeline
        # Sharpening if blur detected
        if self.cfg.enable_sharpen and blur_var < self.cfg.blur_threshold:
            pil_img = _apply_unsharp_mask(pil_img, radius=self.cfg.unsharp_radius,
                                          percent=self.cfg.unsharp_percent,
                                          threshold=self.cfg.unsharp_threshold)
            enhancements_applied.append("unsharp_mask")
            np_rgb = _pil_to_numpy_rgb(pil_img)
            # re-evaluate blur quickly
            blur_var = _measure_blur_cv2(np_rgb)
            metadata["raw_quality_metrics"]["blur_variance_post_sharpen"] = blur_var

        # Contrast enhancement: CLAHE if contrast low
        if self.cfg.enable_clahe and contrast < self.cfg.contrast_min:
            np_rgb = _apply_clahe_on_rgb(np_rgb)
            enhancements_applied.append("clahe")
            # update PIL image to keep consistent pipeline
            pil_img = _numpy_to_pil_rgb(np_rgb)
            brightness, contrast = _measure_brightness_contrast(np_rgb)
            metadata["raw_quality_metrics"]["contrast_std_post_clahe"] = contrast

        # Denoising if noisy
        if self.cfg.enable_denoise:
            # Heuristic: if image appears noisy (std dev high but blur low maybe skip)
            if contrast > 60 or metadata["raw_quality_metrics"].get("blur_variance_post_sharpen", blur_var) < (self.cfg.blur_threshold / 2):
                np_rgb = _apply_denoise(np_rgb, self.cfg.denoise_h)
                enhancements_applied.append("denoise")
                pil_img = _numpy_to_pil_rgb(np_rgb)
                brightness, contrast = _measure_brightness_contrast(np_rgb)
                metadata["raw_quality_metrics"]["brightness_mean_post_denoise"] = brightness
                metadata["raw_quality_metrics"]["contrast_std_post_denoise"] = contrast

        # Final quality recompute
        final_blur = _measure_blur_cv2(np_rgb)
        final_brightness, final_contrast = _measure_brightness_contrast(np_rgb)
        final_quality = _compute_quality_score(final_blur, final_brightness, final_contrast, self.cfg)
        metadata["quality_score"] = round(float(final_quality), 4)
        metadata["raw_quality_metrics"]["blur_variance_final"] = final_blur
        metadata["raw_quality_metrics"]["brightness_mean_final"] = final_brightness
        metadata["raw_quality_metrics"]["contrast_std_final"] = final_contrast

        # If final quality below configured minimum, raise QualityThresholdError
        if final_quality < self.cfg.quality_score_min:
            metadata["enhancements_applied"] = enhancements_applied
            elapsed = time.perf_counter() - start_time
            metadata["processing_time"] = round(elapsed, 4)
            metadata["processed_format"] = "RGB"
            raise QualityThresholdError(
                f"Page {page_number} quality too low: score={final_quality:.4f}"
            )

        # Normalize dimensions to be within max dimension limits
        h_final, w_final = np_rgb.shape[:2]
        if max(w_final, h_final) > self.cfg.max_dimension_px:
            scale = self.cfg.max_dimension_px / max(w_final, h_final)
            target_w = int(round(w_final * scale))
            target_h = int(round(h_final * scale))
            np_rgb = cv2.resize(np_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
            enhancements_applied.append(f"normalized_to_{target_w}x{target_h}")
            metadata["dimensions"] = (np_rgb.shape[1], np_rgb.shape[0])

        # Compose metadata
        elapsed = time.perf_counter() - start_time
        metadata["processing_time"] = round(elapsed, 4)
        metadata["enhancements_applied"] = enhancements_applied
        metadata["processed_format"] = "RGB"

        # Return numpy array and metadata
        return {
            "image_array": np_rgb,
            "metadata": metadata
        }

    def process_pdf(self, path: str) -> List[Dict[str, Any]]:
        """
        Main entrypoint to process a PDF file path.
        Returns a list of dicts each containing:
          - 'image_array': numpy ndarray (H,W,3) uint8 in RGB
          - 'metadata': dict per spec
        """
        overall_start = time.perf_counter()
        # Validate file
        self._validate_input_pdf(path)

        # Use ThreadPoolExecutor with timeout enforcement at the document level.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._process_document, path)
            try:
                result = future.result(timeout=self.cfg.timeout_seconds)
                total_elapsed = time.perf_counter() - overall_start
                _log(self.cfg, f"Document processed in {total_elapsed:.3f}s, pages: {len(result)}")
                return result
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise ProcessingTimeoutError(f"Processing exceeded timeout of {self.cfg.timeout_seconds} seconds")
            except Exception as e:
                if isinstance(e, DocumentPreprocessorError):
                    raise
                raise DocumentPreprocessorError(f"Failed to process PDF: {e}\n{traceback.format_exc()}")

    def _process_document(self, path: str) -> List[Dict[str, Any]]:
        """
        Internal method that actually performs the page-by-page processing.
        It intentionally processes pages sequentially to limit memory footprint.
        """
        results: List[Dict[str, Any]] = []
        pil_images = self._convert_pdf_pages(path, self.cfg.max_pages)

        if len(pil_images) == 0:
            raise InvalidPDFError("No pages found in PDF or unable to convert pages.")

        for idx, pil_img in enumerate(pil_images, start=1):
            _log(self.cfg, f"Processing page {idx}/{len(pil_images)}")
            try:
                page_result = self._process_single_page(pil_img, page_number=idx)
                img_arr = page_result["image_array"]
                # Ensure contiguous and uint8
                if not img_arr.flags["C_CONTIGUOUS"]:
                    img_arr = np.ascontiguousarray(img_arr)
                    page_result["image_array"] = img_arr
                if img_arr.dtype != np.uint8:
                    page_result["image_array"] = img_arr.astype(np.uint8)
                    page_result["image_array"] = page_result["image_array"].astype(np.uint8)

                try:
                    if self.cfg.save_images or self.cfg.save_dir:
                        out_dir = self.cfg.save_dir or os.path.join(os.getcwd(), "outputs", "preprocessed")
                        os.makedirs(out_dir, exist_ok=True)

                        preview_fmt = (self.cfg.preview_format or "png").lower()
                        ext = "png" if preview_fmt == "png" else "jpg"
                        base_name = os.path.splitext(os.path.basename(path))[0] if path else "doc"
                        page_fname = f"{base_name}_page_{idx}.{ext}"
                        page_path = os.path.join(out_dir, page_fname)

                        # Map to PIL format string
                        pil_format = "PNG" if ext == "png" else "JPEG"
                        save_kwargs = {}
                        if pil_format == "JPEG":
                            save_kwargs["quality"] = int(self.cfg.preview_quality or 90)

                        Image.fromarray(np.uint8(img_arr)).save(page_path, format=pil_format, **save_kwargs)
                        page_result["image_path"] = page_path
                except Exception as e_save:
                    _log(self.cfg, f"[Warning] failed to save preview for page {idx}: {e_save}")

                # --- New: optionally attach preview_bytes (small JPEG/PNG bytes) ---
                try:
                    if self.cfg.return_preview_bytes:
                        buf = io.BytesIO()
                        fmt = (self.cfg.preview_format or "png").upper()
                        if fmt == "JPG":
                            fmt = "JPEG"
                        save_kwargs = {}
                        if fmt == "JPEG":
                            save_kwargs["quality"] = int(self.cfg.preview_quality or 90)
                            save_kwargs["optimize"] = True
                        Image.fromarray(np.uint8(img_arr)).save(buf, format=fmt, **save_kwargs)
                        buf.seek(0)
                        page_result["preview_bytes"] = buf.read()
                except Exception as e_preview:
                    _log(self.cfg, f"[Warning] failed to produce preview_bytes for page {idx}: {e_preview}")

                # Ensure metadata exists (keep existing metadata structure)
                if "metadata" not in page_result or not isinstance(page_result["metadata"], dict):
                    page_result.setdefault("metadata", {"page_number": idx, "dimensions": (img_arr.shape[1], img_arr.shape[0])})

                results.append(page_result)

            except QualityThresholdError as qte:
                _log(self.cfg, f"Quality threshold failed for page {idx}: {qte}")
                raise
            except Exception as e:
                _log(self.cfg, f"Failed processing page {idx}: {e}")
                raise PDFReadError(f"Failed processing page {idx}: {e}")

        return results


def process_pdf_file(path: str, cfg: Optional[PreprocessorConfig] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to process a PDF with default or provided config.
    Returns list of dicts as defined.
    """
    pre = DocumentPreprocessor(cfg)
    return pre.process_pdf(path)

# -------------------------
# CLI: Replace existing `if __name__ == "__main__":` block with this
# -------------------------
if __name__ == "__main__":
    import argparse
    import os
    from PIL import Image
    import numpy as np

    parser = argparse.ArgumentParser(description="Document preprocessor CLI for Indian ID documents.")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--poppler-path", help="poppler binaries path (if required)", default=None)
    parser.add_argument("--max-pages", help="Max pages to process", type=int, default=5)
    parser.add_argument("--timeout", help="Timeout seconds", type=int, default=180)
    parser.add_argument("--no-clahe", help="Disable CLAHE contrast enhancement", action="store_true")
    parser.add_argument("--verbose", help="Enable verbose logging", action="store_true")
    parser.add_argument("--preview", help="Preview processed images after processing (opens OS image viewer)", action="store_true")
    parser.add_argument("--save-dir", help="Directory to save preprocessed images (PNG)", default=None)
    parser.add_argument("--quiet", help="Suppress printing page-by-page summary", action="store_true")

    args = parser.parse_args()

    cfg = PreprocessorConfig(
        poppler_path=args.poppler_path,
        max_pages=args.max_pages,
        timeout_seconds=args.timeout,
        enable_clahe=not args.no_clahe,
        verbose=args.verbose,
        save_images=args.save_images,
        save_dir=args.save_dir,
        return_preview_bytes=args.return_preview_bytes,
        preview_format=args.preview_format,
        preview_quality=args.preview_quality,
    )


    # Ensure save directory exists if provided
    if args.save_dir:
        try:
            os.makedirs(args.save_dir, exist_ok=True)
        except Exception as e:
            print(f"Failed to create save directory '{args.save_dir}': {e}")
            raise

    try:
        out = process_pdf_file(args.pdf_path, cfg)
        print(f"Processed {len(out)} pages successfully.")
        for page in out:
            md = page["metadata"]
            if not args.quiet:
                print(
                    f"Page {md['page_number']}: dims={md['dimensions']}, "
                    f"quality={md['quality_score']}, enhancements={md['enhancements_applied']}, "
                    f"time={md['processing_time']}s"
                )

            # Save image if requested
            if args.save_dir:
                try:
                    img_arr = page["image_array"]
                    save_path = os.path.join(args.save_dir, f"page_{md['page_number']}.png")
                    Image.fromarray(np.uint8(img_arr)).save(save_path)
                    if args.verbose:
                        print(f"[Saved] Page {md['page_number']} -> {save_path}")
                except Exception as e:
                    print(f"[Warning] Failed to save page {md['page_number']}: {e}")

            # Preview if requested (opens in default OS viewer)
            if args.preview:
                try:
                    img_arr = page["image_array"]
                    # Use PIL Image.show(); note: this launches an external viewer and returns immediately on many platforms.
                    Image.fromarray(np.uint8(img_arr)).show(title=f"Page {md['page_number']}")
                except Exception as e:
                    print(f"[Warning] Failed to preview page {md['page_number']}: {e}")

    except DocumentPreprocessorError as e:
        print("Processing failed:", repr(e))
        raise
