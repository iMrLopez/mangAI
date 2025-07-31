"""
OCR Processor Module
Extracts text from images using PaddleOCR
"""

import os
import cv2
import json
import logging
import numpy as np
from PIL import Image
from typing import List, Dict
from paddleocr import PaddleOCR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRProcessor:
    def __init__(
        self,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = False,
        lang: str = "en"
    ):
        """Initialize the OCR processor"""
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            lang=lang
        )

    def extract_text(self, image_paths: List[str], language: str = "en") -> List[Dict]:
        """
        Extract text from a list of image file paths using OCR.

        Args:
            image_paths (List[str]): List of image file paths
            language (str): OCR language code

        Returns:
            List[Dict]: Each dict includes file path, raw_text, cleaned_text, text_regions, and confidence
        """
        processed_frames = []

        for image_path in image_paths:
            try:
                logger.info(f"Processing image: {image_path}")
                image = Image.open(image_path).convert("RGB")
                processed_image = self._preprocess_image(image)

                json_path = self._run_ocr_on_image(image)
                raw_text = self._extract_text_from_json(json_path)

                cleaned_text = self._clean_text(raw_text)
                text_regions = self._detect_text_regions(processed_image)
                confidence = self._get_ocr_confidence(processed_image, raw_text)

                processed_frames.append({
                    "image_path": image_path,
                    "raw_text": raw_text,
                    "cleaned_text": cleaned_text,
                    "text_regions": text_regions,
                    "ocr_confidence": confidence
                })

            except Exception as e:
                logger.error(f"Error processing image {image_path}: {str(e)}")
                processed_frames.append({
                    "image_path": image_path,
                    "raw_text": "",
                    "cleaned_text": "",
                    "text_regions": [],
                    "ocr_confidence": 0.0,
                    "error": str(e)
                })

        return processed_frames

    def _run_ocr_on_image(self, image: Image.Image, output_dir: str = "ocr_output") -> str:
        """Run OCR on an image and save result to JSON"""
        image_np = np.array(image)
        result = self.ocr.predict(image_np)

        os.makedirs(output_dir, exist_ok=True)
        output_json_path = os.path.join(output_dir, "output.json")

        for res in result:
            res.save_to_img(output_dir)
            res.save_to_json(output_json_path)

        return output_json_path

    def _extract_text_from_json(self, json_path: str) -> str:
        """Extract concatenated text from PaddleOCR result JSON"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return " ".join(data.get("rec_texts", []))

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Convert to grayscale and optionally add enhancement steps"""
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if image_np.ndim == 3 else image_np
        return gray

    def _clean_text(self, raw_text: str) -> str:
        """Basic cleanup of OCR text"""
        if not raw_text:
            return ""
        cleaned = ' '.join(raw_text.split())
        cleaned = cleaned.replace('|', 'I').replace('0', 'O')
        return cleaned.strip()

    def _detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """Mock detection of text regions"""
        height, width = image.shape[:2]
        return [{
            "bbox": (width // 4, height // 4, 3 * width // 4, 3 * height // 4),
            "confidence": 0.85
        }]

    def _get_ocr_confidence(self, image: np.ndarray, text: str) -> float:
        """Mock confidence score based on text length"""
        if not text:
            return 0.0
        base_confidence = min(0.9, len(text) / 50.0)
        return max(0.1, base_confidence)
