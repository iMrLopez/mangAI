"""
OCR Processor Module
Extracts text from manga frames using OCR technology
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
import tempfile
import os


class OCRProcessor:
    def __init__(self):
        """Initialize the OCR processor for English only"""
        self.supported_languages = {
            "en": "eng"  # English only
        }
        self.current_language = "en"
    
    def set_language(self, language: str):
        """Set the OCR language (English only)"""
        if language != "en":
            raise ValueError(f"Only English is supported. Provided: {language}")
        
        self.current_language = "en"
        print(f"OCR language set to: English")
    
    def extract_text(self, frames: List[Dict], language: str = "en") -> List[Dict]:
        """
        Extract text from detected frames using OCR
        
        Args:
            frames: List of frame dictionaries from FrameDetector
            language: Language code for OCR
        
        Returns:
            List of frames with extracted text
        """
        self.set_language(language)
        
        processed_frames = []
        
        for frame in frames:
            try:
                # Get the cropped image
                cropped_image = frame["cropped_image"]
                
                # Preprocess image for better OCR
                processed_image = self._preprocess_image(cropped_image)
                
                # Extract text using OCR
                extracted_text = self._perform_ocr(processed_image)
                
                # Create processed frame dictionary
                processed_frame = {
                    **frame,  # Copy original frame data
                    "raw_text": extracted_text,
                    "cleaned_text": self._clean_text(extracted_text),
                    "text_regions": self._detect_text_regions(processed_image),
                    "ocr_confidence": self._get_ocr_confidence(processed_image, extracted_text)
                }
                
                processed_frames.append(processed_frame)
                
            except Exception as e:
                print(f"Error processing frame {frame.get('frame_id', 'unknown')}: {str(e)}")
                # Add frame with empty text on error
                processed_frame = {
                    **frame,
                    "raw_text": "",
                    "cleaned_text": "",
                    "text_regions": [],
                    "ocr_confidence": 0.0,
                    "error": str(e)
                }
                processed_frames.append(processed_frame)
        
        print(f"OCR processing complete. Processed {len(processed_frames)} frames.")
        return processed_frames
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # TODO: Implement actual preprocessing
        # - Noise reduction
        # - Contrast enhancement
        # - Binarization
        # - Deskewing
        
        # For now, return simple processing
        # Apply Gaussian blur to reduce noise
        # blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        
        # Apply threshold to get binary image
        # _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return gray
    
    def _perform_ocr(self, image: np.ndarray) -> str:
        """
        Perform OCR on preprocessed image
        """
        # TODO: Implement actual OCR using pytesseract
        # import pytesseract
        # 
        # lang_code = self.supported_languages[self.current_language]
        # 
        # # Save image temporarily for pytesseract
        # with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        #     cv2.imwrite(tmp_file.name, image)
        #     
        #     # Perform OCR
        #     text = pytesseract.image_to_string(
        #         tmp_file.name,
        #         lang=lang_code,
        #         config='--psm 6'  # Assume uniform block of text
        #     )
        #     
        #     # Clean up temporary file
        #     os.unlink(tmp_file.name)
        #     
        #     return text.strip()
        
        # Mock OCR result for scaffold
        mock_texts = [
            "Hello! How are you doing today?",
            "This is an amazing adventure!",
            "We need to find the treasure quickly!",
            "The mysterious door opens slowly...",
            "What lies beyond this point?"
        ]
        
        import random
        return random.choice(mock_texts)
    
    def _clean_text(self, raw_text: str) -> str:
        """
        Clean and normalize extracted text
        """
        if not raw_text:
            return ""
        
        # Remove extra whitespace
        cleaned = ' '.join(raw_text.split())
        
        # Remove common OCR artifacts
        cleaned = cleaned.replace('|', 'I')
        cleaned = cleaned.replace('0', 'O')  # Common OCR mistake
        
        # TODO: Add more cleaning rules based on language
        
        return cleaned.strip()
    
    def _detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Detect text regions in the image for better processing
        """
        # TODO: Implement text region detection
        # This could use techniques like:
        # - EAST text detector
        # - CRAFT text detector
        # - Contour analysis
        
        # For now, return mock regions
        height, width = image.shape[:2]
        mock_regions = [
            {
                "bbox": (width//4, height//4, 3*width//4, 3*height//4),
                "confidence": 0.85
            }
        ]
        
        return mock_regions
    
    def _get_ocr_confidence(self, image: np.ndarray, text: str) -> float:
        """
        Get confidence score for OCR results
        """
        # TODO: Implement actual confidence calculation
        # pytesseract can provide confidence scores
        
        # Mock confidence based on text length (longer text = higher confidence)
        if not text:
            return 0.0
        
        base_confidence = min(0.9, len(text) / 50.0)
        return max(0.1, base_confidence)
    
    def get_text_statistics(self, processed_frames: List[Dict]) -> Dict:
        """Get statistics about extracted text"""
        total_chars = sum(len(frame.get("cleaned_text", "")) for frame in processed_frames)
        avg_confidence = np.mean([frame.get("ocr_confidence", 0) for frame in processed_frames])
        
        return {
            "total_frames": len(processed_frames),
            "frames_with_text": len([f for f in processed_frames if f.get("cleaned_text", "")]),
            "total_characters": total_chars,
            "average_confidence": avg_confidence
        }
