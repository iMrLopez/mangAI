"""
Configuration module for MangAI application
"""

import os
from typing import Dict, Any


class Config:
    """Application configuration"""
    
    # Application settings
    APP_NAME = "MangAI"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Model paths
    MODEL_PATHS = {
        "frame": "./models/yolo8l_50epochs_frame/best.pt",
        "panel": "./models/yolo8l_50epochs/best.pt",
        "text-frame": "./models/yolo8l_50epochs/best.pt"
    }
    
    # Supported languages (English only)
    SUPPORTED_LANGUAGES = {
        "en": {"name": "English", "ocr": "eng", "tts": "en-US"}
    }
    
    # Default settings
    DEFAULT_LANGUAGE = "en"  # English only
    DEFAULT_YOLO_MODEL = os.getenv("DEFAULT_YOLO_MODEL", "frame")
    TTS_SPEECH_RATE = int(os.getenv("TTS_SPEECH_RATE", "150"))
    
    # Confidence thresholds
    YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.25"))
    OCR_CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.3"))
    
    # File settings
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_TYPES = ["png", "jpg", "jpeg"]
    AUDIO_OUTPUT_DIR = "./audio_output"
    MAX_AUDIO_FILE_AGE_HOURS = int(os.getenv("MAX_AUDIO_FILE_AGE_HOURS", "24"))
    
    # OCR settings (English only)
    TESSERACT_CONFIG = os.getenv("TESSERACT_CONFIG", "--psm 6")
    
    @classmethod
    def get_model_path(cls, model_type: str) -> str:
        """Get the path for a specific model type"""
        path = cls.MODEL_PATHS.get(model_type)
        if path is None:
            raise ValueError(f"Model type '{model_type}' not found")
        return path
    
    @classmethod
    def get_language_config(cls, language: str) -> Dict[str, Any]:
        """Get configuration for a specific language"""
        return cls.SUPPORTED_LANGUAGES.get(language, cls.SUPPORTED_LANGUAGES[cls.DEFAULT_LANGUAGE])
    
    @classmethod
    def validate_config(cls) -> Dict[str, bool]:
        """Validate configuration and return status"""
        validation = {}
        
        # Check model files
        for model_type, path in cls.MODEL_PATHS.items():
            validation[f"model_{model_type}"] = os.path.exists(path)
        
        # Check directories
        validation["audio_output_dir"] = os.path.exists(cls.AUDIO_OUTPUT_DIR) or cls._create_dir(cls.AUDIO_OUTPUT_DIR)
        
        return validation
    
    @classmethod
    def _create_dir(cls, path: str) -> bool:
        """Create directory if it doesn't exist"""
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception:
            return False
