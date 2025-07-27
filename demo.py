"""
Demo script to test the MangAI pipeline with sample images
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.frame_detector import FrameDetector
from modules.ocr_processor import OCRProcessor
from modules.tts_generator import TTSGenerator
from config import Config


def process_extracted_text_simple(extracted_texts):
    """Simple text processing without LLM"""
    text_fragments = []
    for frame in extracted_texts:
        cleaned_text = frame.get("cleaned_text", "").strip()
        if cleaned_text:
            text_fragments.append({
                "text": cleaned_text,
                "reading_order": frame.get("reading_order", frame.get("frame_id", 0))
            })
    
    if not text_fragments:
        return "No readable text was found in this manga page."
    
    # Sort by reading order
    text_fragments.sort(key=lambda x: x["reading_order"])
    
    # Combine texts
    combined_texts = [fragment["text"] for fragment in text_fragments]
    final_text = ". ".join(combined_texts)
    
    # Ensure proper ending
    if final_text and not final_text.endswith(('.', '!', '?')):
        final_text += '.'
    
    return final_text


def demo_pipeline():
    """Demonstrate the full MangAI pipeline"""
    print("🚀 MangAI Demo Pipeline")
    print("=" * 50)
    
    # Initialize configuration
    config = Config()
    print(f"📋 Configuration loaded: {config.APP_NAME} v{config.APP_VERSION}")
    
    # Check configuration
    config_status = config.validate_config()
    print("\n🔧 Configuration Status:")
    for key, status in config_status.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {key}: {status}")
    
    # Initialize modules
    print("\n🔨 Initializing modules...")
    frame_detector = FrameDetector()
    ocr_processor = OCRProcessor()
    tts_generator = TTSGenerator()
    print("  ✅ All modules initialized")
    
    # Find test images
    test_images_dir = project_root / "images"
    test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    
    if not test_images:
        print("❌ No test images found in ./images/ directory")
        return
    
    # Process first test image
    test_image = test_images[0]
    print(f"\n🖼️  Processing test image: {test_image.name}")
    
    try:
        # Step 1: Frame Detection
        print("  🔍 Step 1: Detecting frames...")
        frames = frame_detector.detect_frames(str(test_image), "frame")
        print(f"     Found {len(frames)} frames")
        
        # Show frame detection details
        for i, frame in enumerate(frames[:3]):  # Show first 3 frames
            print(f"     Frame {i}: {frame['class_name']} (confidence: {frame['confidence']:.2f})")
        
        # Extract frames to see the actual cropped images
        output_dir = frame_detector.extract_frames_to_folder()
        print(f"     Frames extracted to: {output_dir}")
        
        # Step 2: OCR Processing
        print("  📝 Step 2: Extracting text...")
        extracted_texts = ocr_processor.extract_text(frames, "en")
        text_count = len([f for f in extracted_texts if f.get("cleaned_text")])
        print(f"     Extracted text from {text_count} frames")
        
        # Show some extracted text samples
        for frame in extracted_texts[:2]:  # Show first 2 with text
            if frame.get("cleaned_text"):
                print(f"     Sample text: \"{frame['cleaned_text'][:50]}...\"")
        
        # Step 3: Text Processing (Simple combination)
        print("  📝 Step 3: Processing text...")
        processed_text = process_extracted_text_simple(extracted_texts)
        print(f"     Generated text: {len(processed_text)} characters")
        print(f"     Preview: \"{processed_text[:100]}...\"")
        
        # Step 4: TTS Generation
        print("  🎵 Step 4: Generating audio...")
        audio_path = tts_generator.generate_audio(processed_text, "en")
        print(f"     Audio saved to: {audio_path}")
        
        # Show statistics
        print("\n📊 Processing Statistics:")
        frame_stats = frame_detector.get_frame_statistics(frames)
        text_stats = ocr_processor.get_text_statistics(extracted_texts)
        
        # Simple text processing statistics
        input_chars = sum(len(frame.get("cleaned_text", "")) for frame in extracted_texts)
        output_chars = len(processed_text)
        compression_ratio = output_chars / max(1, input_chars)
        
        tts_stats = tts_generator.get_tts_statistics(processed_text)
        
        print(f"  📊 Frames: {frame_stats['count']} (avg confidence: {frame_stats.get('avg_confidence', 0):.2f})")
        print(f"  📊 Text: {text_stats['total_characters']} chars, {text_stats['frames_with_text']} frames with text")
        print(f"  📊 Processing: {compression_ratio:.2f} text ratio (input to output)")
        print(f"  📊 TTS: {tts_stats['estimated_duration_seconds']}s duration, {tts_stats['text_length_words']} words")
        
        print(f"\n✅ Demo completed successfully!")
        print(f"🎵 Generated audio file: {audio_path}")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        if config.DEBUG:
            import traceback
            traceback.print_exc()


def test_individual_modules():
    """Test each module individually"""
    print("\n🧪 Testing Individual Modules")
    print("=" * 50)
    
    # Test Frame Detector
    print("Testing Frame Detector...")
    try:
        detector = FrameDetector()
        print("  ✅ Frame Detector initialized")
    except Exception as e:
        print(f"  ❌ Frame Detector failed: {e}")
    
    # Test OCR Processor
    print("Testing OCR Processor...")
    try:
        ocr = OCRProcessor()
        print("  ✅ OCR Processor initialized")
    except Exception as e:
        print(f"  ❌ OCR Processor failed: {e}")
    
    # Test TTS Generator
    print("Testing TTS Generator...")
    try:
        tts = TTSGenerator()
        print("  ✅ TTS Generator initialized")
    except Exception as e:
        print(f"  ❌ TTS Generator failed: {e}")
    
    print("  ℹ️  LLM Processor removed - using simple text processing")


if __name__ == "__main__":
    print("🌟 Welcome to MangAI Demo!")
    print("This script demonstrates the complete manga-to-audio pipeline.\n")
    
    # Test modules first
    test_individual_modules()
    
    # Run full demo
    demo_pipeline()
    
    print("\n🎯 Next Steps:")
    print("  1. Run the web app: streamlit run app.py")
    print("  2. Or use Docker: docker-compose up --build")
    print("  3. Visit http://localhost:8501 to use the web interface")
    print("\nThank you for trying MangAI! 🚀")
