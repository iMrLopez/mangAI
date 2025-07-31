from modules.frame_detector import FrameDetector
from modules.ocr_processor import OCRProcessor

# Initialize the frame detector
detector = FrameDetector()
ocr = OCRProcessor()

# Run frame detection and ordering
detector.detect_frames("./images/test2.jpg", "frame")

# Extract frames to folder
frames = detector.extract_frames()
print(f"Frames extracted: {frames}")

# Get statistics
stats = detector.get_frame_statistics(detector.frames)
print(f"Detection statistics: {stats}")

# Extract text via OCR module
results = ocr.extract_text(frames, "en")
for res in results:
    print(res["image_path"], res["cleaned_text"], f"Conf: {res['ocr_confidence']:.2f}")