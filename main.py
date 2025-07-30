from modules.frame_detector import FrameDetector

# Initialize the frame detector
detector = FrameDetector()

# Run frame detection and ordering
detector.detect_frames("./images/test2.jpg", "frame")

# Extract frames to folder
frames = detector.extract_frames()
print(f"Frames extracted: {frames}")

# Get statistics
stats = detector.get_frame_statistics(detector.frames)
print(f"Detection statistics: {stats}")
