from modules.frame_detector import FrameDetector

# Initialize the frame detector
detector = FrameDetector()

# Run frame detection and ordering
detector.detect_frames("./images/test4.jpg", "frame")

# Extract frames to folder
output_folder = detector.extract_frames_to_folder()
print(f"Frames extracted to: {output_folder}")

# Get statistics
stats = detector.get_frame_statistics(detector.frames)
print(f"Detection statistics: {stats}")
