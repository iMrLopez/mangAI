"""
Frame Detector Module
Uses YOLO models to detect and extract manga frames/panels from images
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import os
from datetime import datetime
from ultralytics import YOLO
from config import Config


class FrameDetector:
    def __init__(self):
        """Initialize the frame detector with available YOLO models"""
        self.config = Config()
        self.model_paths = self.config.MODEL_PATHS
        self.current_model = None
        self.current_model_type = None
        self.confidence_threshold = self.config.YOLO_CONFIDENCE_THRESHOLD
        
        # Class definitions
        self.class_names = ["frame", "face", "text", "body"]
        self.allowed_classes = []
        
        # Internal state
        self.frames = []
        self.current_image_path = ""
    
    def load_model(self, model_type: str, confidence: Optional[float] = None):
        """Load the specified YOLO model"""
        if model_type not in self.model_paths:
            raise ValueError(f"Model type '{model_type}' not available. Choose from: {list(self.model_paths.keys())}")
        
        model_path = self.model_paths[model_type]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load YOLO model
        self.current_model = YOLO(model_path)
        self.current_model_type = model_type
        
        if confidence is not None:
            self.confidence_threshold = confidence
        
        # Set allowed classes based on model type
        self._set_allowed_classes(model_type)
        
        print(f"Model '{model_type}' loaded from {model_path} with confidence {self.confidence_threshold}")
    
    def _set_allowed_classes(self, model_type: str):
        """Set allowed classes based on model type"""
        self.allowed_classes = []
        
        for i, cls in enumerate(self.class_names):
            if model_type == "frame" and i == 1:  # Skip face class for frame detection
                break
            elif model_type == "text-frame" and i == 1:  # Skip face class for text-frame detection
                continue
            else:
                self.allowed_classes.append(cls)
    
    def detect_frames(self, image_path: str, model_type: str = "frame") -> List[Dict]:
        """
        Detect frames/panels in the manga image
        
        Args:
            image_path: Path to the manga image
            model_type: Type of model to use ('frame', 'panel', or 'text-frame')
        
        Returns:
            List of detected frames with bounding boxes and cropped images
        """
        # Load model if different from current
        if self.current_model_type != model_type:
            self.load_model(model_type)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.current_image_path = image_path
        
        # Run YOLO detection
        if self.current_model is None:
            raise RuntimeError(f"Model not loaded. Call load_model() first.")
        
        results = self.current_model(image_path, conf=self.confidence_threshold)
        
        # Process results
        frames = self._process_yolo_results(results[0], image)
        
        # Order frames by reading sequence
        ordered_frames = self.order_frames(frames)
        
        self.frames = ordered_frames
        return ordered_frames
    
    def _process_yolo_results(self, result, image: np.ndarray) -> List[Dict]:
        """Process YOLO detection results into frame dictionaries"""
        frames = []
        
        if result.boxes is None:
            return frames
        
        class_ids = result.boxes.cls.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        for i, cls_id in enumerate(class_ids):
            if self.current_model is None:
                continue
                
            class_name = self.current_model.names[int(cls_id)]
            
            if class_name in self.allowed_classes:
                box = boxes[i]
                x1, y1, x2, y2 = map(int, box)
                
                # Create frame dictionary
                frame = {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(confidences[i]),
                    "cropped_image": image[y1:y2, x1:x2],
                    "frame_id": i,
                    "class_name": class_name,
                    "area": (x2 - x1) * (y2 - y1),
                    # Additional fields for ordering algorithm
                    "xmin": x1,
                    "ymin": y1,
                    "xmax": x2,
                    "ymax": y2,
                    "analyzed": False,
                    "rank": 1000,
                    "reading_order": -1
                }
                
                frames.append(frame)
        
        print(f"Detected {len(frames)} frames/panels")
        return frames
    
    def order_frames(self, frames: List[Dict]) -> List[Dict]:
        """
        Order frames based on manga reading order using the original algorithm
        
        Args:
            frames: List of detected frames
        
        Returns:
            Ordered list of frames
        """
        if not frames:
            return frames
        
        # Get image dimensions from the first frame's parent image
        image = cv2.imread(self.current_image_path)
        height, width = image.shape[:2]
        
        # Prepare frames for ordering algorithm
        frames_list = []
        for frame in frames:
            algorithm_object = {
                "distance": 0,
                "analyzed": "no",
                "rank": 1000,
                "xmin": frame["xmin"],
                "ymin": frame["ymin"],
                "xmax": frame["xmax"],
                "ymax": frame["ymax"],
                "xmin_norm": frame["xmin"] / width,
                "ymin_norm": frame["ymin"] / height,
                "xmax_norm": frame["xmax"] / width,
                "ymax_norm": frame["ymax"] / height,
                "candidate": "no",
                "original_frame": frame  # Keep reference to original frame data
            }
            frames_list.append(algorithm_object)
        
        # Apply ordering algorithm
        tolerance = 0.03
        global_rank = 0
        target = np.array([1.0, 0.0])  # Initial target (top-right corner)
        
        while global_rank < len(frames_list):
            global_rank, y_min_tol, index_updated_element = self._get_frame_top_right_corner(
                frames_list, global_rank, tolerance, target
            )
            target = np.array([
                frames_list[index_updated_element]["xmin_norm"],
                frames_list[index_updated_element]["ymin_norm"]
            ])
            global_rank = self._search_horizontal_frames(global_rank, frames_list, y_min_tol, target)
            target = np.array([
                frames_list[index_updated_element]["xmax_norm"],
                frames_list[index_updated_element]["ymax_norm"]
            ])
        
        # Sort by rank and update original frames
        frames_list.sort(key=lambda x: x["rank"])
        
        ordered_frames = []
        for i, frame_data in enumerate(frames_list):
            original_frame = frame_data["original_frame"]
            original_frame["reading_order"] = i
            original_frame["rank"] = frame_data["rank"]
            ordered_frames.append(original_frame)
        
        return ordered_frames
    
    def _search_horizontal_frames(self, rank: int, tmp_array: List[Dict], y_min_tol: float, target: np.ndarray) -> int:
        """Search for frames at the same horizontal level"""
        any_frame_at_same_horizontal_level = "no"
        candidates = []
        
        for frame in tmp_array:
            if frame["analyzed"] == "yes":
                continue
            elif frame["ymax_norm"] < y_min_tol:
                # Calculate distance from previous frame identified = target
                top_right_corner = np.array([frame["xmax_norm"], frame["ymin_norm"]])
                frame["distance"] = abs(np.linalg.norm(top_right_corner - target))
                frame["candidate"] = "yes"
                any_frame_at_same_horizontal_level = "yes"
                candidates.append(frame)
        
        if any_frame_at_same_horizontal_level == "yes":
            # Find minimum distance
            min_distance = min(obj["distance"] for obj in candidates)
            # Assign ascending rank
            for frame in tmp_array:
                if min_distance == frame["distance"]:
                    frame["analyzed"] = "yes"
                    frame["rank"] = rank
                    rank = rank + 1
        
        return rank
    
    def _get_frame_top_right_corner(self, tmp_array: List[Dict], rank: int, tolerance: float, target: np.ndarray) -> Tuple[int, float, int]:
        """Get frame with top right corner closest to target"""
        candidates = []
        
        for frame in tmp_array:
            if frame["analyzed"] == "yes":
                continue
            else:
                top_right_corner = np.array([frame["xmax_norm"], frame["ymin_norm"]])
                frame["distance"] = np.linalg.norm(top_right_corner - target)
                frame["candidate"] = "yes"
                candidates.append(frame)
        
        # Find minimum distance
        min_distance = min(obj["distance"] for obj in candidates)
        
        min_index = 0
        y_min_tol = 0.0
        for i, frame in enumerate(tmp_array):
            if frame["distance"] == min_distance:
                frame["analyzed"] = "yes"
                frame["rank"] = rank
                rank = rank + 1
                y_min_tol = frame["ymax_norm"] * (1 + tolerance)
                min_index = i
            frame["candidate"] = "no"
        
        return rank, y_min_tol, min_index
    
    def extract_frames_to_folder(self, output_dir: Optional[str] = None) -> str:
        """
        Extract detected frames to individual image files
        
        Args:
            output_dir: Output directory path. If None, creates timestamped folder
        
        Returns:
            Path to the output directory
        """
        if not self.frames:
            raise ValueError("No frames detected. Run detect_frames() first.")
        
        if output_dir is None:
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = f"extracted_frames_{timestamp_str}"
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load original image
        image = cv2.imread(self.current_image_path)
        
        # Extract and save each frame
        for frame in self.frames:
            x1, y1, x2, y2 = frame["bbox"]
            cropped = image[y1:y2, x1:x2]
            
            frame_number = frame.get("reading_order", frame.get("rank", frame["frame_id"]))
            filename = f"frame_{frame_number:03d}_{frame['class_name']}.jpg"
            save_path = os.path.join(output_dir, filename)
            
            cv2.imwrite(save_path, cropped)
        
        print(f"Extracted {len(self.frames)} frames to {output_dir}")
        return output_dir
    
    def visualize_detections(self, image_path: str, model_type: str = "frame", save_path: Optional[str] = None):
        """
        Visualize detection results with bounding boxes
        
        Args:
            image_path: Path to the manga image
            model_type: Type of model to use
            save_path: Path to save visualization (optional)
        """
        # Load model if needed
        if self.current_model_type != model_type:
            self.load_model(model_type)
        
        if self.current_model is None:
            raise RuntimeError(f"Model not loaded. Call load_model() first.")
        
        # Run detection with visualization
        results = self.current_model(image_path, conf=self.confidence_threshold, save=True)
        
        # Display results
        for r in results:
            im_array = r.plot()  # BGR image with boxes
            im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
            
            if save_path:
                cv2.imwrite(save_path, im_array)
                print(f"Visualization saved to {save_path}")
            
            # For Jupyter/interactive environments
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 8))
                plt.imshow(im_rgb)
                plt.title(f"YOLOv8 Detection Results - {model_type}")
                plt.axis("off")
                plt.show()
            except ImportError:
                print("Matplotlib not available for visualization")
    
    def get_frame_statistics(self, frames: List[Dict]) -> Dict:
        """Get statistics about detected frames"""
        if not frames:
            return {"count": 0}
        
        confidences = [frame["confidence"] for frame in frames]
        areas = [frame.get("area", 0) for frame in frames]
        class_counts = {}
        
        for frame in frames:
            class_name = frame.get("class_name", "unknown")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            "count": len(frames),
            "avg_confidence": np.mean(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences),
            "avg_area": np.mean(areas) if areas else 0,
            "class_distribution": class_counts,
            "total_area": sum(areas)
        }
    
    def get_segments_detected(self, image_path: str, model_type: str = "frame") -> List[Tuple[str, List[float]]]:
        """
        Return a list of detected segments as (class_name, [x1, y1, x2, y2])
        This method maintains compatibility with the original interface
        """
        frames = self.detect_frames(image_path, model_type)
        segments = []
        
        for frame in frames:
            class_name = frame["class_name"]
            bbox = list(frame["bbox"])  # Convert tuple to list
            segments.append((class_name, bbox))
        
        return segments
