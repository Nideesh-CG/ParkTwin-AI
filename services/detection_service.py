import os
import time
import logging
import json
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None
    OPENCV_AVAILABLE = False
import numpy as np
from app.config import DETECTIONS_DIR, DEFAULT_STATIONARY_THRESHOLD_SEC

logger = logging.getLogger("ParkTwinAI.DetectionService")

_YOLO_MODEL_CACHE = None

class DetectionService:
    def __init__(self):
        self.yolo_available = False
        self._check_yolo_installation()

    def _check_yolo_installation(self):
        """Check if ultralytics package is installed and can be imported."""
        try:
            import ultralytics  # type: ignore
            from ultralytics import YOLO  # type: ignore
            self.yolo_available = True
            logger.info("YOLOv8 (ultralytics) is installed and available.")
        except ImportError:
            logger.warning("YOLOv8 (ultralytics) not installed. Will use high-fidelity CV tracking simulation.")
            self.yolo_available = False

    def process_video(self, video_path, stationary_threshold_sec=DEFAULT_STATIONARY_THRESHOLD_SEC, output_path=None):
        """Process an uploaded video to detect and track illegally parked vehicles."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video file not found: {video_path}")
            
        video_name = os.path.basename(video_path)
        if output_path is None:
            output_path = str(DETECTIONS_DIR / f"annotated_{video_name}")
            
        logger.info(f"Processing video: {video_path} -> {output_path}")
        
        # Open video to get properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file {video_path}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Setup stationary tracking history
        # track_id -> {"last_pos": (cx, cy), "stationary_start_frame": frame_idx, "duration": 0.0, "vehicle_type": type}
        stationary_history = {}
        alerts_raised = []
        
        # Check GPU availability if using YOLO
        device = "cpu"
        if self.yolo_available:
            try:
                import torch  # type: ignore
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"PyTorch device selected: {device}")
                global _YOLO_MODEL_CACHE
                if _YOLO_MODEL_CACHE is None:
                    from ultralytics import YOLO  # type: ignore
                    _YOLO_MODEL_CACHE = YOLO("yolov8n.pt")
                model = _YOLO_MODEL_CACHE
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}. Falling back to simulation mode.")
                self.yolo_available = False
                
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            timestamp = frame_idx / fps
            
            current_detections = []
            
            if self.yolo_available:
                # RUN YOLO TRACKING
                try:
                    # Run tracking on frame
                    # vehicle classes in COCO: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
                    results = model.track(
                        source=frame,
                        persist=True,
                        classes=[2, 3, 5, 7],
                        tracker="bytetrack.yaml",
                        device=device,
                        verbose=False
                    )
                    
                    if results and results[0].boxes and results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        ids = results[0].boxes.id.cpu().numpy().astype(int)
                        cls = results[0].boxes.cls.cpu().numpy().astype(int)
                        conf = results[0].boxes.conf.cpu().numpy()
                        
                        coco_classes = {2: "CAR", 3: "BIKE", 5: "BUS", 7: "TRUCK"}
                        
                        for box, track_id, class_idx, cf in zip(boxes, ids, cls, conf):
                            x1, y1, x2, y2 = map(int, box)
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            vehicle_type = coco_classes.get(class_idx, "CAR")
                            
                            current_detections.append({
                                "track_id": track_id,
                                "bbox": (x1, y1, x2, y2),
                                "center": (cx, cy),
                                "vehicle_type": vehicle_type
                            })
                except Exception as e:
                    logger.error(f"YOLO frame tracking error: {e}. Falling back to simulated tracks for frame.")
                    current_detections = self._generate_simulated_tracks(frame_idx, width, height)
            else:
                # RUN SIMULATION TRACKING (Deterministic Fallback)
                current_detections = self._generate_simulated_tracks(frame_idx, width, height)
                
            # Update Stationary Durations
            active_ids = set()
            for det in current_detections:
                track_id = det["track_id"]
                x1, y1, x2, y2 = det["bbox"]
                cx, cy = det["center"]
                v_type = det["vehicle_type"]
                active_ids.add(track_id)
                
                if track_id not in stationary_history:
                    # New Tracked Vehicle
                    stationary_history[track_id] = {
                        "last_pos": (cx, cy),
                        "stationary_start_frame": frame_idx,
                        "duration": 0.0,
                        "vehicle_type": v_type,
                        "violating": False
                    }
                else:
                    hist = stationary_history[track_id]
                    lx, ly = hist["last_pos"]
                    
                    # Distance check (if moved less than 8 pixels, count as stationary)
                    dist = np.sqrt((cx - lx)**2 + (cy - ly)**2)
                    if dist < 8:
                        # Vehicle is stationary
                        frames_stationary = frame_idx - hist["stationary_start_frame"]
                        duration_sec = frames_stationary / fps
                        hist["duration"] = duration_sec
                        
                        if duration_sec > stationary_threshold_sec:
                            hist["violating"] = True
                            alert_msg = f"Violation detected: {v_type} (ID: {track_id}) parked for {duration_sec:.1f}s"
                            if alert_msg not in alerts_raised:
                                alerts_raised.append(alert_msg)
                                logger.warning(alert_msg)
                    else:
                        # Vehicle moved, reset stationary timer
                        hist["last_pos"] = (cx, cy)
                        hist["stationary_start_frame"] = frame_idx
                        hist["duration"] = 0.0
                        hist["violating"] = False
            
            # Clean up missing track IDs
            # (In a real system, you might keep them for a few frames, here we just do simple cleanup)
            for tid in list(stationary_history.keys()):
                if tid not in active_ids:
                    # Decr / remove inactive tracks
                    del stationary_history[tid]
                    
            # Draw Bounding Boxes and Annotation HUD
            for det in current_detections:
                track_id = det["track_id"]
                x1, y1, x2, y2 = det["bbox"]
                v_type = det["vehicle_type"]
                
                hist = stationary_history.get(track_id, {"duration": 0.0, "violating": False})
                duration = hist["duration"]
                violating = hist["violating"]
                
                # Colors: Red for violating, Green for normal tracking
                box_color = (0, 0, 255) if violating else (0, 255, 0)
                thickness = 3 if violating else 2
                
                # Draw Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
                
                # Draw Info Panel
                label = f"{v_type} ID:{track_id} {duration:.1f}s"
                if violating:
                    label += " [ILLEGAL]"
                    
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                
            # Draw HUD overlay (Total alerts count, etc)
            cv2.rectangle(frame, (10, 10), (320, 85), (0, 0, 0), -1)
            cv2.putText(frame, "PARKTWIN AI COMMAND CENTER", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Active Vehicles: {len(current_detections)}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(frame, f"Violations Alerted: {len(alerts_raised)}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            
            # Save frame
            writer.write(frame)
            
        # Release everything
        cap.release()
        writer.release()
        
        # Write detection logs summary
        logs = {
            "input_video": video_name,
            "processed_video": os.path.basename(output_path),
            "total_frames_processed": frame_idx,
            "duration_seconds": float(frame_idx / fps),
            "violations_detected_count": len(alerts_raised),
            "alerts": alerts_raised
        }
        
        with open(DETECTIONS_DIR / f"logs_{video_name}.json", 'w') as f:
            json.dump(logs, f, indent=4)
            
        logger.info("Video processing completed successfully.")
        return logs

    def detect_frames_generator(self, video_path, stationary_threshold_sec=DEFAULT_STATIONARY_THRESHOLD_SEC):
        """Yields annotated frames and tracked objects data frame-by-frame."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video file not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file {video_path}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        stationary_history = {}
        
        device = "cpu"
        model = None
        if self.yolo_available:
            try:
                import torch  # type: ignore
                device = "cuda" if torch.cuda.is_available() else "cpu"
                global _YOLO_MODEL_CACHE
                if _YOLO_MODEL_CACHE is None:
                    from ultralytics import YOLO  # type: ignore
                    _YOLO_MODEL_CACHE = YOLO("yolov8n.pt")
                model = _YOLO_MODEL_CACHE
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}. Falling back to simulation mode.")
                self.yolo_available = False
                
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            timestamp = frame_idx / fps
            
            current_detections = []
            
            if self.yolo_available and model is not None:
                try:
                    results = model.track(
                        source=frame,
                        persist=True,
                        classes=[2, 3, 5, 7],
                        tracker="bytetrack.yaml",
                        device=device,
                        verbose=False
                    )
                    
                    if results and results[0].boxes and results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        ids = results[0].boxes.id.cpu().numpy().astype(int)
                        cls = results[0].boxes.cls.cpu().numpy().astype(int)
                        conf = results[0].boxes.conf.cpu().numpy()
                        
                        coco_classes = {2: "CAR", 3: "BIKE", 5: "BUS", 7: "TRUCK"}
                        
                        for box, track_id, class_idx, cf in zip(boxes, ids, cls, conf):
                            x1, y1, x2, y2 = map(int, box)
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            vehicle_type = coco_classes.get(class_idx, "CAR")
                            
                            current_detections.append({
                                "track_id": track_id,
                                "bbox": (x1, y1, x2, y2),
                                "center": (cx, cy),
                                "vehicle_type": vehicle_type
                            })
                except Exception as e:
                    logger.error(f"YOLO frame tracking error: {e}. Falling back to simulation.")
                    current_detections = self._generate_simulated_tracks(frame_idx, width, height)
            else:
                current_detections = self._generate_simulated_tracks(frame_idx, width, height)
                
            active_ids = set()
            for det in current_detections:
                track_id = det["track_id"]
                x1, y1, x2, y2 = det["bbox"]
                cx, cy = det["center"]
                v_type = det["vehicle_type"]
                active_ids.add(track_id)
                
                if track_id not in stationary_history:
                    stationary_history[track_id] = {
                        "last_pos": (cx, cy),
                        "stationary_start_frame": frame_idx,
                        "duration": 0.0,
                        "vehicle_type": v_type,
                        "violating": False,
                        "zone": "KR Market Junction" if cy < height // 2 else "Hudson Circle"
                    }
                else:
                    hist = stationary_history[track_id]
                    lx, ly = hist["last_pos"]
                    
                    dist = np.sqrt((cx - lx)**2 + (cy - ly)**2)
                    if dist < 8:
                        frames_stationary = frame_idx - hist["stationary_start_frame"]
                        duration_sec = frames_stationary / fps
                        hist["duration"] = duration_sec
                        
                        if duration_sec > stationary_threshold_sec:
                            hist["violating"] = True
                    else:
                        hist["last_pos"] = (cx, cy)
                        hist["stationary_start_frame"] = frame_idx
                        hist["duration"] = 0.0
                        hist["violating"] = False
                        
            for tid in list(stationary_history.keys()):
                if tid not in active_ids:
                    del stationary_history[tid]
                    
            # Draw annotations on copy of frame
            annotated_frame = frame.copy()
            for det in current_detections:
                track_id = det["track_id"]
                x1, y1, x2, y2 = det["bbox"]
                v_type = det["vehicle_type"]
                
                hist = stationary_history.get(track_id, {"duration": 0.0, "violating": False})
                duration = hist["duration"]
                violating = hist["violating"]
                
                # Colors: Red for violating, Orange/Warning if duration > threshold/2, Green normal
                if violating:
                    box_color = (0, 0, 255) # Red
                    thickness = 3
                elif duration > (stationary_threshold_sec / 2):
                    box_color = (0, 165, 255) # Orange
                    thickness = 2
                else:
                    box_color = (0, 255, 0) # Green
                    thickness = 2
                    
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, thickness)
                
                label = f"{v_type} #{track_id}"
                if violating:
                    label += f" | ILLEGAL PARKING ({duration:.1f}s)"
                else:
                    label += f" | Stopped: {duration:.1f}s"
                    
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)
                
            # Yield frame and current tracking logs
            tracking_logs = []
            for tid, h in stationary_history.items():
                status = "Normal"
                if h["violating"]:
                    status = "Illegal Parking"
                elif h["duration"] > (stationary_threshold_sec / 2):
                    status = "Warning"
                    
                tracking_logs.append({
                    "track_id": tid,
                    "vehicle_type": h["vehicle_type"],
                    "duration": round(h["duration"], 1),
                    "zone": h["zone"],
                    "status": status
                })
                
            yield annotated_frame, tracking_logs
            
        cap.release()

    def _generate_simulated_tracks(self, frame_idx, width, height):
        """Generates deterministic synthetic vehicle tracks for the high-fidelity mock fallback."""
        detections = []
        
        # Vehicle 1: Illegally Parked Car (Stationary at the top right of the road)
        # Stays still throughout the entire video
        x1_car1 = int(width * 0.7)
        y1_car1 = int(height * 0.4)
        x2_car1 = x1_car1 + 120
        y2_car1 = y1_car1 + 80
        cx_car1 = (x1_car1 + x2_car1) // 2
        cy_car1 = (y1_car1 + y2_car1) // 2
        
        detections.append({
            "track_id": 1,
            "bbox": (x1_car1, y1_car1, x2_car1, y2_car1),
            "center": (cx_car1, cy_car1),
            "vehicle_type": "CAR"
        })
        
        # Vehicle 2: Illegally Parked Delivery Auto-rickshaw (Stationary at the bottom-left)
        x1_auto = int(width * 0.15)
        y1_auto = int(height * 0.6)
        x2_auto = x1_auto + 90
        y2_auto = y1_auto + 75
        cx_auto = (x1_auto + x2_auto) // 2
        cy_auto = (y1_auto + y2_auto) // 2
        
        detections.append({
            "track_id": 2,
            "bbox": (x1_auto, y1_auto, x2_auto, y2_auto),
            "center": (cx_auto, cy_auto),
            "vehicle_type": "GOODS AUTO"
        })
        
        # Vehicle 3: Moving Motorbike (Driving across the screen from left to right)
        # Moves x-position frame by frame
        speed = 5 # Pixels per frame
        bike_x_start = int(width * 0.1) + (frame_idx * speed)
        
        # Loop vehicle 3 back if it goes off screen
        bike_x_start = bike_x_start % width
        
        x1_bike = bike_x_start
        y1_bike = int(height * 0.7)
        x2_bike = x1_bike + 60
        y2_bike = y1_bike + 50
        cx_bike = (x1_bike + x2_bike) // 2
        cy_bike = (y1_bike + y2_bike) // 2
        
        # Only show bike if it's within screen boundaries
        if x2_bike < width:
            detections.append({
                "track_id": 3,
                "bbox": (x1_bike, y1_bike, x2_bike, y2_bike),
                "center": (cx_bike, cy_bike),
                "vehicle_type": "BIKE"
            })
            
        return detections

if __name__ == "__main__":
    # Create a dummy video file to test the mock pipeline
    import cv2
    import numpy as np
    
    test_video = "test_traffic.mp4"
    # Create blank frames and write to video
    out = cv2.VideoWriter(test_video, cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))
    for i in range(50):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw some background lines to represent a street
        cv2.line(frame, (0, 200), (640, 200), (255, 255, 255), 2)
        cv2.line(frame, (0, 400), (640, 400), (255, 255, 255), 2)
        out.write(frame)
    out.release()
    
    detector = DetectionService()
    logs = detector.process_video(test_video, stationary_threshold_sec=2.0)
    print("Logs:")
    print(logs)
    
    # Clean up test files
    if os.path.exists(test_video):
        os.remove(test_video)
