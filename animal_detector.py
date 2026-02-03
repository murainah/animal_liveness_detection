import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time


class AnimalDetector:
    """
    Animal detection and motion tracking system for livestock monitoring
    """
    
    def __init__(self, livestock_type='all', confidence_threshold=0.65):
        """
        Initialize the detector
        
        Args:
            livestock_type: 'goat', 'sheep', 'cattle', or 'all'
            confidence_threshold: 0.65 
        """
        self.model = YOLO('yolov8n.pt')  
        
        self.confidence_threshold = 0.65
        self.livestock_type = livestock_type
        self.adaptive_confidence = True
        self.base_confidence = 0.65
        self.confidence_range = (0.45, 0.75)
        self.current_confidence = 0.65
        
        # Animal class mappings (COCO dataset classes)
        self.animal_classes = {
            'sheep': 19,
            'cattle': 20,  
            'all': [19, 20] 
        }
        
        # Motion tracking for liveness detection
        self.tracks = defaultdict(list)
        self.motion_threshold = 3 
        self.bbox_history = defaultdict(list)  
        self.still_frame_count = defaultdict(int)  
        self.max_still_frames = 300  # Alert if still for more than 300 frames (~10 seconds at 30fps)
        self.animals_with_long_stillness = set() 
        
    def get_target_classes(self):
        """Get the target class IDs based on livestock type"""
        if self.livestock_type == 'all':
            return self.animal_classes['all']
        return [self.animal_classes.get(self.livestock_type, 19)]
    
    def calculate_centroid(self, bbox):
        """Calculate centroid of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def adjust_confidence_adaptively(self, frame):
        """
        Automatically adjust confidence based on lighting conditions
        Better detection in poor lighting, fewer false positives in bright conditions
        """
        if not self.adaptive_confidence:
            return self.base_confidence
        
        # Calculating frame brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        
        if brightness < 80:  
            adjusted = max(self.confidence_range[0], self.base_confidence - 0.15)
        elif brightness > 180:  
            adjusted = min(self.confidence_range[1], self.base_confidence + 0.05)
        else:  
            adjusted = self.base_confidence
        
        self.current_confidence = adjusted  
        return adjusted

    def detect_motion(self, track_id, current_centroid, current_bbox):
        """
        Detect liveness/motion in animals and track stillness duration
        
        Args:
            track_id: Unique identifier for the tracked animal
            current_centroid: Current position (x, y)
            current_bbox: Current bounding box (x1, y1, x2, y2)
            
        Returns:
            bool: True if any sign of life/movement detected in recent frames
        """
        self.tracks[track_id].append(current_centroid)
        self.bbox_history[track_id].append(current_bbox)
        
       
        if len(self.tracks[track_id]) > 20:
            self.tracks[track_id].pop(0)
            self.bbox_history[track_id].pop(0)
        
        # Need at least 5 frames to detect subtle movements
        if len(self.tracks[track_id]) < 5:
            return False
        
        has_motion = False
        
        #Checking centroid displacement
        prev_pos = self.tracks[track_id][-2]
        curr_pos = self.tracks[track_id][-1]
        
        centroid_displacement = np.sqrt(
            (curr_pos[0] - prev_pos[0])**2 + 
            (curr_pos[1] - prev_pos[1])**2
        )
        
        if centroid_displacement > self.motion_threshold:
            has_motion = True
        
        #checking  bounding box size changes
        if len(self.bbox_history[track_id]) >= 2:
            prev_bbox = self.bbox_history[track_id][-2]
            curr_bbox = self.bbox_history[track_id][-1]
            
            #Calculating bbox area change
            prev_area = (prev_bbox[2] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[1])
            curr_area = (curr_bbox[2] - curr_bbox[0]) * (curr_bbox[3] - curr_bbox[1])
            
            # If area changes by more than 2% the it alive 
            if prev_area > 0:
                area_change_percent = abs(curr_area - prev_area) / prev_area * 100
                if area_change_percent > 2:
                    has_motion = True
        
        # to Check for consistent micro-movements over time window
        if len(self.tracks[track_id]) >= 10:
            # Calculate total displacement over last 10 frames
            positions = self.tracks[track_id][-10:]
            total_movement = 0
            
            for i in range(1, len(positions)):
                displacement = np.sqrt(
                    (positions[i][0] - positions[i-1][0])**2 + 
                    (positions[i][1] - positions[i-1][1])**2
                )
                total_movement += displacement
            
            
            if total_movement > 5:
                has_motion = True
        
        # Track stillness duration
        if has_motion:
            self.still_frame_count[track_id] = 0  
            if track_id in self.animals_with_long_stillness:
                self.animals_with_long_stillness.remove(track_id)
        else:
            self.still_frame_count[track_id] += 1
            
            if self.still_frame_count[track_id] >= self.max_still_frames:
                self.animals_with_long_stillness.add(track_id)
        
        return has_motion
    
    def process_frame(self, frame):
        """
        Process a single frame for animal detection and motion tracking
        
        Args:
            frame: Input video frame
            
        Returns:
            processed_frame: Frame with annotations
            detections: Dictionary with detection results
        """
       
        results = self.model.track(
            frame, 
            persist=True,
            classes=self.get_target_classes(),
            conf=self.adjust_confidence_adaptively(frame), 
            verbose=False
        )
        
        detections = {
            'animal_count': 0,
            'animals_with_motion': 0,
            'bounding_boxes': [],
            'motion_status': []
        }
        
        annotated_frame = frame.copy()
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            
            # Getting the track ID
            track_ids = results[0].boxes.id
            if track_ids is not None:
                track_ids = track_ids.cpu().numpy().astype(int)
            else:
                track_ids = range(len(boxes))
            
            detections['animal_count'] = len(boxes)
            
            for i, (box, conf, cls_id, track_id) in enumerate(
                zip(boxes, confidences, class_ids, track_ids)
            ):
                x1, y1, x2, y2 = map(int, box)
                
                # Calculating the centroid and detect liveness
                centroid = self.calculate_centroid(box)
                has_motion = self.detect_motion(track_id, centroid, box)
                
                if has_motion:
                    detections['animals_with_motion'] += 1
                
                
                animal_label = self.livestock_type if self.livestock_type != 'all' else 'animal'
                
                
                color = (0, 255, 0) if has_motion else (0, 0, 255)
                
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
               
                label = f"{animal_label} #{track_id} ({'Alive' if has_motion else 'Still'})"
                label_with_conf = f"{label} {conf:.2f}"
                
               
                (text_width, text_height), _ = cv2.getTextSize(
                    label_with_conf, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    annotated_frame, 
                    (x1, y1 - text_height - 10), 
                    (x1 + text_width, y1), 
                    color, 
                    -1
                )
                
                
                cv2.putText(
                    annotated_frame, 
                    label_with_conf, 
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    2
                )
                
                
                cv2.circle(
                    annotated_frame, 
                    (int(centroid[0]), int(centroid[1])), 
                    5, 
                    color, 
                    -1
                )
                
                detections['bounding_boxes'].append({
                    'id': int(track_id),
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'has_motion': has_motion
                })
                
                detections['motion_status'].append(has_motion)
        
        
        has_stillness_alert = len(self.animals_with_long_stillness) > 0
        
        summary_text = [
            f"Livestock: {self.livestock_type.upper()}",
            f"Current Animals: {detections['animal_count']}",
            f"Confidence: {int(self.current_confidence * 100)}%",  
            f"Alert: {'YES - Animal(s) still too long!' if has_stillness_alert else 'No issues detected'}"
        ]
        
        
        alert_color = (0, 0, 255) if has_stillness_alert else (0, 255, 0)  
        
        y_offset = 30
        for i, text in enumerate(summary_text):
            
            text_color = alert_color if i == 2 else (255, 255, 255)
            
            cv2.putText(
                annotated_frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2
            )
            cv2.putText(
                annotated_frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                1
            )
            y_offset += 30
        
        return annotated_frame, detections
    
    def process_video(self, video_source, output_path=None, display=False):
        """
        Process video file or camera stream
        
        Args:
            video_source: Path to video file or camera index (0 for webcam)
            output_path: Path to save output video (optional)
            display: Whether to display video while processing
            
        Returns:
            summary: Dictionary with processing summary
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        max_animals_seen = 0
        total_motion_frames = 0
        animals_in_current_frame = set()
        ever_had_stillness_alert = False
        
        print(f"Processing video: {video_source}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                
                processed_frame, detections = self.process_frame(frame)
                
                # Check if any animal has been still too long
                if len(self.animals_with_long_stillness) > 0:
                    ever_had_stillness_alert = True
                
                
                frame_count += 1
                max_animals_seen = max(max_animals_seen, detections['animal_count'])
                if detections['animals_with_motion'] > 0:
                    total_motion_frames += 1
                
              
                if writer:
                    writer.write(processed_frame)
                
                if display:
                    cv2.imshow('Animal Motion Detection', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
              
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                    print(f"Progress: {progress:.1f}% - Animals: {detections['animal_count']}, "
                          f"Moving: {detections['animals_with_motion']}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        processing_time = time.time() - start_time
        
        summary = {
            'total_frames': frame_count,
            'max_animals_in_frame': max_animals_seen,
            'frames_with_motion': total_motion_frames,
            'motion_percentage': (total_motion_frames / frame_count * 100) if frame_count > 0 else 0,
            'stillness_alert': ever_had_stillness_alert,
            'alert_message': 'One or more animals showed no signs of life for extended period' if ever_had_stillness_alert else 'All animals showed normal activity',
            'processing_time': processing_time,
            'fps_processed': frame_count / processing_time if processing_time > 0 else 0,
            'livestock_type': self.livestock_type,
            'adaptive_confidence_used': True, 
            'avg_confidence': int(self.current_confidence * 100) 
        }
        
        return summary


if __name__ == "__main__":
    detector = AnimalDetector(livestock_type='sheep', confidence_threshold=0.5)
    
    summary = detector.process_video(
        video_source='/Users/abubakri/Downloads/goat_video.mp4',
        output_path='output_video.mp4',
        display=True
    )
    
    print("\n=== Processing Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
