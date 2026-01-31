# Animal Motion Detection System

An end-to-end livestock motion detection system using YOLOv8 and Flask. This application detects animals in video feeds (uploaded videos or live camera), tracks their movements, and provides detailed analytics about animal count and motion/liveness detection.

## Features

- **Multiple Input Sources**: 
  - Upload video files (MP4, AVI, MOV, MKV)
  - Live camera feed support
  
- **Livestock Type Selection**: 
  - Goat and Sheep
  - Cattle
  - All animals
  
- **Real-time Detection & Tracking**:
  - Animal detection using YOLOv8
  - **Liveness detection** with multi-method analysis:
    - Centroid tracking (head movements, walking)
    - Bounding box size changes (breathing, body expansion)
    - Micro-movement analysis (ear twitching, slight shifts)
  - Individual animal tracking with unique IDs
  - Visual indicators (green for alive/showing liveness, red for no signs detected)
  
- **Comprehensive Analytics**:
  - Total animal count
  - Number of animals with motion
  - Motion percentage
  - Processing statistics
  - Individual animal status
  
- **Web Interface**:
  - User-friendly Flask web application
  - Real-time camera streaming
  - Video download capability
  - Responsive design

## System Requirements

- Python 3.8+
- Webcam (optional, for live detection)
- 4GB+ RAM recommended
- CUDA-compatible GPU (optional, for faster processing)

## Installation

### 1. Clone or Download the Project

```bash
# Create project directory
mkdir animal_liveness_detection
cd animal_liveness_detection
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv ald

# Activate virtual environment
# On Windows:
amd\Scripts\activate
# On Linux/Mac:
source ald/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The first time you run the application, YOLOv8 will automatically download the model weights (~6MB for yolov8n.pt).

## Project Structure

```
animal_liveness_detection/
│
├── animal_detector.py      # Core detection and tracking logic
├── app.py                  # Flask web application
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── templates/
│   └── index.html         # Web interface template
│
├── static/
│   ├── css/
│   │   └── style.css      # Styling
│   └── js/
│       └── script.js      # Frontend JavaScript
│
├── uploads/               # Temporary uploaded videos (auto-created)
└── outputs/               # Processed videos (auto-created)
```

## Usage

### Starting the Application

1. **Activate your virtual environment** (if using one)

2. **Run the Flask application**:
   ```bash
   python app.py
   ```

3. **Open your browser** and navigate to:
   ```
   http://localhost:5001
   ```

### Using the Web Interface

#### Configuration

1. **Select Livestock Type**: Choose from Goat, Sheep, Cattle, or All Animals
2. **Adjust Detection Confidence**: Use the slider to set detection threshold (10%-95%)

#### Option 1: Upload Video

1. Click the **"Upload Video"** tab
2. Drag and drop a video file or click "Choose File"
3. Click **"Start Processing"**
4. Wait for processing to complete
5. View results and download the processed video

#### Option 2: Live Camera

1. Click the **"Live Camera"** tab
2. Select your camera source
3. Click **"Start Camera"**
4. View real-time detections and statistics
5. Click **"Stop Camera"** when done

### Command Line Usage

You can also use the animal detector directly from Python:

```python
from animal_detector import AnimalDetector

# Initialize detector
detector = AnimalDetector(
    livestock_type='sheep',  # or 'cattle', 'all'
    confidence_threshold=0.5
)

# Process video file
summary = detector.process_video(
    video_source='path/to/video.mp4',
    output_path='path/to/output.mp4',
    display=True  # Show video while processing
)

# Print results
print(summary)
```

### Processing Webcam Stream

```python
from animal_detector import AnimalDetector

detector = AnimalDetector(livestock_type='cattle')


summary = detector.process_video(
    video_source=0,  # 0 for default webcam
    display=True
)
```

## Understanding the Output

### Visual Annotations

- **Green Bounding Box**: Animal shows signs of life (breathing, moving, ear twitching)
- **Red Bounding Box**: No clear signs of life detected (may need attention)
- **Dot in Center**: Animal centroid (tracking point)
- **Label**: Shows animal type, ID, liveness status, and confidence

### Statistics Provided

**Primary Animal Info:**
1. **Total Unique Animals**: Total number of individual animals detected throughout the video
2. **Animals Showing Life**: Number of animals that showed any signs of life (breathing, movement, etc.)
3. **Animals with No Signs**: Number of animals that showed no detectable signs of life (⚠️ need attention)


**Individual Animal Tracking:**
- **Alive Animal IDs**: List of animal IDs that showed signs of life (e.g., 1, 2, 3, 5, 7)
- **No Signs IDs**: List of animal IDs with no detected signs (e.g., 4, 6) - these need immediate attention

### Liveness Detection

The system detects signs of life through multiple methods:

**Method 1: Centroid Movement**
- Tracks the center point of each animal
- Detects head movements, walking, and position changes
- Threshold: 3 pixels (very sensitive)

**Method 2: Bounding Box Changes**
- Monitors changes in the animal's detected size
- Detects breathing (chest expansion/contraction)
- Triggers on 2% or more area change

**Method 3: Micro-Movement Analysis**
- Analyzes movement patterns over 10 frames
- Detects subtle movements like ear twitching, tail swishing
- Accumulates tiny movements that indicate life

**Result**: Animals showing ANY of these signs are marked as "Alive" (green), while those with no detectable movement for several frames are marked as needing attention (red).

## Configuration Options

### In `animal_detector.py`

```python
# Liveness detection sensitivity (pixels)
self.motion_threshold = 3  # Lower = detects smaller movements (breathing, ear twitching)

# Tracking history (frames)
if len(self.tracks[track_id]) > 20:  # Adjust for longer/shorter history
```

**Adjusting Sensitivity:**
- `motion_threshold = 1-2`: Ultra-sensitive (detects breathing easily)
- `motion_threshold = 3-5`: Balanced (detects most signs of life) ← Default
- `motion_threshold = 10+`: Less sensitive (only obvious movements)

### In `app.py`

```python
# Maximum upload file size
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# Server configuration
app.run(debug=True, host='0.0.0.0', port=5001)
```

## Troubleshooting

### Issue: "Could not open video source"
- **Solution**: Check camera permissions or verify video file path

### Issue: "No animals detected"
- **Solution**: 
  - Lower the confidence threshold
  - Ensure video contains sheep, goats, or cattle
  - Check lighting conditions in video

### Issue: Slow processing
- **Solution**:
  - Use a GPU (CUDA) if available
  - Reduce video resolution
  - Use a lighter YOLO model (already using yolov8n - nano)

### Issue: Too many false "no signs of life" detections
- **Solution**: Decrease `motion_threshold` in `animal_detector.py` (e.g., from 3 to 1)

### Issue: Everything marked as "alive" even when still
- **Solution**: Increase `motion_threshold` in `animal_detector.py` (e.g., from 3 to 5 or 7)

## Model Information

This system uses **YOLOv8n** (Nano) for object detection:
- **Speed**: ~45 FPS on CPU, ~200+ FPS on GPU
- **Accuracy**: Optimized for real-time performance
- **Classes**: Trained on COCO dataset
  - Class 19: Sheep (also used for goats)
  - Class 20: Cow (cattle)

### Upgrading to Better Models

For higher accuracy, you can use larger models:

```python
# In animal_detector.py, change:
self.model = YOLO('yolov8n.pt')  # Nano (fastest)
# to:
self.model = YOLO('yolov8s.pt')  # Small
self.model = YOLO('yolov8m.pt')  # Medium
self.model = YOLO('yolov8l.pt')  # Large
self.model = YOLO('yolov8x.pt')  # Extra Large (most accurate)
```

## API Endpoints

### Upload Video
```
POST /upload
Content-Type: multipart/form-data

Parameters:
- video: Video file
- livestock_type: 'sheep and goat', 'cattle', or 'all'
- confidence: Float (0.1 to 0.95)

Returns: JSON with processing summary
```

### Start Camera
```
POST /camera/start
Content-Type: application/json

Body:
{
  "livestock_type": "sheep",
  "confidence": 0.5,
  "camera_index": 0
}

Returns: Success status
```

### Stop Camera
```
POST /camera/stop

Returns: Success status
```

### Get Camera Stats
```
GET /camera/stats

Returns: JSON with current detection statistics
```

### Camera Feed
```
GET /camera/feed

Returns: MJPEG stream
```

## Performance Optimization

1. **Use GPU acceleration**: Install CUDA-enabled PyTorch
2. **Reduce video resolution**: Process at lower resolution for speed
3. **Use batch processing**: Process multiple frames at once (advanced)

## Future Enhancements

Potential improvements for the system:
- [ ] Support for custom-trained models tailored to specific livestock breeds
- [ ] Activity classification (eating, sleeping, standing, walking)
- [ ]  health-related behavioral monitoring
- [ ] Multi-camera and multi-pen support
- [ ] Alert system for abnormal or low-activity behavior
- [ ] Historical data analytics and long-term behavior trends
- [ ] Feed intake and grazing pattern estimation
- [ ] Automated stocking density monitoring
- [ ] Edge deployment for low-connectivity farm environments
- [ ] Lightweight model optimization for real-time inference
- [ ] Temporal modeling of behavior using video sequences
- [ ] Anomaly detection for early identification of unusual patterns
- [ ] Explainable AI for livestock behavior interpretation
- [ ] Farmer dashboard and mobile interface


## License

This project is provided as-is for educational and commercial use.

## Acknowledgments

- **YOLOv8** by Ultralytics
- **Flask** web framework
- **OpenCV** for video processing

## Support

**Happy Detecting!**
