from flask import Flask, render_template, request, jsonify, Response, send_file
import cv2
import os
import uuid
import json
from datetime import datetime
from animal_detector import AnimalDetector
import threading
import time

app = Flask(__name__)


UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size


camera_active = False
camera_detector = None
camera_frame = None
camera_detections = None
camera_lock = threading.Lock()


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and processing"""
    try:
        
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        livestock_type = request.form.get('livestock_type', 'all')
        
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv'}), 400
        
        
        unique_id = str(uuid.uuid4())
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        input_filename = f"{unique_id}_input.{file_extension}"
        output_filename = f"{unique_id}_output.mp4"
        
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Save uploaded file
        file.save(input_path)
        
        
        detector = AnimalDetector(
            livestock_type=livestock_type 
        )
        
        summary = detector.process_video(
            video_source=input_path,
            output_path=output_path,
            display=False
        )
        
        response = {
            'success': True,
            'summary': summary,
            'output_video': output_filename,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download_video(filename):
    """Download processed video"""
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/camera/start', methods=['POST'])
def start_camera():
    """Start camera streaming"""
    global camera_active, camera_detector
    
    try:
        data = request.get_json()
        livestock_type = data.get('livestock_type', 'all')
        camera_index = int(data.get('camera_index', 0))
       
        
        if camera_active:
            return jsonify({'error': 'Camera already active'}), 400
        
        camera_detector = AnimalDetector(
            livestock_type=livestock_type
            
        )
        
        # Start camera thread
        camera_thread = threading.Thread(target=camera_worker, args=(camera_index,))
        camera_thread.daemon = True
        camera_thread.start()
        
        camera_active = True
        
        return jsonify({'success': True, 'message': 'Camera started'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera streaming"""
    global camera_active
    
    camera_active = False
    time.sleep(0.5)  
    
    return jsonify({'success': True, 'message': 'Camera stopped'})


def camera_worker(camera_index):
    """Worker function for camera processing"""
    global camera_active, camera_detector, camera_frame, camera_detections, camera_lock
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        camera_active = False
        return
    
    while camera_active:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, detections = camera_detector.process_frame(frame)
        
        # Update shared variables
        with camera_lock:
            camera_frame = processed_frame
            camera_detections = detections
    
    cap.release()


@app.route('/camera/feed')
def camera_feed():
    """Video streaming route"""
    def generate():
        global camera_frame, camera_lock
        
        while camera_active:
            with camera_lock:
                if camera_frame is not None:
                    
                    ret, buffer = cv2.imencode('.jpg', camera_frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.03) 
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera/stats')
def camera_stats():
    """Get current camera detection statistics"""
    global camera_detections, camera_lock
    
    with camera_lock:
        if camera_detections is not None:
            return jsonify({
                'active': camera_active,
                'animal_count': camera_detections['animal_count'],
                'animals_with_motion': camera_detections['animals_with_motion'],
                'bounding_boxes': camera_detections['bounding_boxes']
            })
    
    return jsonify({'active': camera_active, 'animal_count': 0, 'animals_with_motion': 0})


@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    """Clean up old uploaded and output files"""
    try:
       
        current_time = time.time()
        removed_count = 0
        
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > 3600:  
                        os.remove(file_path)
                        removed_count += 1
        
        return jsonify({
            'success': True,
            'message': f'Removed {removed_count} old files'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
 