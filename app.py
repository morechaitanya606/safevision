"""
Safety Goggles Detection - Flask Web Application
Provides a web dashboard for detecting safety goggles in images, videos,
and real-time webcam feeds using YOLO-based computer vision.
"""

import os
import uuid
import time
import json
from flask import (
    Flask, render_template, request, jsonify,
    Response, send_file, url_for
)
import cv2
import numpy as np
from detector import SafetyDetector

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(os.path.dirname(__file__), 'results')

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize detector
detector = SafetyDetector()

# Global state for webcam
webcam_active = False


@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('index.html')


@app.route('/detect/image', methods=['POST'])
def detect_image():
    """
    Handle image upload and run safety detection.
    Returns annotated image and detection results as JSON.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        return jsonify({"error": f"Unsupported format. Use: {', '.join(allowed_extensions)}"}), 400

    try:
        # Save uploaded file
        filename = f"{uuid.uuid4().hex}{ext}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        # Run detection
        annotated_image, results = detector.detect_image(upload_path)

        # Save annotated image
        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        cv2.imwrite(result_path, annotated_image)

        # Clean up upload
        os.remove(upload_path)

        # Build response
        response = {
            "success": True,
            "result_image": f"/results/{result_filename}",
            "detection": {
                "persons_detected": results["persons_detected"],
                "goggles_detected": results["goggles_detected"],
                "safety_status": results["safety_status"],
                "overall_safe": results["overall_safe"],
                "other_ppe": results["other_ppe"],
                "persons": [
                    {
                        "id": p["id"],
                        "wearing_goggles": p["wearing_goggles"],
                        "goggles_confidence": round(p["goggles_confidence"], 2),
                        "ppe_items": p["ppe_items"]
                    }
                    for p in results["persons"]
                ]
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/detect/video', methods=['POST'])
def detect_video():
    """
    Handle video upload and run safety detection on all frames.
    Returns annotated video and aggregated results.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Validate file type
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        return jsonify({"error": f"Unsupported format. Use: {', '.join(allowed_extensions)}"}), 400

    try:
        # Save uploaded file
        filename = f"{uuid.uuid4().hex}{ext}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        # Run detection on video
        result_filename = f"result_{uuid.uuid4().hex}.mp4"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)

        results = detector.detect_video(upload_path, result_path)

        # Clean up upload
        os.remove(upload_path)

        response = {
            "success": True,
            "result_video": f"/results/{result_filename}",
            "detection": results
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/detect/webcam/start', methods=['POST'])
def start_webcam():
    """Start webcam detection."""
    global webcam_active
    webcam_active = True
    return jsonify({"success": True, "message": "Webcam started"})


@app.route('/detect/webcam/stop', methods=['POST'])
def stop_webcam():
    """Stop webcam detection."""
    global webcam_active
    webcam_active = False
    return jsonify({"success": True, "message": "Webcam stopped"})


@app.route('/video_feed')
def video_feed():
    """MJPEG video stream endpoint for webcam detection."""
    return Response(
        _generate_webcam_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def _generate_webcam_stream():
    """Generate webcam frames with detection overlay."""
    global webcam_active

    try:
        for frame in detector.generate_webcam_frames(camera_index=0):
            if not webcam_active:
                break
            yield frame
    except Exception as e:
        print(f"[ERROR] Webcam stream error: {e}")


@app.route('/results/<filename>')
def serve_result(filename):
    """Serve a result file (annotated image or video)."""
    file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path)


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "detection_method": "YOLOv8 + Haar Cascades",
        "timestamp": time.time()
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  SAFETY GOGGLES DETECTION SYSTEM")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
