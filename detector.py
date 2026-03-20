"""
Safety Goggles & PPE Detection Engine
Uses YOLOv8 for person detection and OpenCV Haar cascades for glasses/goggles detection.
Supports image, video, and real-time webcam detection.
"""

import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from PIL import Image


class SafetyDetector:
    """Core detection engine for safety goggles and PPE detection."""

    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.35):
        """
        Initialize the safety detector.

        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold

        # Load YOLOv8 model for person + object detection
        print("[INFO] Loading YOLOv8 model...")
        self.model = YOLO(model_path)
        print("[INFO] YOLOv8 model loaded successfully!")

        # COCO class IDs we care about
        self.person_class_id = 0       # 'person'
        # COCO doesn't have goggles, but we can detect common objects
        # We'll use face/eye cascade approach for goggles detection

        # Load Haar cascades for face and eye detection
        print("[INFO] Loading OpenCV Haar cascades...")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.eye_glasses_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
        )
        print("[INFO] Haar cascades loaded!")

        # COCO classes for other safety-related objects
        self.safety_coco_classes = {
            # Additional objects that might appear in safety contexts
        }

        # Colors for drawing
        self.COLOR_SAFE = (0, 200, 100)      # Green
        self.COLOR_UNSAFE = (0, 80, 255)      # Red-Orange
        self.COLOR_WARNING = (0, 200, 255)    # Yellow
        self.COLOR_PPE = (255, 180, 0)        # Cyan
        self.COLOR_PERSON = (200, 150, 50)    # Light blue
        self.COLOR_FACE = (255, 200, 100)     # Light cyan
        self.COLOR_GOGGLES = (0, 255, 180)    # Bright green

    def detect_image(self, image_path):
        """
        Run safety detection on an image.

        Args:
            image_path: Path to the input image

        Returns:
            tuple: (annotated_image, results_dict)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        annotated, results = self._process_frame(image)
        return annotated, results

    def detect_image_bytes(self, image_bytes):
        """
        Run safety detection on image bytes.

        Args:
            image_bytes: Raw image bytes

        Returns:
            tuple: (annotated_image, results_dict)
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image bytes")

        annotated, results = self._process_frame(image)
        return annotated, results

    def detect_video(self, video_path, output_path, progress_callback=None):
        """
        Run safety detection on a video file.

        Args:
            video_path: Path to input video
            output_path: Path to save annotated video
            progress_callback: Optional callback(current_frame, total_frames)

        Returns:
            dict: Aggregated results across all frames
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        aggregated_results = {
            "total_frames": total_frames,
            "frames_with_persons": 0,
            "frames_with_goggles": 0,
            "frames_without_goggles": 0,
            "max_persons_in_frame": 0,
            "safety_score": 0.0
        }

        frame_count = 0
        last_annotated = None
        process_every_n = max(1, fps // 8)  # Process ~8 frames per second

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process every Nth frame for speed
            if frame_count % process_every_n == 0 or frame_count == 1:
                annotated, results = self._process_frame(frame)
                last_annotated = annotated

                # Update aggregated results
                if results["persons_detected"] > 0:
                    aggregated_results["frames_with_persons"] += 1
                    aggregated_results["max_persons_in_frame"] = max(
                        aggregated_results["max_persons_in_frame"],
                        results["persons_detected"]
                    )

                    if results["goggles_detected"] > 0:
                        aggregated_results["frames_with_goggles"] += 1
                    else:
                        aggregated_results["frames_without_goggles"] += 1
            else:
                annotated = last_annotated if last_annotated is not None else frame

            out.write(annotated)

            if progress_callback:
                progress_callback(frame_count, total_frames)

        cap.release()
        out.release()

        # Calculate overall safety score
        if aggregated_results["frames_with_persons"] > 0:
            aggregated_results["safety_score"] = round(
                (aggregated_results["frames_with_goggles"] /
                 aggregated_results["frames_with_persons"]) * 100, 1
            )

        return aggregated_results

    def generate_webcam_frames(self, camera_index=0):
        """
        Generator that yields MJPEG frames from webcam with detection overlay.

        Args:
            camera_index: Camera device index (0 = default)

        Yields:
            bytes: JPEG-encoded frame with detection annotations
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError("Could not access webcam")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        frame_count = 0
        last_annotated = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Process every 3rd frame for performance
                if frame_count % 3 == 0 or frame_count == 1:
                    annotated, results = self._process_frame(frame)
                    last_annotated = annotated
                else:
                    annotated = last_annotated if last_annotated is not None else frame

                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', annotated, [
                    cv2.IMWRITE_JPEG_QUALITY, 80
                ])
                if not ret:
                    continue

                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       frame_bytes + b'\r\n')

        finally:
            cap.release()

    def _process_frame(self, frame):
        """
        Process a single frame for safety detection.

        Strategy:
        1. Use YOLOv8 to detect persons
        2. For each person, extract the head/face region
        3. Use Haar cascades to detect face → then check for eyewear
        4. Eyewear detection logic:
           - Detect face region
           - Check for eyes WITH glasses (haarcascade_eye_tree_eyeglasses)
           - Compare with bare eyes (haarcascade_eye)
           - If glasses cascade detects more confidently → wearing goggles/glasses

        Args:
            frame: OpenCV BGR image

        Returns:
            tuple: (annotated_frame, results_dict)
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        results = {
            "persons_detected": 0,
            "goggles_detected": 0,
            "other_ppe": [],
            "persons": [],
            "safety_status": "NO_PERSON",
            "overall_safe": False
        }

        # Stage 1: Detect persons using YOLOv8
        person_detections = self.model(frame, conf=self.confidence_threshold, verbose=False)

        persons = []
        for det in person_detections:
            for box in det.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id == self.person_class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    persons.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf,
                        "wearing_goggles": False,
                        "goggles_confidence": 0.0,
                        "ppe_items": []
                    })

        results["persons_detected"] = len(persons)

        if len(persons) == 0:
            self._draw_status_bar(annotated, "No person detected", self.COLOR_WARNING, w)
            return annotated, results

        # Stage 2: For each person, detect goggles/glasses using Haar cascades
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for person in persons:
            px1, py1, px2, py2 = person["bbox"]
            person_w = px2 - px1
            person_h = py2 - py1

            # Extract head region (top 40% of person bounding box)
            head_y1 = max(0, py1 - int(person_h * 0.1))  # Extend slightly above
            head_y2 = min(h, py1 + int(person_h * 0.45))
            head_x1 = max(0, px1)
            head_x2 = min(w, px2)

            head_region = gray[head_y1:head_y2, head_x1:head_x2]

            if head_region.size == 0:
                continue

            # Detect faces in the head region
            min_face_size = max(30, int(person_w * 0.2))
            faces = self.face_cascade.detectMultiScale(
                head_region, 1.1, 5,
                minSize=(min_face_size, min_face_size)
            )

            face_found = False
            for (fx, fy, fw, fh) in faces:
                face_found = True

                # Get absolute coordinates for drawing
                abs_fx = head_x1 + fx
                abs_fy = head_y1 + fy

                # Extract face ROI
                face_roi = head_region[fy:fy + fh, fx:fx + fw]

                # Focus on the eye region (upper 60% of face)
                eye_region_h = int(fh * 0.6)
                eye_region = face_roi[0:eye_region_h, :]

                if eye_region.size == 0:
                    continue

                # Method 1: Detect eyes with glasses/goggles
                eyes_with_glasses = self.eye_glasses_cascade.detectMultiScale(
                    eye_region, 1.05, 3,
                    minSize=(int(fw * 0.15), int(fh * 0.08))
                )

                # Method 2: Detect bare eyes
                bare_eyes = self.eye_cascade.detectMultiScale(
                    eye_region, 1.1, 5,
                    minSize=(int(fw * 0.12), int(fh * 0.06))
                )

                # Decision logic for goggles detection:
                # If eye_tree_eyeglasses detects eyes BUT regular eye cascade
                # doesn't (or detects fewer), it likely means glasses/goggles are present
                wearing_eyewear = False
                confidence = 0.0

                if len(eyes_with_glasses) >= 1:
                    # Glasses cascade found eyes - likely wearing eyewear
                    if len(bare_eyes) < len(eyes_with_glasses):
                        # Bare eye detector finds fewer - strong signal for glasses
                        wearing_eyewear = True
                        confidence = 0.85
                    elif len(bare_eyes) == 0 and len(eyes_with_glasses) >= 1:
                        # No bare eyes but glasses detected
                        wearing_eyewear = True
                        confidence = 0.90
                    else:
                        # Both detect - could be glasses, moderate confidence
                        wearing_eyewear = True
                        confidence = 0.65

                    # Additional check: analyze the eye region for goggle-like features
                    # Safety goggles typically have larger, more reflective regions
                    edge_score = self._analyze_eyewear_edges(
                        frame[abs_fy:abs_fy + eye_region_h,
                              abs_fx:abs_fx + fw]
                    )
                    if edge_score > 0.5:
                        confidence = min(confidence + 0.1, 0.95)

                elif len(bare_eyes) == 0 and len(eyes_with_glasses) == 0:
                    # Can't detect any eyes - might be covered by goggles
                    # Check for reflective/structured regions in eye area
                    edge_score = self._analyze_eyewear_edges(
                        frame[abs_fy:abs_fy + eye_region_h,
                              abs_fx:abs_fx + fw]
                    )
                    if edge_score > 0.6:
                        wearing_eyewear = True
                        confidence = 0.55

                if wearing_eyewear:
                    person["wearing_goggles"] = True
                    person["goggles_confidence"] = confidence
                    results["goggles_detected"] += 1

                    # Draw goggles indicator on face
                    for (ex, ey, ew, eh) in eyes_with_glasses:
                        abs_ex = abs_fx + ex
                        abs_ey = abs_fy + ey
                        cv2.rectangle(annotated,
                                      (abs_ex, abs_ey),
                                      (abs_ex + ew, abs_ey + eh),
                                      self.COLOR_GOGGLES, 2)

                    # Draw face rectangle
                    cv2.rectangle(annotated,
                                  (abs_fx, abs_fy),
                                  (abs_fx + fw, abs_fy + fh),
                                  self.COLOR_FACE, 1)

                break  # Only process first face per person

            # If no face found, check the upper portion for any eyewear-like structure
            if not face_found and person_h > 100:
                upper_region = frame[head_y1:head_y2, head_x1:head_x2]
                edge_score = self._analyze_eyewear_edges(upper_region)
                if edge_score > 0.7:
                    person["wearing_goggles"] = True
                    person["goggles_confidence"] = 0.45
                    results["goggles_detected"] += 1

        # Draw person bounding boxes with safety status
        for i, person in enumerate(persons):
            px1, py1, px2, py2 = person["bbox"]

            if person["wearing_goggles"]:
                color = self.COLOR_SAFE
                status = f"SAFE - Goggles ({person['goggles_confidence']:.0%})"
            else:
                color = self.COLOR_UNSAFE
                status = "UNSAFE - No Goggles!"

            # Draw person box with thicker border
            cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 3)

            # Draw corner accents for modern look
            corner_len = min(25, (px2 - px1) // 4)
            # Top-left
            cv2.line(annotated, (px1, py1), (px1 + corner_len, py1), color, 4)
            cv2.line(annotated, (px1, py1), (px1, py1 + corner_len), color, 4)
            # Top-right
            cv2.line(annotated, (px2, py1), (px2 - corner_len, py1), color, 4)
            cv2.line(annotated, (px2, py1), (px2, py1 + corner_len), color, 4)
            # Bottom-left
            cv2.line(annotated, (px1, py2), (px1 + corner_len, py2), color, 4)
            cv2.line(annotated, (px1, py2), (px1, py2 - corner_len), color, 4)
            # Bottom-right
            cv2.line(annotated, (px2, py2), (px2 - corner_len, py2), color, 4)
            cv2.line(annotated, (px2, py2), (px2, py2 - corner_len), color, 4)

            # Draw status label
            label = f"Person {i+1}: {status}"
            self._draw_label(annotated, label, px1, py1 - 15, color, font_scale=0.55)

            results["persons"].append({
                "id": i + 1,
                "wearing_goggles": person["wearing_goggles"],
                "goggles_confidence": person["goggles_confidence"],
                "ppe_items": person["ppe_items"],
                "bbox": person["bbox"]
            })

        # Determine overall safety status
        persons_with_goggles = sum(1 for p in persons if p["wearing_goggles"])
        if persons_with_goggles == len(persons):
            results["safety_status"] = "ALL_SAFE"
            results["overall_safe"] = True
            status_text = f"ALL SAFE - {len(persons)} person(s) wearing goggles"
            self._draw_status_bar(annotated, status_text, self.COLOR_SAFE, w)
        elif persons_with_goggles > 0:
            results["safety_status"] = "PARTIAL"
            results["overall_safe"] = False
            status_text = f"WARNING - {persons_with_goggles}/{len(persons)} wearing goggles"
            self._draw_status_bar(annotated, status_text, self.COLOR_WARNING, w)
        else:
            results["safety_status"] = "UNSAFE"
            results["overall_safe"] = False
            status_text = f"UNSAFE - {len(persons)} person(s) WITHOUT goggles!"
            self._draw_status_bar(annotated, status_text, self.COLOR_UNSAFE, w)

        return annotated, results

    def _analyze_eyewear_edges(self, region):
        """
        Analyze a region for eyewear-like features using edge detection.
        Safety goggles typically have strong horizontal edges and structured shapes.

        Args:
            region: BGR image region (eye area)

        Returns:
            float: Score 0-1 indicating likelihood of eyewear
        """
        if region is None or region.size == 0:
            return 0.0

        try:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
            h, w = gray.shape[:2]

            if h < 10 or w < 10:
                return 0.0

            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Calculate edge density
            edge_density = np.sum(edges > 0) / (h * w)

            # Check for horizontal lines (goggles frames)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20,
                                     minLineLength=w // 4, maxLineGap=10)

            line_score = 0.0
            if lines is not None:
                horizontal_lines = 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if angle < 30 or angle > 150:  # Near-horizontal
                        horizontal_lines += 1
                line_score = min(horizontal_lines / 3.0, 1.0)

            # Check for circular/elliptical shapes (goggle lenses)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                                        param1=50, param2=30,
                                        minRadius=w // 8, maxRadius=w // 2)
            circle_score = 0.3 if circles is not None else 0.0

            # Combine scores
            score = (edge_density * 2 + line_score * 0.5 + circle_score) / 3.0
            return min(score, 1.0)

        except Exception:
            return 0.0

    def _draw_label(self, image, text, x, y, color, font_scale=0.55):
        """Draw a text label with background on the image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Ensure label stays within image bounds
        y = max(text_h + baseline + 10, y)
        x = max(0, x)

        # Background rectangle with slight transparency
        overlay = image.copy()
        cv2.rectangle(overlay,
                      (x, y - text_h - baseline - 8),
                      (x + text_w + 10, y + 4),
                      color, -1)
        cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)

        # Text (dark for contrast)
        cv2.putText(image, text, (x + 5, y - 4),
                    font, font_scale, (0, 0, 0), thickness)

    def _draw_status_bar(self, image, text, color, width):
        """Draw a status bar at the top of the image."""
        bar_height = 50
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, bar_height), color, -1)
        cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        text_x = (width - text_w) // 2
        text_y = (bar_height + text_h) // 2

        # Draw text with outline for better readability
        cv2.putText(image, text, (text_x, text_y),
                    font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(image, text, (text_x, text_y),
                    font, font_scale, (255, 255, 255), thickness)
