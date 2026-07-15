import sys
import os
import json
import time
import base64
import random
import threading
import cv2 as cv
import numpy as np
# from huggingface_hub import hf_hub_download

# Print to stderr for debugging so stdout remains clean JSON
def log(msg):
    sys.stderr.write(f"[Python Worker] {msg}\n")
    sys.stderr.flush()

class YuNet:
    def __init__(self, modelPath: str, confThreshold: float = 0.5):
        # Default input size, will be set dynamically per frame
        self._model = cv.FaceDetectorYN.create(
            modelPath, "", (320, 320), confThreshold, 0.3, 5000, 0, 0
        )

    def setInputSize(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        faces = self._model.detect(image)
        return np.empty((0, 15), dtype=np.float32) if faces[1] is None else faces[1]

class SFace:
    def __init__(self, modelPath: str, disType: int = 0):
        self._model = cv.FaceRecognizerSF.create(modelPath, "", 0, 0)
        self._disType = disType  # 0 cosine, 1 norml2

    def infer(self, image, face_bbox_landmarks_etc):
        try:
            if image is None or image.size == 0:
                return None
            if not np.isfinite(face_bbox_landmarks_etc).all():
                return None
            aligned = self._model.alignCrop(image, face_bbox_landmarks_etc.astype(np.float32))
            if aligned is None or aligned.size == 0 or aligned.shape[0] == 0 or aligned.shape[1] == 0:
                return None
            feat = self._model.feature(aligned)
            if feat is None or feat.size == 0:
                return None
            return feat
        except Exception as e:
            log(f"Error in SFace.infer: {str(e)}")
            return None

    def score(self, feat1, feat2) -> float:
        if feat1 is None or feat2 is None:
            return 0.0
        try:
            return float(self._model.match(feat1, feat2, self._disType))
        except Exception as e:
            log(f"Error in SFace.score: {str(e)}")
            return 0.0

# Global variables for model paths and main thread instances
yunet_path = ""
sface_path = ""
detector = None
recog = None
gender_path = ""
gender_net = None

candidates = []
candidates_lock = threading.Lock()

# Multi-camera threads and stop events dictionaries
stream_threads = {}
stream_stop_events = {}
streams_lock = threading.Lock()

def load_models():
    global detector, recog, yunet_path, sface_path, gender_net, gender_path

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")

    # ── Expected local filenames ──────────────────────────────────────────
    # Place these files in <detectors>/models/  (or <detectors>/ directly)
    #   face_detection_yunet_2023mar.onnx
    #   face_recognition_sface_2021dec.onnx
    #   gender.onnx
    YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
    SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"
    GENDER_FILENAME = "gender.onnx"

    def find_local(filename):
        """Look in detectors/models/ then detectors/ directly."""
        for d in (models_dir, base_dir):
            candidate = os.path.join(d, filename)
            if os.path.exists(candidate):
                return candidate
        return None

    try:
        # ── YuNet face detector ─────────────────────────────────────────────
        yunet_path = find_local(YUNET_FILENAME)
        if yunet_path:
            log(f"Loading local YuNet model: {yunet_path}")
        # elif hf_hub_download is not None:
        #     log("Local YuNet model not found — downloading from HuggingFace...")
        #     yunet_path = hf_hub_download("opencv/face_detection_yunet", YUNET_FILENAME)
        else:
            raise FileNotFoundError(
                f"YuNet model not found. Place '{YUNET_FILENAME}' in "
                f"'{models_dir}' or '{base_dir}', or install huggingface_hub."
            )

        # ── SFace face recognizer ───────────────────────────────────────────
        sface_path = find_local(SFACE_FILENAME)
        if sface_path:
            log(f"Loading local SFace model: {sface_path}")
        # elif hf_hub_download is not None:
        #     log("Local SFace model not found — downloading from HuggingFace...")
        #     sface_path = hf_hub_download("opencv/face_recognition_sface", SFACE_FILENAME)
        else:
            raise FileNotFoundError(
                f"SFace model not found. Place '{SFACE_FILENAME}' in "
                f"'{models_dir}' or '{base_dir}', or install huggingface_hub."
            )

        log("Loading main thread instances...")
        detector = YuNet(yunet_path)
        recog = SFace(sface_path)

        # ── Gender classification model (always local-only) ─────────────────
        gender_path = find_local(GENDER_FILENAME)
        if gender_path:
            log(f"Loading gender model: {gender_path}")
            gender_net = cv.dnn.readNetFromONNX(gender_path)
        else:
            log(f"WARNING: {GENDER_FILENAME} not found in '{models_dir}' or '{base_dir}' "
                f"— gender_classification capability will be unavailable.")

        log("Models loaded in memory.")
    except Exception as e:
        log(f"Error loading models: {str(e)}")
        sys.exit(1)

def classify_gender(net, img, box):
    if net is None or img is None or img.size == 0:
        return None
    try:
        h_img, w_img = img.shape[:2]
        x, y, w, h = box.astype(int)
        
        # Ensure coordinates are within image boundaries
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
            
        # Resize to 640x640, scale by 1/255.0, swap BGR to RGB
        blob = cv.dnn.blobFromImage(crop, 1.0 / 255.0, (640, 640), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        preds = net.forward()
        
        idx = np.argmax(preds[0])
        return "Female" if idx == 0 else "Male"
    except Exception as e:
        log(f"Error in classify_gender: {str(e)}")
        return None

def extract_best_face_embedding(img_path):
    img = cv.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    detector.setInputSize((img.shape[1], img.shape[0]))
    faces = detector.infer(img)
    if faces.shape[0] == 0:
        return None
    
    # Get the largest face by bounding box area (w * h) to avoid enrolling background faces
    areas = faces[:, 2] * faces[:, 3]
    best_idx = np.argmax(areas)
    best_face = faces[best_idx]
    
    # Enforce quality checks for enrollment templates
    x, y, w, h = best_face[:4].astype(int)
    conf = best_face[-1]
    
    if conf < 0.80:
        raise ValueError(f"Face detection confidence is too low ({conf:.2f} < 0.80). Use a clearer photo.")
    if w < 80 or h < 80:
        raise ValueError(f"Face is too small ({w}x{h} < 80x80 pixels). Use a closer photo of the face.")
        
    feat = recog.infer(img, best_face[:-1])
    if feat is None:
        raise ValueError("Could not extract face embedding feature.")
    return feat.flatten().tolist()

def compare_face_features(feat, threshold, dis_type, thread_recog):
    with candidates_lock:
        local_candidates = list(candidates)
        
    if not local_candidates:
        return None, -1.0
        
    cand_scores = []
    for cand in local_candidates:
        cand_id = cand.get("person_id")
        cand_name = cand.get("name")
        embs = cand.get("embeddings", [])
        
        if not embs:
            continue
            
        scores = []
        for emb_list in embs:
            emb_arr = np.array(emb_list, dtype=np.float32).reshape(1, -1)
            score = thread_recog.score(feat, emb_arr)
            scores.append(score)
            
        # Sort scores: Cosine -> descending (highest first); L2 -> ascending (lowest first)
        scores.sort(reverse=(dis_type == 0))
        
        # Use nearest-neighbor match (highest score among all enrolled templates of this person)
        best_cand_score = scores[0]
            
        cand_scores.append({
            "person_id": cand_id,
            "name": cand_name,
            "score": best_cand_score
        })
        
    if not cand_scores:
        return None, -1.0
        
    # Sort candidates by score
    if dis_type == 0: # Cosine: higher score first
        cand_scores.sort(key=lambda x: x["score"], reverse=True)
    else: # L2: lower score first
        cand_scores.sort(key=lambda x: x["score"])
        
    best_cand = cand_scores[0]
    best_score = best_cand["score"]
    
    is_match = False
    if dis_type == 0: # Cosine
        is_match = best_score >= threshold
    else: # L2
        is_match = best_score <= threshold
                    
    if is_match:
        return {"person_id": best_cand["person_id"], "name": best_cand["name"]}, best_score
    else:
        return None, best_score


def crop_and_save_face(img, box, crops_dir):
    h_img, w_img = img.shape[:2]
    x, y, w, h = box.astype(int)
    
    # 60% margin padding to show clear face with surrounding head/shoulders context
    pad_w = int(w * 0.60)
    pad_h = int(h * 0.60)
    
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w_img, x + w + pad_w)
    y2 = min(h_img, y + h + pad_h)
    
    if x2 > x1 and y2 > y1:
        crop = img[y1:y2, x1:x2]
        crop_filename = f"crop_{int(time.time())}_{random.randint(1000, 9999)}.jpg"
        crop_path = os.path.join(crops_dir, crop_filename)
        cv.imwrite(crop_path, crop)
        return crop_filename
    return None

def process_video_for_enrollment(video_path, crops_dir):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    fps = cap.get(cv.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Fallback
        
    # Process every frame of the video to capture all available faces
    frame_interval = 1
        
    frame_count = 0
    accepted_faces = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            detector.setInputSize((frame.shape[1], frame.shape[0]))
            faces = detector.infer(frame)
            
            if faces is not None and faces.shape[0] > 0:
                # Find the largest face bounding box
                areas = faces[:, 2] * faces[:, 3]
                best_idx = np.argmax(areas)
                best_face = faces[best_idx]
                
                x, y, w, h = best_face[:4].astype(int)
                conf = best_face[-1]
                
                # Minimum face size of 80x80 and detection confidence >= 0.90
                if conf >= 0.90 and w >= 80 and h >= 80:
                    feat = recog.infer(frame, best_face[:-1])
                    if feat is not None:
                        # Use existing padded cropping logic
                        crop_filename = crop_and_save_face(frame, best_face[:4], crops_dir)
                        if crop_filename:
                            accepted_faces.append({
                                "filename": crop_filename,
                                "embedding": feat.flatten().tolist()
                            })
                            
                        # Cap total enrollment count per video scan to 500 templates
                        if len(accepted_faces) >= 500:
                            break
                                
        frame_count += 1
        
    cap.release()
    return accepted_faces

def process_single_image(img_path, threshold, dis_type, crops_dir):
    img = cv.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
        
    detector.setInputSize((img.shape[1], img.shape[0]))
    faces = detector.infer(img)
    
    results = []
    for f in faces:
        box = f[:4]
        conf = f[-1]
        
        is_known = False
        match = None
        score = 0.0
        feat = None
        
        # Only run SFace recognition on high-confidence face detections to prevent false matches
        if conf >= 0.80:
            feat = recog.infer(img, f[:-1])
            if feat is not None:
                match, score = compare_face_features(feat, threshold, dis_type, recog)
                is_known = match is not None
            
        crop_filename = crop_and_save_face(img, box, crops_dir)
        
        gender = None
        if gender_net is not None:
            gender = classify_gender(gender_net, img, box)
            
        results.append({
            "box": box.astype(int).tolist(),
            "score": score,
            "match": match,
            "is_known": is_known,
            "crop_filename": crop_filename,
            "embedding": feat.flatten().tolist() if feat is not None else None,
            "gender": gender
        })
        
    return results

class VideoGrabber(threading.Thread):
    def __init__(self, rtsp_url, camera_id):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.cap = None
        self.running = True
        self.latest_frame = None
        self.need_frame = True  # Start with True so we get the first frame
        self.frame_lock = threading.Lock()
        self.frame_event = threading.Event() # Set when a new frame is decoded
        self.daemon = True

    def run(self):
        log(f"[{self.camera_id}] Connecting to RTSP URL...")
        try:
            self.cap = cv.VideoCapture(self.rtsp_url, cv.CAP_FFMPEG)
            if self.cap is not None:
                self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            log(f"[{self.camera_id}] VideoCapture init exception: {str(e)}")
            self.cap = None
        
        reconnect_delay = 1.0
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    log(f"[{self.camera_id}] Connection offline. Reconnecting in {reconnect_delay}s...")
                    if self.cap is not None:
                        try:
                            self.cap.release()
                        except:
                            pass
                        self.cap = None
                    time.sleep(reconnect_delay)
                    if not self.running:
                        break
                    self.cap = cv.VideoCapture(self.rtsp_url, cv.CAP_FFMPEG)
                    if self.cap is not None:
                        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                    reconnect_delay = min(reconnect_delay * 2, 10.0)
                    continue
                    
                reconnect_delay = 1.0
                
                # Grab the frame packet (very fast, no H.264 decoding)
                ok = self.cap.grab()
                if not ok:
                    log(f"[{self.camera_id}] Frame grab error. Resetting connection...")
                    if self.cap is not None:
                        self.cap.release()
                        self.cap = None
                    continue
                
                # Decode (retrieve) the frame ONLY if requested by the processor thread
                if self.need_frame:
                    ok, frame = self.cap.retrieve()
                    if ok and frame is not None:
                        with self.frame_lock:
                            self.latest_frame = frame
                        self.frame_event.set() # Notify that a new frame is ready
                        self.need_frame = False
            except Exception as e:
                log(f"[{self.camera_id}] Exception in VideoGrabber thread loop: {str(e)}")
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None
                time.sleep(1.0)
                
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        log(f"[{self.camera_id}] VideoGrabber thread finished.")

    def get_frame(self, timeout=0.2):
        # Wait up to 200ms for a new frame to be decoded
        got_new = self.frame_event.wait(timeout)
        if got_new:
            self.frame_event.clear()
            with self.frame_lock:
                frame = self.latest_frame
            self.need_frame = True # Request next frame decode
            return frame
        else:
            # If timeout, return the latest frame to keep processor running
            with self.frame_lock:
                return self.latest_frame

    def stop(self):
        self.running = False

def rtsp_stream_processor(camera_id, camera_name, rtsp_url, threshold, dis_type, crops_dir, 
                          line_crossing_enabled, line_y, line_direction, line_x_start, line_x_end, stop_event):
    log(f"[{camera_id}] Starting camera stream thread: {camera_name} (Line Crossing: {line_crossing_enabled}, Y: {line_y}, Dir: {line_direction}, X: {line_x_start}-{line_x_end})")
    
    # Separate thread-local models to prevent race conditions on inference
    thread_detector = YuNet(yunet_path)
    thread_recog = SFace(sface_path, disType=dis_type)
    
    # Load thread-local gender model
    thread_gender_net = None
    if os.path.exists(gender_path):
        thread_gender_net = cv.dnn.readNetFromONNX(gender_path)
        
    grabber = VideoGrabber(rtsp_url, camera_id)
    grabber.start()
    
    last_event_time = {}
    det_width = 1024
    
    # Face tracking state
    tracked_faces = []
    next_track_id = 0
    
    last_frame_time = 0.0
    
    try:
        while not stop_event.is_set():
            frame = grabber.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
                
            now = time.time()
            dt = now - last_frame_time if last_frame_time > 0.0 else 0.2
            last_frame_time = now
            
            h_img, w_img = frame.shape[:2]
            
            # Optimization: Resize frame to 640px width for YuNet face detection (reduces CPU by up to 90% for 1080p feeds)
            if w_img > det_width:
                scale = det_width / float(w_img)
                det_h = int(h_img * scale)
                det_frame = cv.resize(frame, (det_width, det_h))
            else:
                scale = 1.0
                det_frame = frame
                det_h, det_width = h_img, w_img
                
            thread_detector.setInputSize((det_width, det_h))
            faces = thread_detector.infer(det_frame)
            
            detected_faces_data = []
            now = time.time()
            
            if line_crossing_enabled:
                # ---------------------------------------------------------
                # Line Crossing Face Detection & Tracking Mode
                # ---------------------------------------------------------
                current_tracked = []
                matched_ids = set()
                y_line = h_img * line_y
                x_start = w_img * line_x_start
                x_end = w_img * line_x_end
                
                for f in faces:
                    f_orig = f.copy()
                    if scale != 1.0:
                        f_orig[:14] = f_orig[:14] / scale
                        
                    box = f_orig[:4]
                    x, y, w, h = box.astype(int)
                    conf = f_orig[-1]
                    
                    if w < 15 or h < 15:
                        continue
                        
                    cx = x + w / 2
                    cy = y + h / 2
                    
                    best_match = None
                    best_dist = float('inf')
                    # Scale search radius dynamically based on frame-step time (dt) to prevent track splits on slow hardware
                    base_max_dist = w_img * 0.08
                    max_dist = min(base_max_dist * (dt / 0.2), w_img * 0.20)
                    
                    for tf in tracked_faces:
                        if tf["id"] in matched_ids:
                            continue
                        # If a track has already crossed and exited the line, do not match it with a new face above the line
                        if tf["crossed"] and cy <= y_line:
                            continue
                        tx, ty = tf["last_center"]
                        # Prevent hijacking: if track is crossed, don't match with a face behind the track's movement direction
                        if tf["crossed"] and len(tf["ys"]) >= 2:
                            track_dir = tf["ys"][-1] - tf["ys"][0]
                            if track_dir > 0 and cy < ty: # Moving down, face is above track
                                continue
                            elif track_dir < 0 and cy > ty: # Moving up, face is below track
                                continue
                        dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)
                        if dist < max_dist and dist < best_dist:
                            best_dist = dist
                            best_match = tf
                            
                    if best_match is not None:
                        matched_ids.add(best_match["id"])
                        prev_cy = best_match["last_center"][1]
                        best_match["last_bbox"] = [x, y, w, h]
                        best_match["last_center"] = (cx, cy)
                        best_match["last_seen"] = now
                        best_match["ys"].append(cy)
                        if len(best_match["ys"]) > 10:
                            best_match["ys"].pop(0)
                            
                        crossed_trigger = False
                        if not best_match["crossed"]:
                            # Check if face trajectory came from the other side of the line
                            has_started_other_side = False
                            if line_direction == 'in' or line_direction == 'both':
                                if cy > y_line:
                                    # 1. Standard crossing: previous coordinate was above line
                                    if any(y <= y_line for y in best_match["ys"][:-1]):
                                        has_started_other_side = True
                                    # 2. Low-FPS recovery: track started just below the line and is moving down
                                    elif len(best_match["ys"]) >= 2 and best_match["ys"][-1] > best_match["ys"][0] and best_match["ys"][0] < (y_line + h_img * 0.15):
                                        has_started_other_side = True
                            if line_direction == 'out' or line_direction == 'both':
                                if cy < y_line:
                                    # 1. Standard crossing: previous coordinate was below line
                                    if any(y >= y_line for y in best_match["ys"][:-1]):
                                        has_started_other_side = True
                                    # 2. Low-FPS recovery: track started just above the line and is moving up
                                    elif len(best_match["ys"]) >= 2 and best_match["ys"][-1] < best_match["ys"][0] and best_match["ys"][0] > (y_line - h_img * 0.15):
                                        has_started_other_side = True
                                        
                            if has_started_other_side and (x_start <= cx <= x_end):
                                crossed_trigger = True
                                    
                        if crossed_trigger:
                            # Try recognizing if confidence is sufficient (>= 0.60)
                            if conf >= 0.55:
                                best_match["crossed"] = True
                                log(f"[{camera_id}] Track #{best_match['id']} crossed line Y={int(y_line)} (X span: {int(x_start)}-{int(x_end)}) in direction: {line_direction}")
                                
                                is_known = False
                                match = None
                                score = 0.0
                                feat = None
                                
                                feat = thread_recog.infer(frame, f_orig[:-1])
                                if feat is not None:
                                    match, score = compare_face_features(feat, threshold, dis_type, thread_recog)
                                    is_known = match is not None
                                    
                                    crop_filename = crop_and_save_face(frame, box, crops_dir)
                                    gender = classify_gender(thread_gender_net, frame, box) if thread_gender_net is not None else None
                                    
                                    detected_faces_data.append({
                                        "box": [int(x), int(y), int(w), int(h)],
                                        "score": score,
                                        "match": match,
                                        "is_known": is_known,
                                        "crop_filename": crop_filename,
                                        "embedding": feat.flatten().tolist() if feat is not None else None,
                                        "gender": gender
                                    })
                        current_tracked.append(best_match)
                    else:
                        new_tf = {
                            "id": next_track_id,
                            "last_bbox": [x, y, w, h],
                            "last_center": (cx, cy),
                            "crossed": False,
                            "last_seen": now,
                            "ys": [cy]
                        }
                        next_track_id += 1
                        current_tracked.append(new_tf)
                        
                # Preserve unmatched tracks for a short period (0.3 seconds) to handle brief dropouts
                for tf in tracked_faces:
                    if tf["id"] not in matched_ids:
                        if now - tf["last_seen"] < 0.3:
                            current_tracked.append(tf)
                            
                tracked_faces = current_tracked
            else:
                # ---------------------------------------------------------
                # Standard Mode (Full Frame Detection on all frames)
                # ---------------------------------------------------------
                for f in faces:
                    f_orig = f.copy()
                    if scale != 1.0:
                        f_orig[:14] = f_orig[:14] / scale
                        
                    box = f_orig[:4]
                    x, y, w, h = box.astype(int)
                    conf = f_orig[-1]
                    
                    if w < 15 or h < 15:
                        continue
                    
                    is_known = False
                    match = None
                    score = 0.0
                    feat = None
                    
                    if conf >= 0.60:
                        feat = thread_recog.infer(frame, f_orig[:-1])
                        if feat is not None:
                            match, score = compare_face_features(feat, threshold, dis_type, thread_recog)
                            is_known = match is not None
                    
                    person_key = match["person_id"] if is_known else "UNKNOWN"
                    cooldown = 3.0
                    
                    if is_known or conf >= 0.80:
                        if person_key not in last_event_time or (now - last_event_time[person_key]) > cooldown:
                            last_event_time[person_key] = now
                            
                            crop_filename = crop_and_save_face(frame, box, crops_dir)
                            gender = classify_gender(thread_gender_net, frame, box) if thread_gender_net is not None else None
                                
                            detected_faces_data.append({
                                "box": [int(x), int(y), int(w), int(h)],
                                "score": score,
                                "match": match,
                                "is_known": is_known,
                                "crop_filename": crop_filename,
                                "embedding": feat.flatten().tolist() if feat is not None else None,
                                "gender": gender
                            })
            
            # Emit recognition match events for specific camera
            if detected_faces_data:
                sys.stdout.write(json.dumps({
                    "event": "stream_match",
                    "camera_id": camera_id,
                    "camera_name": camera_name,
                    # Original-frame dimensions so the frontend can normalize
                    # box coordinates and overlay them correctly on any video size,
                    # and so it can position a line-crossing overlay consistently.
                    "frame_width": int(w_img),
                    "frame_height": int(h_img),
                    "line_crossing": {
                        "enabled": bool(line_crossing_enabled),
                        "y": float(line_y),
                        "direction": line_direction,
                        "x_start": float(line_x_start),
                        "x_end": float(line_x_end),
                    },
                    "faces": detected_faces_data
                }) + "\n")
                sys.stdout.flush()
                
            # Regulate thread loop rate adaptively to conserve CPU on ARM64 while preventing frame skips
            if len(tracked_faces) > 0:
                # High-speed tracking mode: sleep minimal time to capture every frame
                time.sleep(0.01)
            else:
                # Low-power standby mode: sleep 0.15s (~6.6 FPS) to save CPU when scene is empty
                time.sleep(0.15)
            
    except Exception as e:
        log(f"[{camera_id}] Error in processor loop: {str(e)}")
    finally:
        grabber.stop()
        log(f"[{camera_id}] Camera stream thread terminated.")

def main():
    global candidates, stream_threads, stream_stop_events
    load_models()
    
    sys.stdout.write(json.dumps({"event": "ready"}) + "\n")
    sys.stdout.flush()
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
                
            req = json.loads(line.strip())
            cmd = req.get("cmd")
            
            if cmd == "extract_embedding":
                img_path = req.get("img_path")
                try:
                    emb = extract_best_face_embedding(img_path)
                    if emb:
                        res = {"status": "success", "embedding": emb}
                    else:
                        res = {"status": "error", "message": "No face detected in image."}
                except Exception as e:
                    res = {"status": "error", "message": str(e)}
                sys.stdout.write(json.dumps({"cmd": "extract_embedding", "response": res}) + "\n")
                sys.stdout.flush()
                
            elif cmd == "process_video_enrollment":
                video_path = req.get("video_path")
                crops_dir = req.get("crops_dir")
                try:
                    faces = process_video_for_enrollment(video_path, crops_dir)
                    res = {"status": "success", "faces": faces}
                except Exception as e:
                    res = {"status": "error", "message": str(e)}
                sys.stdout.write(json.dumps({"cmd": "process_video_enrollment", "response": res}) + "\n")
                sys.stdout.flush()
                
            elif cmd == "recognize_image":
                img_path = req.get("img_path")
                local_candidates = req.get("candidates", [])
                threshold = req.get("threshold", 0.60)
                dis_type = req.get("dis_type", 0)
                crops_dir = req.get("crops_dir", ".")
                
                with candidates_lock:
                    candidates = local_candidates
                    
                try:
                    faces = process_single_image(img_path, threshold, dis_type, crops_dir)
                    res = {"status": "success", "faces": faces}
                except Exception as e:
                    res = {"status": "error", "message": str(e)}
                sys.stdout.write(json.dumps({"cmd": "recognize_image", "response": res}) + "\n")
                sys.stdout.flush()
                
            elif cmd == "start_stream":
                camera_id = req.get("camera_id")
                camera_name = req.get("camera_name", "Unnamed Camera")
                rtsp_url = req.get("rtsp_url")
                local_candidates = req.get("candidates", [])
                threshold = req.get("threshold", 0.60)
                dis_type = req.get("dis_type", 0)
                crops_dir = req.get("crops_dir", ".")
                line_crossing_enabled = req.get("line_crossing_enabled", False)
                line_y = req.get("line_y", 0.6)
                line_direction = req.get("line_direction", "in")
                line_x_start = req.get("line_x_start", 0.0)
                line_x_end = req.get("line_x_end", 1.0)
                
                with candidates_lock:
                    candidates = local_candidates
                
                with streams_lock:
                    # Stop stream if running
                    if camera_id in stream_threads:
                        log(f"Restarting camera {camera_id}...")
                        stream_stop_events[camera_id].set()
                        stream_threads[camera_id].join(timeout=2.0)
                        
                    stop_event = threading.Event()
                    stream_stop_events[camera_id] = stop_event
                    
                    thread = threading.Thread(
                        target=rtsp_stream_processor,
                        args=(camera_id, camera_name, rtsp_url, threshold, dis_type, crops_dir, 
                              line_crossing_enabled, line_y, line_direction, line_x_start, line_x_end, stop_event),
                        daemon=True
                    )
                    stream_threads[camera_id] = thread
                    thread.start()
                
                res = {"status": "success", "message": f"Camera stream thread {camera_id} started."}
                sys.stdout.write(json.dumps({"cmd": "start_stream", "camera_id": camera_id, "response": res}) + "\n")
                sys.stdout.flush()
                
            elif cmd == "stop_stream":
                camera_id = req.get("camera_id")
                
                with streams_lock:
                    if camera_id in stream_threads:
                        stream_stop_events[camera_id].set()
                        
                        # Join in background to not block the main reading thread
                        def cleanup_thread(cid):
                            if cid in stream_threads:
                                stream_threads[cid].join(timeout=2.0)
                                del stream_threads[cid]
                                if cid in stream_stop_events:
                                    del stream_stop_events[cid]
                        
                        threading.Thread(target=cleanup_thread, args=(camera_id,), daemon=True).start()
                        res = {"status": "success", "message": f"Camera {camera_id} stopped."}
                    else:
                        res = {"status": "success", "message": "Camera stream was not running."}
                        
                sys.stdout.write(json.dumps({"cmd": "stop_stream", "camera_id": camera_id, "response": res}) + "\n")
                sys.stdout.flush()
                
            elif cmd == "update_candidates":
                local_candidates = req.get("candidates", [])
                with candidates_lock:
                    candidates = local_candidates
                res = {"status": "success", "message": "Candidates synced across all stream threads."}
                sys.stdout.write(json.dumps({"cmd": "update_candidates", "response": res}) + "\n")
                sys.stdout.flush()
                
            elif cmd == "add_template":
                person_id = req.get("person_id")
                embedding = req.get("embedding")
                with candidates_lock:
                    found = False
                    for cand in candidates:
                        if cand.get("person_id") == person_id:
                            if "embeddings" not in cand:
                                cand["embeddings"] = []
                            cand["embeddings"].append(embedding)
                            found = True
                            break
                    if not found:
                        candidates.append({
                            "person_id": person_id,
                            "name": req.get("name", "Unknown"),
                            "embeddings": [embedding]
                        })
                res = {"status": "success", "message": "Embedding appended successfully."}
                sys.stdout.write(json.dumps({"cmd": "add_template", "response": res}) + "\n")
                sys.stdout.flush()
                
            else:
                res = {"status": "error", "message": f"Unknown command: {cmd}"}
                sys.stdout.write(json.dumps({"cmd": cmd, "response": res}) + "\n")
                sys.stdout.flush()
                
        except Exception as e:
            log(f"Error reading/processing command: {str(e)}")
            res = {"status": "error", "message": str(e)}
            sys.stdout.write(json.dumps({"response": res}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
