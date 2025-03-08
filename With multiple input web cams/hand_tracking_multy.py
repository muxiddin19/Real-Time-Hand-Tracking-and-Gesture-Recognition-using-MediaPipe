import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os
import threading
import time
import sys
from collections import deque

# Initialize MediaPipe Hands solutions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define virtual keyboard layout and functions
class VirtualKeyboard:
    def __init__(self):
        # Define keyboard layout
        self.keys = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/'],
            ['SHIFT', 'SPACE', 'BACKSPACE', 'ENTER']
        ]
        
        # Key states and properties
        self.key_status = {}  # To track which keys are activated
        self.key_press_time = {}  # To track how long a key has been pressed
        self.key_cooldown = {}  # To prevent rapid repeated activations
        self.hover_key = None  # Currently hovered key
        
        # Initialize all keys to inactive
        for row in self.keys:
            for key in row:
                self.key_status[key] = False
                self.key_press_time[key] = 0
                self.key_cooldown[key] = 0
                
        # Text display
        self.typed_text = ""
        self.max_text_length = 100
        
        # Key press time threshold (seconds)
        self.key_press_threshold = 0.6
        self.cooldown_time = 0.5
        
    def draw_keyboard(self, frame, hand_landmarks=None, keyboard_region=(20, 300, 600, 450)):
        """Draw the virtual keyboard on the frame and handle hover interactions"""
        x, y, w, h = keyboard_region
        rows = len(self.keys)
        
        # Create keyboard background
        keyboard_overlay = frame.copy()
        cv2.rectangle(keyboard_overlay, (x, y), (x + w, y + h), (30, 30, 30), -1)
        
        # Transparent overlay
        alpha = 0.7
        cv2.addWeighted(keyboard_overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Calculate finger tip position if landmarks are provided
        index_finger_tip = None
        if hand_landmarks:
            # Convert landmarks to pixel coordinates
            h_frame, w_frame, _ = frame.shape
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            if index_finger.x >= 0 and index_finger.y >= 0:  # Check if valid
                index_finger_tip = (int(index_finger.x * w_frame), int(index_finger.y * h_frame))
        
        # Reset hover status
        self.hover_key = None
        
        # Draw each key row
        for i, row in enumerate(self.keys):
            # Special handling for last row (function keys)
            if i == rows - 1:
                # Function keys row has fewer but wider keys
                key_width = w // len(row)
                for j, key in enumerate(row):
                    key_x = x + j * key_width
                    key_y = y + i * (h // rows)
                    key_w = key_width
                    key_h = h // rows
                    
                    # Check if finger is hovering over this key
                    if index_finger_tip and key_x <= index_finger_tip[0] <= key_x + key_w and key_y <= index_finger_tip[1] <= key_y + key_h:
                        self.hover_key = key
                        
                    # Determine key color based on state
                    if self.key_status[key]:
                        key_color = (80, 170, 80)  # Green for active
                    elif key == self.hover_key:
                        key_color = (80, 80, 170)  # Blue for hover
                    else:
                        key_color = (60, 60, 60)  # Gray for inactive
                    
                    # Draw key
                    cv2.rectangle(frame, (key_x, key_y), (key_x + key_w, key_y + key_h), key_color, -1)
                    cv2.rectangle(frame, (key_x, key_y), (key_x + key_w, key_y + key_h), (200, 200, 200), 1)
                    
                    # Center text on key
                    text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    text_x = key_x + (key_w - text_size[0]) // 2
                    text_y = key_y + (key_h + text_size[1]) // 2
                    cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
            else:
                # Regular letter keys
                key_width = w // len(row)
                for j, key in enumerate(row):
                    key_x = x + j * key_width
                    key_y = y + i * (h // rows)
                    key_w = key_width
                    key_h = h // rows
                    
                    # Check if finger is hovering over this key
                    if index_finger_tip and key_x <= index_finger_tip[0] <= key_x + key_w and key_y <= index_finger_tip[1] <= key_y + key_h:
                        self.hover_key = key
                        
                    # Determine key color based on state
                    if self.key_status[key]:
                        key_color = (80, 170, 80)  # Green for active
                    elif key == self.hover_key:
                        key_color = (80, 80, 170)  # Blue for hover
                    else:
                        key_color = (60, 60, 60)  # Gray for inactive
                    
                    # Draw key
                    cv2.rectangle(frame, (key_x, key_y), (key_x + key_w, key_y + key_h), key_color, -1)
                    cv2.rectangle(frame, (key_x, key_y), (key_x + key_w, key_y + key_h), (200, 200, 200), 1)
                    
                    # Center text on key
                    text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                    text_x = key_x + (key_w - text_size[0]) // 2
                    text_y = key_y + (key_h + text_size[1]) // 2
                    cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw text display area
        text_display_y = y - 60
        cv2.rectangle(frame, (x, text_display_y - 40), (x + w, text_display_y + 20), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, text_display_y - 40), (x + w, text_display_y + 20), (200, 200, 200), 1)
        
        # Draw cursor and make sure text fits the display
        display_text = self.typed_text[-40:] if len(self.typed_text) > 40 else self.typed_text
        if time.time() % 1 > 0.5:  # Blinking cursor
            display_text += "|"
        cv2.putText(frame, display_text, (x + 10, text_display_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 1)
                    
        # Return the key currently being hovered
        return self.hover_key
        
    def update_key_states(self, key, is_pressed):
        """Update key states and handle key press logic"""
        now = time.time()
        
        # If the key is in cooldown, ignore this press
        if now < self.key_cooldown.get(key, 0):
            return False
            
        # Key press started
        if is_pressed and not self.key_status.get(key, False):
            self.key_status[key] = True
            self.key_press_time[key] = now
            return False  # Not activated yet, just started pressing
            
        # Key is being held down
        elif is_pressed and self.key_status.get(key, False):
            # Check if key has been pressed long enough to register
            if now - self.key_press_time.get(key, now) >= self.key_press_threshold:
                self.key_status[key] = False  # Reset key state
                self.key_cooldown[key] = now + self.cooldown_time  # Set cooldown
                return True  # Key activated
            return False  # Still pressing but not long enough
            
        # Key was released before activation threshold
        elif not is_pressed and self.key_status.get(key, False):
            self.key_status[key] = False
            return False
            
        return False

    def process_key_action(self, key):
        """Handle key press actions"""
        if key == "SPACE":
            self.typed_text += " "
        elif key == "BACKSPACE":
            self.typed_text = self.typed_text[:-1] if self.typed_text else ""
        elif key == "ENTER":
            self.typed_text += "\n"
        elif key == "SHIFT":
            # Toggle case for next key press would go here
            pass
        else:
            if len(self.typed_text) < self.max_text_length:
                self.typed_text += key

def detect_available_cameras(max_cameras=4):
    """
    Scan and detect available cameras on the system.
    Returns a list of available camera indices.
    """
    available_cameras = []
    print("Scanning for available cameras...")
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"Camera {i} detected and working")
            else:
                print(f"Camera {i} detected but not returning frames")
            cap.release()
        else:
            print(f"Camera {i} not available")
            
    print(f"Found {len(available_cameras)} working cameras: {available_cameras}")
    return available_cameras

class HandGestureRecognizer:
    """Class to handle hand gesture recognition for specific keyboard actions"""
    def __init__(self):
        # Gesture state
        self.pinch_threshold = 0.05  # Distance threshold for pinch detection
        self.prior_gestures = deque(maxlen=5)  # Store recent gesture history
        
    def recognize_gestures(self, hand_landmarks):
        """Recognize gestures based on hand landmarks"""
        if not hand_landmarks:
            return None
            
        gestures = []
        
        # Check for pinch gesture (index finger and thumb)
        if self._is_pinch(hand_landmarks):
            gestures.append("PINCH")
            
        # Check for pointing gesture (index finger extended, others closed)
        if self._is_pointing(hand_landmarks):
            gestures.append("POINTING")
            
        # Check for fist gesture (all fingers closed)
        if self._is_fist(hand_landmarks):
            gestures.append("FIST")
            
        # Store gesture history
        self.prior_gestures.append(gestures)
        
        return gestures
        
    def _is_pinch(self, hand_landmarks):
        """Detect pinch gesture (thumb and index finger close together)"""
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Calculate distance between thumb and index finger tips
        distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + 
                          (thumb_tip.y - index_tip.y)**2 + 
                          (thumb_tip.z - index_tip.z)**2)
                          
        return distance < self.pinch_threshold
        
    def _is_pointing(self, hand_landmarks):
        """Detect pointing gesture (index finger extended, others closed)"""
        # Check if index finger is extended
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        
        # Index finger should be extended (tip higher than PIP joint)
        index_extended = index_tip.y < index_pip.y
        
        # Check other fingers are folded
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        middle_folded = middle_tip.y > middle_pip.y
        
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
        ring_folded = ring_tip.y > ring_pip.y
        
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
        pinky_folded = pinky_tip.y > pinky_pip.y
        
        # For pointing, we want index extended but others folded
        return index_extended and middle_folded and ring_folded and pinky_folded
        
    def _is_fist(self, hand_landmarks):
        """Detect fist gesture (all fingers closed)"""
        # Check all fingertips are below their PIP joints (folded into palm)
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        index_folded = index_tip.y > index_pip.y
        
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        middle_folded = middle_tip.y > middle_pip.y
        
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
        ring_folded = ring_tip.y > ring_pip.y
        
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
        pinky_folded = pinky_tip.y > pinky_pip.y
        
        return index_folded and middle_folded and ring_folded and pinky_folded
        
    def gesture_state_changed(self, gesture):
        """Check if a gesture state just changed based on history"""
        if len(self.prior_gestures) < 2:
            return False
            
        # Check if current state has the gesture but previous didn't
        return (gesture in self.prior_gestures[-1] and 
                gesture not in self.prior_gestures[-2])

class CameraProcessor:
    def __init__(self, camera_id, width=640, height=480, role="secondary"):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.role = role  # "main" or "secondary"
        self.cap = cv2.VideoCapture(camera_id)
        
        # Try different backends if the default doesn't work
        if not self.cap.isOpened():
            print(f"Trying different backend for camera {camera_id}")
            # Try DirectShow backend on Windows
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        
        # Configure camera settings if successfully opened
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # Try to set a reasonable FPS for better performance while ensuring tracking works
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            
            # Verify and print the actual settings
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"Camera {camera_id} initialized with resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
        else:
            print(f"Failed to open camera {camera_id}")
        
        # Initialize MediaPipe Hands for this camera
        self.hands = None  # Will be initialized in start() method
        
        # Configure tracking parameters based on role
        self.detection_confidence = 0.6 if role == "main" else 0.5
        self.tracking_confidence = 0.6 if role == "main" else 0.5
        self.model_complexity = 1    # Moderate complexity for all cameras
            
        self.latest_frame = None
        self.latest_processed_frame = None
        self.latest_hand_landmarks = None  # Store the hand landmarks
        self.running = False
        self.lock = threading.Lock()
        self.frames_processed = 0
        self.processing_time = 0
        self.start_time = None
        self.is_connected = self.cap.isOpened()
        
        # Add health monitoring
        self.last_frame_time = 0
        self.health_status = "Unknown"
        self.hand_detected = False  # Track if hand was detected in last frame
        
        # Add hand tracking data
        self.hand_pose_data = {}  # Dictionary to store hand joint positions
        self.hand_gesture_recognizer = HandGestureRecognizer()
        self.detected_gestures = []
        
        # Variables for smoothing finger positions
        self.finger_positions = {}
        self.position_history = {}
        self.smoothing_window = 5

    def start(self):
        if not self.is_connected:
            print(f"Cannot start camera {self.camera_id} - not connected")
            return False
        
        # Initialize the MediaPipe Hands here to ensure it's in the correct thread context
        self.hands = mp_hands.Hands(
            static_image_mode=False,  # Use tracking between frames for better performance
            max_num_hands=2,  # Track up to 2 hands
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
            model_complexity=self.model_complexity
        )
        
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self.process_frames)
        self.thread.daemon = True
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()
        # Explicitly close the hands object
        if self.hands:
            self.hands.close()

    def process_frames(self):
        consecutive_failures = 0
        max_failures = 5  # Restart camera after this many consecutive failures
        
        while self.running:
            if not self.cap.isOpened():
                self.health_status = "Disconnected"
                print(f"Camera {self.camera_id} disconnected. Attempting to reconnect...")
                self.cap = cv2.VideoCapture(self.camera_id)
                if not self.cap.isOpened():
                    # Try DirectShow backend on Windows
                    self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
                
                if self.cap.isOpened():
                    # Reconfigure camera settings
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.cap.set(cv2.CAP_PROP_FPS, 15)
                    print(f"Camera {self.camera_id} reconnected")
                    consecutive_failures = 0
                    self.health_status = "Connected"
                else:
                    print(f"Failed to reconnect camera {self.camera_id}")
                    time.sleep(1)  # Wait before trying again
                    continue
            
            success, image = self.cap.read()
            if not success:
                consecutive_failures += 1
                self.health_status = f"Failing ({consecutive_failures}/{max_failures})"
                print(f"Failed to capture image from camera {self.camera_id} ({consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    print(f"Too many failures for camera {self.camera_id}, attempting to restart...")
                    self.cap.release()
                    consecutive_failures = 0
                    
                time.sleep(0.1)
                continue
            
            # Reset failure counter on successful frame capture
            consecutive_failures = 0
            self.health_status = "Healthy"
            self.last_frame_time = time.time()
            
            # Always process every frame for hand detection
            start_process = time.time()
            
            try:
                # Make a copy of the image to avoid modification issues
                image_copy = image.copy()
                
                # Convert image to RGB for MediaPipe
                image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
                
                # Process image with MediaPipe Hands
                image_rgb.flags.writeable = False
                results = self.hands.process(image_rgb)
                
                # Set writeable back to True for drawing
                image_rgb.flags.writeable = True
                processed_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                # Update hand detection status and landmarks
                self.hand_detected = results.multi_hand_landmarks is not None
                
                # Store latest hand landmarks
                with self.lock:
                    if self.hand_detected and results.multi_hand_landmarks:
                        self.latest_hand_landmarks = results.multi_hand_landmarks[0]  # Take first hand for now
                        
                        # Process hand gestures if this is the main camera
                        if self.role == "main":
                            self.detected_gestures = self.hand_gesture_recognizer.recognize_gestures(self.latest_hand_landmarks)
                        
                        # Extract and store hand joint positions with smoothing
                        self._update_hand_pose_data(results.multi_hand_landmarks[0])
                    else:
                        self.latest_hand_landmarks = None
                        self.detected_gestures = []
                
                # Draw hand landmarks if detected
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            processed_image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # Draw additional gesture information if this is the main camera
                        if self.role == "main" and self.detected_gestures:
                            # Draw gesture name at top of screen
                            gesture_text = ", ".join(self.detected_gestures)
                            cv2.putText(processed_image, f"Gesture: {gesture_text}", 
                                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display camera ID, role, and hand tracking status
                role_text = "MAIN" if self.role == "main" else "AUX"
                status_color = (0, 255, 0) if self.hand_detected else (0, 0, 255)
                status_text = "Hands Detected" if self.hand_detected else "No Hands"
                cv2.putText(processed_image, f"Camera {self.camera_id} ({role_text}): {status_text}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Calculate processing metrics
                process_end = time.time()
                process_time = process_end - start_process
                
                with self.lock:
                    self.latest_frame = image.copy()
                    self.latest_processed_frame = processed_image
                    self.frames_processed += 1
                    self.processing_time += process_time
                    
            except Exception as e:
                print(f"Error processing frame from camera {self.camera_id}: {str(e)}")
                # Continue despite errors to prevent freezing the application
                with self.lock:
                    if image is not None:
                        self.latest_frame = image.copy()
                        # Add error message to the frame if processing failed
                        cv2.putText(image, f"Error: {str(e)[:30]}", (10, 70), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        self.latest_processed_frame = image.copy()

    def _update_hand_pose_data(self, hand_landmarks):
        """Extract and smooth hand joint positions"""
        # Initialize positions dictionary if needed
        if not self.position_history:
            for landmark_id in range(21):  # MediaPipe tracks 21 hand landmarks
                self.position_history[landmark_id] = deque(maxlen=self.smoothing_window)
        
        # Extract current positions
        h, w, _ = self.latest_frame.shape
        current_positions = {}
        
        for landmark_id in range(21):
            landmark = hand_landmarks.landmark[landmark_id]
            # Convert normalized coordinates to pixel coordinates
            px = min(int(landmark.x * w), w - 1)
            py = min(int(landmark.y * h), h - 1)
            pz = landmark.z
            
            # Add to history
            self.position_history[landmark_id].append((px, py, pz))
            
            # Calculate smoothed position (average of last N positions)
            if len(self.position_history[landmark_id]) > 0:
                x_avg = sum([p[0] for p in self.position_history[landmark_id]]) / len(self.position_history[landmark_id])
                y_avg = sum([p[1] for p in self.position_history[landmark_id]]) / len(self.position_history[landmark_id])
                z_avg = sum([p[2] for p in self.position_history[landmark_id]]) / len(self.position_history[landmark_id])
                current_positions[landmark_id] = (int(x_avg), int(y_avg), z_avg)
            else:
                current_positions[landmark_id] = (px, py, pz)
        
        # Update the finger positions
        self.finger_positions = current_positions
        
        # Store standard hand keypoints in a more accessible format
        self.hand_pose_data = {
            "wrist": current_positions.get(mp_hands.HandLandmark.WRIST, (0, 0, 0)),
            "thumb_tip": current_positions.get(mp_hands.HandLandmark.THUMB_TIP, (0, 0, 0)),
            "index_finger_tip": current_positions.get(mp_hands.HandLandmark.INDEX_FINGER_TIP, (0, 0, 0)),
            "middle_finger_tip": current_positions.get(mp_hands.HandLandmark.MIDDLE_FINGER_TIP, (0, 0, 0)),
            "ring_finger_tip": current_positions.get(mp_hands.HandLandmark.RING_FINGER_TIP, (0, 0, 0)),
            "pinky_tip": current_positions.get(mp_hands.HandLandmark.PINKY_TIP, (0, 0, 0))
        }

    def get_latest_frames(self):
        with self.lock:
            return self.latest_frame, self.latest_processed_frame
    
    def get_performance_metrics(self):
        with self.lock:
            if self.frames_processed == 0 or self.start_time is None:
                return 0, 0
            
            elapsed = time.time() - self.start_time
            fps = self.frames_processed / elapsed if elapsed > 0 else 0
            avg_process_time = self.processing_time / self.frames_processed if self.frames_processed > 0 else 0
            return fps, avg_process_time
            
    def get_health_status(self):
        # Check if we haven't received a frame in a while
        if self.health_status == "Healthy" and time.time() - self.last_frame_time > 2.0:
            return "Stalled"
        return self.health_status
    
    def is_hand_detected(self):
        return self.hand_detected
        
    def get_hand_landmarks(self):
        with self.lock:
            return self.latest_hand_landmarks
            
    def get_hand_pose_data(self):
        with self.lock:
            return self.hand_pose_data.copy()
            
    def get_detected_gestures(self):
        with self.lock:
            return self.detected_gestures.copy()

def optimize_gif_size(frames, target_size_mb=10, initial_quality=100, min_quality=30):
    """
    Optimize a list of PIL Image frames to target a specific file size.
    Returns the optimized frames and the quality level used.
    """
    target_size_bytes = target_size_mb * 1024 * 1024
    current_quality = initial_quality
    optimized_frames = frames
    
    while current_quality >= min_quality:
        # Create temporary optimized frames
        temp_frames = []
        for img in frames:
            # Convert to RGB mode if necessary
            if img.mode != 'RGB':
                img_rgb = img.convert('RGB')
            else:
                img_rgb = img
                
            # Create a BytesIO object to measure size
            buffer = io.BytesIO()
            img_rgb.save(buffer, format="JPEG", quality=current_quality)
            temp_frames.append(Image.open(buffer))
            
        # Estimate total size (rough approximation)
        total_est_size = sum(len(io.BytesIO(frame.tobytes()).getvalue()) for frame in temp_frames)
        
        if total_est_size <= target_size_bytes:
            # We've reached our target size
            optimized_frames = temp_frames
            break
            
        # Reduce quality and try again
        current_quality -= 5
        
    print(f"Optimized frames to quality level: {current_quality}")
    return optimized_frames, current_quality

def main():
    # Find available cameras
    available_cameras = detect_available_cameras()
    
    if not available_cameras:
        print("No cameras detected. Please connect a camera and try again.")
        sys.exit(1)
        
    # Use the first camera as main (for keyboard interaction)
    main_camera_id = available_cameras[0]
    
    # Initialize camera processors
    camera_processors = []
    
    # Main camera for hand tracking and keyboard interaction
    main_processor = CameraProcessor(
        camera_id=main_camera_id,
        width=640,
        height=480,
        role="main"
    )
    if main_processor.is_connected:
        camera_processors.append(main_processor)
        print(f"Main camera (ID {main_camera_id}) initialized")
    else:
        print("Failed to initialize main camera. Exiting.")
        sys.exit(1)
    
    # Additional cameras (if available)
    for i, cam_id in enumerate(available_cameras[1:]):
        secondary_processor = CameraProcessor(
            camera_id=cam_id,
            width=640,
            height=480,
            role="secondary"
        )
        if secondary_processor.is_connected:
            camera_processors.append(secondary_processor)
            print(f"Secondary camera {i+1} (ID {cam_id}) initialized")
    
    # Start processing in all cameras
    for processor in camera_processors:
        processor.start()
    
    # Initialize virtual keyboard
    keyboard = VirtualKeyboard()
    
    try:
        while True:
            # Get main camera for interaction (should be the first one)
            main_processor = camera_processors[0]
            
            # Get frame and hand landmarks from main camera
            _, processed_frame = main_processor.get_latest_frames()
            hand_landmarks = main_processor.get_hand_landmarks()
            
            if processed_frame is None:
                print("No frame available from main camera")
                time.sleep(0.1)
                continue
            
            # Draw virtual keyboard on the frame
            hover_key = keyboard.draw_keyboard(processed_frame, hand_landmarks)
            
            # Check for gestures from main camera
            gestures = main_processor.get_detected_gestures()
            
            # Handle keyboard interaction with pinch gesture
            if hover_key and "PINCH" in gestures:
                # If pinch detected on a key, activate it
                key_activated = keyboard.update_key_states(hover_key, True)
                
                # If key has been activated (held long enough)
                if key_activated:
                    keyboard.process_key_action(hover_key)
                    print(f"Key pressed: {hover_key}")
            else:
                # Reset any previously pressed keys
                for key in keyboard.key_status.keys():
                    keyboard.update_key_states(key, False)
            
            # Display FPS on the main frame
            fps, process_time = main_processor.get_performance_metrics()
            health_status = main_processor.get_health_status()
            cv2.putText(processed_frame, f"FPS: {fps:.1f}, Process: {process_time*1000:.1f}ms, Status: {health_status}", 
                     (10, processed_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Create a multi-view layout for multiple cameras
            if len(camera_processors) > 1:
                # Create a layout for all camera feeds
                rows = (len(camera_processors) + 1) // 2  # Ceiling division
                cols = min(2, len(camera_processors))
                
                # Calculate grid cell size
                grid_width = 320  # Smaller size for secondary views
                grid_height = 240
                
                # Create composite frame
                height = max(processed_frame.shape[0], rows * grid_height)
                width = processed_frame.shape[1] + cols * grid_width
                composite = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Add main processed frame on left
                composite[0:processed_frame.shape[0], 0:processed_frame.shape[1]] = processed_frame
                
                # Add secondary camera views as smaller frames on right
                for i, processor in enumerate(camera_processors[1:]):
                    row = i // cols
                    col = i % cols
                    
                    _, secondary_frame = processor.get_latest_frames()
                    if secondary_frame is not None:
                        # Resize secondary frame to fit grid
                        secondary_resized = cv2.resize(secondary_frame, (grid_width, grid_height))
                        
                        # Place in grid
                        x_offset = processed_frame.shape[1] + col * grid_width
                        y_offset = row * grid_height
                        
                        # Make sure we don't exceed the composite frame bounds
                        h, w = secondary_resized.shape[:2]
                        if y_offset + h <= composite.shape[0] and x_offset + w <= composite.shape[1]:
                            composite[y_offset:y_offset+h, x_offset:x_offset+w] = secondary_resized
                
                # Display the composite frame
                cv2.imshow('Virtual Keyboard with Hand Tracking', composite)
            else:
                # Just show the main processed frame
                cv2.imshow('Virtual Keyboard with Hand Tracking', processed_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error in main loop: {str(e)}")
    finally:
        # Clean up
        for processor in camera_processors:
            processor.stop()
        cv2.destroyAllWindows()
        print("Application terminated")

if __name__ == "__main__":
    import io  # Added import for BytesIO used in optimize_gif_size
    main()
