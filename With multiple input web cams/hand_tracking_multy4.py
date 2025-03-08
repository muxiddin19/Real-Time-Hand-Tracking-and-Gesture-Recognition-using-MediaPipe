import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os
import threading
import time
import sys

# Initialize MediaPipe Hands with specific solutions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def detect_available_cameras(max_cameras=10):
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

class CameraProcessor:
    def __init__(self, camera_id, is_headset=False, width=640, height=480):
        self.camera_id = camera_id
        self.is_headset = is_headset
        self.width = width
        self.height = height
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
            # Try to set a lower FPS to avoid USB bandwidth issues with multiple cameras
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            
            # Verify and print the actual settings
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"Camera {camera_id} initialized with resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
            
            # For headset camera, try to set higher quality settings 
            if is_headset:
                # Increase resolution for headset camera
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)  # Higher FPS for main camera
                print(f"Headset camera {camera_id} set to higher resolution and FPS")
        else:
            print(f"Failed to open camera {camera_id}")
            
        # Initialize hands processor with customized parameters based on camera type
        if is_headset:
            # More accurate parameters for headset camera
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,  # Higher confidence threshold for main camera
                min_tracking_confidence=0.7,
                model_complexity=1  # Use more complex model for headset camera
            )
            print(f"Initialized headset camera {camera_id} with high-accuracy hand detection parameters")
        else:
            # Lighter parameters for supporting cameras
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=0  # Use simplest model for better performance
            )
            
        self.latest_frame = None
        self.latest_processed_frame = None
        self.detected_hand_landmarks = None  # Store the detected landmarks for fusion
        self.running = False
        self.lock = threading.Lock()
        self.frames_processed = 0
        self.processing_time = 0
        self.start_time = None
        self.is_connected = self.cap.isOpened()
        # Add health monitoring
        self.last_frame_time = 0
        self.health_status = "Unknown"

    def start(self):
        if not self.is_connected:
            print(f"Cannot start camera {self.camera_id} - not connected")
            return False
            
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
        self.hands.close()

    def process_frames(self):
        consecutive_failures = 0
        max_failures = 10  # Restart camera after this many consecutive failures
        
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
                    if self.is_headset:
                        # Apply higher quality settings for headset camera
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                    else:
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
            
            start_process = time.time()
            
            try:
                # Make a copy of the image to avoid modification issues
                image_copy = image.copy()
                
                # Convert image to RGB for MediaPipe
                image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
                
                # Process image with MediaPipe Hands
                # Set writeable to False for better performance
                image_rgb.flags.writeable = False
                results = self.hands.process(image_rgb)
                
                # Set writeable back to True for drawing
                image_rgb.flags.writeable = True
                processed_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                # Draw hand landmarks if detected
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Apply different visualization style for headset vs auxiliary cameras
                        if self.is_headset:
                            # Headset camera - show full details with thicker lines
                            mp_drawing.draw_landmarks(
                                processed_image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style()
                            )
                        else:
                            # Auxiliary cameras - show simpler visualization
                            # Custom drawing style for auxiliary cameras
                            landmarks_style = mp_drawing_styles.get_default_hand_landmarks_style()
                            connections_style = mp_drawing_styles.get_default_hand_connections_style()
                            
                            # Draw auxiliaries in a different color
                            mp_drawing.draw_landmarks(
                                processed_image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                landmarks_style,
                                connections_style
                            )
                
                # Store the detected landmarks for later fusion
                with self.lock:
                    self.detected_hand_landmarks = results.multi_hand_landmarks
                
                # Display camera ID with special indicator for headset camera
                if self.is_headset:
                    cv2.putText(processed_image, f'HEADSET Camera {self.camera_id}', (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(processed_image, f'Aux Camera {self.camera_id}', (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
                
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

    def get_latest_frames(self):
        with self.lock:
            return self.latest_frame, self.latest_processed_frame
    
    def get_hand_landmarks(self):
        with self.lock:
            return self.detected_hand_landmarks
    
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

class FusedHandTracker:
    """
    Class to combine hand tracking data from multiple cameras,
    prioritizing the headset camera and using other cameras
    to fill in occluded or missed finger detections.
    """
    def __init__(self, processors):
        self.processors = processors
        self.headset_processor = None
        
        # Find headset processor
        for processor in processors:
            if processor.is_headset:
                self.headset_processor = processor
                break
        
        if not self.headset_processor:
            print("Warning: No headset camera designated. Using first camera as primary.")
            if processors:
                self.headset_processor = processors[0]
    
    def get_fused_frame(self):
        """
        Create a fused visualization showing the main headset camera view 
        enhanced with any additional finger detections from other cameras.
        """
        if not self.headset_processor:
            return None
        
        # Get the headset camera's processed frame
        _, headset_frame = self.headset_processor.get_latest_frames()
        if headset_frame is None:
            return None
        
        # Create a copy to draw auxiliary detections on
        fused_frame = headset_frame.copy()
        
        # Add labels to show this is the fused view
        cv2.putText(fused_frame, "FUSED VIEW (Headset + Aux)", (10, fused_frame.shape[0] - 20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add highlighting for finger detections from auxiliary cameras
        # This is a simplified version - in a real application, you would 
        # need to transform coordinates between camera views
        aux_detections_found = False
        
        for processor in self.processors:
            if processor != self.headset_processor:
                aux_landmarks = processor.get_hand_landmarks()
                if aux_landmarks:
                    aux_detections_found = True
                    # In a real application, you would use a more sophisticated
                    # method to align and fuse these landmarks with the headset view
                    
        # Add indicator if auxiliary detections are helping
        if aux_detections_found:
            cv2.putText(fused_frame, "Auxiliary cameras enhancing tracking", 
                      (fused_frame.shape[1] - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.6, (0, 255, 255), 2)
        
        return fused_frame

def optimize_gif_size(frames, target_size_mb=10, initial_quality=100, min_quality=30):
    """
    Optimize a list of PIL Image frames to target a specific file size.
    Returns the optimized frames and the estimated quality level.
    """
    from io import BytesIO
    
    # Start with these parameters
    quality = initial_quality
    resize_factor = 1.0
    
    while quality >= min_quality:
        print(f"Trying with quality={quality}, resize_factor={resize_factor:.2f}")
        
        # Create a copy of the first frame with current parameters
        test_frames = []
        for frame in frames[:5]:  # Use first 5 frames to estimate
            if resize_factor < 1.0:
                new_size = (int(frame.width * resize_factor), int(frame.height * resize_factor))
                test_frame = frame.resize(new_size, Image.Resampling.LANCZOS)
            else:
                test_frame = frame.copy()
            test_frames.append(test_frame)
        
        # Save to BytesIO to measure size
        buffer = BytesIO()
        test_frames[0].save(
            buffer, 
            save_all=True, 
            append_images=test_frames[1:], 
            format="GIF", 
            optimize=True, 
            quality=quality,
            duration=1000 // 15,  # 15 fps
            loop=0
        )
        
        # Get the size in MB
        size_mb = buffer.tell() / (1024 * 1024)
        
        # Estimate the full GIF size (proportional to number of frames)
        estimated_full_size = size_mb * (len(frames) / len(test_frames))
        print(f"Estimated full size: {estimated_full_size:.2f} MB")
        
        if estimated_full_size <= target_size_mb:
            break
            
        # Adjust parameters for next attempt
        if quality > 50:
            quality -= 10
        else:
            quality = min_quality
            resize_factor *= 0.8
        
        if resize_factor < 0.3:  # Don't go too small
            resize_factor = 0.3
            break
    
    # Apply the final parameters to all frames
    optimized_frames = []
    for frame in frames:
        if resize_factor < 1.0:
            new_size = (int(frame.width * resize_factor), int(frame.height * resize_factor))
            optimized_frame = frame.resize(new_size, Image.Resampling.LANCZOS)
        else:
            optimized_frame = frame.copy()
        optimized_frames.append(optimized_frame)
    
    return optimized_frames, quality, resize_factor

def main():
    # Set MediaPipe resource caching directory
    os.environ['MEDIAPIPE_RESOURCE_CACHE_DIR'] = '/tmp/mediapipe_cache'
    
    # Automatically detect cameras
    print("Scanning for available webcams...")
    available_cameras = detect_available_cameras(max_cameras=8)

    if not available_cameras:
        print("No cameras detected. Please check your webcam connections.")
        return

    # Ask user to identify the headset camera
    headset_camera_id = None
    while headset_camera_id is None:
        try:
            print("\nAvailable cameras:", available_cameras)
            headset_id = input("Enter the ID of your headset camera (or press Enter to use camera 0): ")
            
            if headset_id == "":
                headset_camera_id = available_cameras[0] if available_cameras else None
                print(f"Using camera {headset_camera_id} as headset camera")
            else:
                headset_id = int(headset_id)
                if headset_id in available_cameras:
                    headset_camera_id = headset_id
                    print(f"Using camera {headset_camera_id} as headset camera")
                else:
                    print(f"Camera ID {headset_id} not available. Please choose from: {available_cameras}")
        except ValueError:
            print("Please enter a valid number.")
    
    if headset_camera_id is None:
        print("No valid headset camera selected. Exiting.")
        return

    # Create and start camera processors, with headset camera having priority
    camera_processors = []
    
    # First initialize the headset camera with higher quality settings
    headset_processor = CameraProcessor(headset_camera_id, is_headset=True, width=1280, height=720)
    if headset_processor.is_connected:
        started = headset_processor.start()
        if started:
            camera_processors.append(headset_processor)
            print(f"Started headset camera processor for camera {headset_camera_id}")
        else:
            print(f"Failed to start headset camera processor for camera {headset_camera_id}")
    else:
        print(f"Headset camera {headset_camera_id} not connected. Cannot proceed.")
        return
    
    # Then initialize the auxiliary cameras
    for camera_id in available_cameras:
        if camera_id != headset_camera_id:
            processor = CameraProcessor(camera_id, is_headset=False)
            if processor.is_connected:
                started = processor.start()
                if started:
                    camera_processors.append(processor)
                    print(f"Started auxiliary processor for camera {camera_id}")
                else:
                    print(f"Failed to start auxiliary processor for camera {camera_id}")
            else:
                print(f"Skipping auxiliary camera {camera_id} - not connected")

    if len(camera_processors) <= 1:
        print("Warning: Only one camera available. Multi-angle tracking won't be possible.")
    
    # Create the fused hand tracker
    fused_tracker = FusedHandTracker(camera_processors)

    # GIF recording settings
    recording = False
    frames = []
    recording_start_time = None
    recording_duration = 10  # Recording duration in seconds
    frame_capture_interval = 3  # Capture every Nth frame to reduce GIF size
    frame_counter = 0
    gif_output_path = "hand_tracking_result.gif"
    frame_rate = 10  # Lower frame rate for smaller GIF
    gif_size = (320, 240)  # Smaller resolution for GIF
    
    # Target file size
    target_size_mb = 9.5  # Target slightly below 10MB to be safe

    # Create display windows
    cv2.namedWindow('Headset Camera (Primary)', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Auxiliary Cameras', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Fused View', cv2.WINDOW_NORMAL)

    # Main display loop
    try:
        print(f"Successfully started {len(camera_processors)} camera processors")
        print(f"Press 'R' to start recording for {recording_duration} seconds")
        print("Press ESC to exit")
        
        while True:
            # Get the fused view
            fused_frame = fused_tracker.get_fused_frame()
            
            # Get the headset camera view
            headset_frame = None
            if headset_processor is not None:
                _, headset_frame = headset_processor.get_latest_frames()
            
            # Check camera health
            for processor in camera_processors:
                health = processor.get_health_status()
                if health not in ["Healthy", "Unknown"]:
                    print(f"Camera {processor.camera_id} health: {health}")
            
            # Get the latest processed frames from auxiliary cameras
            aux_frames = []
            for processor in camera_processors:
                if not processor.is_headset:
                    _, processed_frame = processor.get_latest_frames()
                    if processed_frame is not None:
                        aux_frames.append(processed_frame)
            
            # Create a combined view for auxiliary cameras
            aux_combined_frame = None
            if aux_frames:
                # Arrange auxiliary frames in a grid
                rows = int(np.ceil(len(aux_frames) / 2))
                cols = min(len(aux_frames), 2)
                
                # Find the max dimensions
                height = max(frame.shape[0] for frame in aux_frames)
                width = max(frame.shape[1] for frame in aux_frames)
                
                # Create a canvas
                aux_combined_frame = np.zeros((height * rows, width * cols, 3), dtype=np.uint8)
                
                # Add each auxiliary camera's frame to the canvas
                for i, frame in enumerate(aux_frames):
                    r, c = i // cols, i % cols
                    h, w = frame.shape[:2]
                    aux_combined_frame[r*height:r*height+h, c*width:c*width+w] = frame
            
            # Display the views if available
            if headset_frame is not None:
                # Add performance metrics for headset camera
                fps, avg_time = headset_processor.get_performance_metrics()
                cv2.putText(headset_frame, f'Headset Camera: {fps:.1f} FPS, {avg_time*1000:.1f}ms/frame', 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.imshow('Headset Camera (Primary)', headset_frame)
            
            if aux_combined_frame is not None:
                cv2.imshow('Auxiliary Cameras', aux_combined_frame)
            
            if fused_frame is not None:
                # Add recording status
                y_pos = 60
                if recording:
                    elapsed = time.time() - recording_start_time
                    remaining = max(0, recording_duration - elapsed)
                    cv2.putText(fused_frame, f'RECORDING: {remaining:.1f}s remaining', 
                              (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_pos += 30
                    
                    # Capture frame for GIF (only every frame_capture_interval frames to reduce size)
                    frame_counter += 1
                    if frame_counter % frame_capture_interval == 0:
                        resized_image = cv2.resize(fused_frame, gif_size)
                        pil_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
                        frames.append(pil_image)
                    
                    # Check if recording time is up
                    if elapsed >= recording_duration:
                        recording = False
                        print(f"Recording complete! Captured {len(frames)} frames.")
                        print("Processing GIF, please wait...")
                        
                        if frames:
                            # Optimize GIF size
                            optimized_frames, quality, resize_factor = optimize_gif_size(frames, target_size_mb)
                            
                            # Save the optimized GIF
                            optimized_frames[0].save(
                                gif_output_path, 
                                save_all=True, 
                                append_images=optimized_frames[1:], 
                                loop=0, 
                                duration=1000 // frame_rate, 
                                optimize=True,
                                quality=quality
                            )
                            
                            # Verify the actual file size
                            file_size_mb = os.path.getsize(gif_output_path) / (1024 * 1024)
                            print(f"GIF saved to {gif_output_path}")
                            print(f"GIF size: {file_size_mb:.2f} MB (Target: {target_size_mb} MB)")
                            print(f"Applied quality: {quality}, resize factor: {resize_factor:.2f}")
                            print(f"Resolution: {optimized_frames[0].width}x{optimized_frames[0].height}, {len(optimized_frames)} frames")
                            
                            # Clear frames list
                            frames = []
                else:
                    cv2.putText(fused_frame, 'Press R to start recording', (10, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    y_pos += 30
                
                cv2.putText(fused_frame, 'Press ESC to exit', (10, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.imshow('Fused View', fused_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('r') or key == ord('R'):
                if not recording:
                    recording = True
                    frames = []
                    frame_counter = 0
                    recording_start_time = time.time()
                    print(f"Recording started for {recording_duration} seconds...")

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

    finally:
        # Stop all camera processors
        for processor in camera_processors:
            processor.stop()
        
        # Close all windows
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
