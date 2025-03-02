import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure Hands module
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)

# List to store frames for GIF creation
frames = []

# Configure GIF output settings
gif_output_path = "hand_tracking_result.gif"
frame_count = 0
max_frames = 100  # Capture a limited number of frames for a smaller GIF
frame_rate = 15   # FPS for the GIF
gif_size = (320, 240)  # Resizing frames to reduce GIF size

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to capture image from camera.")
        continue
    
    # Convert image to RGB and process
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    
    # Convert back to BGR for OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Draw hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    # Resize frame for smaller GIF size
    resized_image = cv2.resize(image, gif_size)
    
    # Convert to PIL image and append to frames list
    pil_image = Image.fromarray(resized_image)
    frames.append(pil_image)

    # Display FPS
    cv2.putText(image, f'Press ESC to exit', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the image
    cv2.imshow('Hand Tracking', image)
    
    frame_count += 1
    
    # Stop after capturing max_frames
    if frame_count >= max_frames:
        break

    # Exit on ESC key
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Save frames as a GIF
if frames:
    frames[0].save(gif_output_path, save_all=True, append_images=frames[1:], loop=0, duration=1000 // frame_rate, optimize=True)

print(f"GIF saved to {gif_output_path}")

# Release resources
cap.release()
cv2.destroyAllWindows()

