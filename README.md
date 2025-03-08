# Real-Time-Hand-Tracking-and-Gesture-Recognition-using-MediaPipe
This project implements real-time hand tracking and hand gesture recognition using Google's MediaPipe framework. The system accurately detects and tracks hand landmarks in real time, enabling gesture-based interaction. Future updates will extend this work to support VR headsets and Leap Motion devices.

![hand_tracking_result5](https://github.com/user-attachments/assets/d2c12d31-9a09-45d5-a794-68314e471f7c)
![hand_tracking_result4](https://github.com/user-attachments/assets/976b23ef-3b26-4350-8804-bbd8262a0b79)
![hand_tracking_result3](https://github.com/user-attachments/assets/451df26a-727e-41ae-9ce4-e4e4c2c899d7)
![hand_tracking_result](https://github.com/user-attachments/assets/96182a9e-2b68-4fba-8ecc-8c6a41ebedf4)

![hand_tracking_resultU](https://github.com/user-attachments/assets/22649b35-5bf8-424b-998d-ff948318af3e)
![hand_tracking_resultB](https://github.com/user-attachments/assets/2daee80c-7220-4841-b0f0-571595a74718)


# Features

- Real-time Hand Tracking: Utilizes MediaPipe's Hand Tracking module for fast and accurate hand landmark detection.

- Gesture Recognition: Recognizes predefined hand gestures for interaction.

- Scalability: Planned future support for VR headsets and Leap Motion devices.

- Optimized for Performance: Runs efficiently on CPU and GPU for real-time applications.

# Installation

To set up the project, follow these steps:

# Prerequisites

Ensure you have Python installed (preferably Python 3.7+).

# Steps

# Clone the repository
git clone https://github.com/yourusername/mediapipe-hand-tracking.git
cd mediapipe-hand-tracking

# Create a virtual environment (optional but recommended)
python -m venv venv(hand_tracking)
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Usage

Run the hand tracking script using:

python hand_tracking.py

This will open a real-time webcam feed and detect hand landmarks.

# Future Work

- Integration with VR Headsets (Oculus, HTC Vive, etc.)

- Leap Motion Support

- Enhanced Gesture Recognition with Machine Learning

# Contributions

Contributions are welcome! Feel free to fork the repository and submit pull requests.

# License

This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments
- https://blog.tensorflow.org/2020/03/face-and-hand-tracking-in-browser-with-mediapipe-and-tensorflowjs.html
- Google MediaPipe Team

- Open-source contributors
