# Real-Time Eye Blink Detection

This project is a Python script for real-time eye blink detection using a webcam feed. It utilizes computer vision techniques and machine learning models to detect facial landmarks and analyze the aspect ratio of the eyes to determine blinks.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- imutils
- dlib
- scipy
- pygame

## Installation

1. Clone this repository to your local machine:

2. Install the required Python libraries


3. Download the pre-trained facial landmark predictor (`shape_predictor_68_face_landmarks.dat`) from the [dlib website](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the project directory.

## Usage

Run the Python script `eye_blink_detection.py` to start the real-time eye blink detection system. Ensure that your webcam is connected and accessible.


Press the 'q' key to exit the program.

## How it Works

The script uses the following steps to detect eye blinks:

1. Capture video frames from the webcam.
2. Detect faces in each frame using the dlib library.
3. Predict facial landmarks for each detected face.
4. Calculate the eye aspect ratio (EAR) based on the distances between specific facial landmarks.
5. Determine if the EAR falls below a certain threshold, indicating an eye blink.
6. Display real-time video with eye contours and blink alerts.

## Credits

- [OpenCV](https://opencv.org/)
- [imutils](https://github.com/jrosebr1/imutils)
- [dlib](http://dlib.net/)
- [Pygame](https://www.pygame.org/)


