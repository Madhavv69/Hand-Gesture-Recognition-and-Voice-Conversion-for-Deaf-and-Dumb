# Hand-Gesture Recognition and Voice Conversion for Deaf and Dumb

This project aims to assist individuals who are deaf and dumb by recognizing hand gestures using a webcam and converting them into speech. The system uses MediaPipe for hand landmark detection, a Random Forest classifier for gesture recognition, and the `playsound` library to play corresponding audio files.

## Features

- **Real-time Hand Gesture Recognition:** Captures video from the webcam and identifies hand gestures.
- **Voice Conversion:** Plays an audio file corresponding to the recognized gesture.
- **Debounce Mechanism:** Prevents repeated audio playback of the same gesture within a short time interval.

## Installation

1. **Clone the Repository:**
    ```bash
    git clone https:https://github.com/Madhavv69/Hand-Gesture-Recognition-and-Voice-Conversion-for-Deaf-and-Dumb.git
    cd Hand-Gesture-Recognition-and-Voice-Conversion-for-Deaf-and-Dumb
    ```

2. **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare Dataset:**
    - Collect and label hand gesture images. Ensure you have the dataset organized in subdirectories named after each gesture class.

2. **Train the Model:**
    - Run the training script to train the Random Forest model.
    ```bash
    python train_model.py
    ```

3. **Run the Gesture Recognition and Voice Conversion:**
    - Start the main application to begin real-time gesture recognition and voice conversion.
    ```bash
    python hand_gesture_recognition.py
    ```

## File Descriptions

- `train_model.py`: Script for training the Random Forest model using hand landmark data.
- `hand_gesture_recognition.py`: Main application script for real-time hand gesture recognition and voice conversion.
- `requirements.txt`: Lists all the dependencies required to run the project.
- `data.pickle`: Contains preprocessed hand landmark data and labels.
- `model.p`: Contains the trained Random Forest model.
- `audio_files/`: Directory containing audio files for each gesture.

## Hand Gesture Classes

The following hand gestures are recognized by the system:

- A
- B
- L
- V
- Y
- Hello
- I Love You
- Thank You

## Dependencies

- OpenCV
- MediaPipe
- scikit-learn
- numpy
- playsound

## How It Works

1. **Data Collection:** Collect hand gesture images using a webcam and store them in labeled directories.
2. **Model Training:** Extract hand landmarks using MediaPipe and train a Random Forest classifier on the labeled data.
3. **Real-time Recognition:** Capture video from the webcam, detect hand landmarks, predict the gesture, and play the corresponding audio file.

## Contributing

Contributions are welcome! Please fork this repository and submit pull requests.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev) by Google for hand landmark detection.
- [scikit-learn](https://scikit-learn.org) for machine learning algorithms.
- [OpenCV](https://opencv.org) for video capture and processing.

