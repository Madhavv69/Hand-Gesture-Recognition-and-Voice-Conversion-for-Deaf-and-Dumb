import pickle
import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound
import threading
import time
import os

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Define labels
labels_dict = {
    0: 'A',
    1: 'B',
    2: 'L',
    3: 'V',
    4: 'Y',
    5: 'Hello',
    6: 'I Love You',
    7: 'Thank You'
}

# Define audio paths
audio_files = {
    'A': r"C:\Users\Chinu\Downloads\A.wav",
    'B': r"C:\Users\Chinu\Downloads\B.wav",
    'L': r"C:\Users\Chinu\Downloads\L.wav",
    'V': r"C:\Users\Chinu\Downloads\V.wav",
    'Y': r"C:\Users\Chinu\Downloads\Y.wav",
    'Hello': r"C:\Users\Chinu\Downloads\Hello.wav",
    'I Love You': r"C:\Users\Chinu\Downloads\I love You.wav",
    'Thank You': r"C:\Users\Chinu\Downloads\ThankYou.wav"
}

# Function to play sound
def play_sound(file):
    threading.Thread(target=playsound, args=(file,), daemon=True).start()

# Variables to keep track of the last prediction and debounce time
last_prediction = None  # Stores the last predicted character
last_prediction_time = 0  # Stores the time of the last prediction
debounce_time = 2  # Minimum time interval between the same predictions (in seconds)

# Define desired window size
window_width = 800
window_height = 600

# Create and resize the window
cv2.namedWindow('Hand Gesture Recognition', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hand Gesture Recognition', window_width, window_height)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            data_aux = []
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

            print(f"Predicted character: {predicted_character}")  # Debug statement

            # Play audio with debounce mechanism
            current_time = time.time()
            if predicted_character != last_prediction or (current_time - last_prediction_time) > debounce_time:
                last_prediction = predicted_character
                last_prediction_time = current_time
                audio_file = audio_files.get(predicted_character)
                if audio_file:
                    if os.path.exists(audio_file):
                        print(f"Playing audio: {audio_file}")  # Debug statement
                        play_sound(audio_file)
                    else:
                        print(f"Audio file not found: {audio_file}")  # Debug statement

    # Resize the frame to the desired window size
    resized_frame = cv2.resize(frame, (window_width, window_height))

    cv2.imshow('Hand Gesture Recognition', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
