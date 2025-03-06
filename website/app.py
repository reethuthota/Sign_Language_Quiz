from flask import Flask, render_template, Response, jsonify
import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
import random
from collections import deque

app = Flask(__name__)

class HandTrackingDynamic:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.__mode__ = mode
        self.__maxHands__ = maxHands
        self.__detectionCon__ = detectionCon
        self.__trackCon__ = trackCon
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands(
            static_image_mode=mode,
            max_num_hands=maxHands,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findFingers(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(frame, handLms, self.handsMp.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame):
        h, w, _ = frame.shape
        left_hand = np.zeros((21, 3))
        right_hand = np.zeros((21, 3))

        if self.results.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
                is_right = hand_handedness.classification[0].label == "Right"
                hand_data = np.array([[lm.x * w, lm.y * h, lm.z] for lm in hand_landmarks.landmark])
                
                if is_right:
                    right_hand = hand_data
                else:
                    left_hand = hand_data

        return np.concatenate([left_hand.flatten(), right_hand.flatten()])

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Load your trained model
model_path = '/Users/reethu/coding/Projects/Sign-Language-Quiz/model.h5'
model = tf.keras.models.load_model(model_path)

detector = HandTrackingDynamic(maxHands=2)
sequence_buffer = deque(maxlen=30)
labels = ['accident', 'apple', 'argue', 'bad', 'balance', 'bar', 'basketball', 'because', 'bed', 'before', 'bird', 'black', 'blanket', 'bowling', 'brother', 'call', 'candy', 'champion', 'change', 'cheat', 'check', 'cold', 'computer', 'convince', 'cool', 'corn', 'cousin', 'cry', 'dark', 'daughter', 'deaf', 'delay', 'delicious', 'doctor', 'dog', 'drink', 'environment', 'example', 'family', 'far', 'fat', 'fish', 'full', 'give', 'go', 'good', 'government', 'graduate', 'help', 'hot', 'interest', 'language', 'last', 'later', 'laugh', 'leave', 'letter', 'like', 'man', 'many', 'mother', 'move', 'no', 'orange', 'order', 'perspective', 'pizza', 'play', 'room', 'sandwich', 'score', 'secretary', 'shirt', 'short', 'silly', 'snow', 'son', 'soon', 'study', 'sweet', 'take', 'tall', 'tell', 'thanksgiving', 'theory', 'thin', 'thursday', 'trade', 'wait', 'walk', 'what', 'white', 'who', 'why', 'woman', 'work', 'write', 'year', 'yes', 'yesterday']

prediction_buffer = deque(maxlen=10)
confidence_threshold = 0.5
target_label = None

quiz_active = False
timer_expired = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_target_word')
def get_target_word():
    global target_label, quiz_active, timer_expired
    quiz_active = True
    timer_expired = False
    target_label = random.choice(labels)
    return jsonify({'target': target_label})

@app.route('/check_for_correct')
def check_for_correct():
    global quiz_active, timer_expired
    return jsonify({
        'quiz_ended': not quiz_active or timer_expired,
        'timer_expired': timer_expired
    })

@app.route('/stop_quiz')
def stop_quiz():
    global quiz_active, timer_expired
    quiz_active = False
    timer_expired = True
    return jsonify({'status': 'success'})

@app.route('/reset_quiz_state')
def reset_quiz_state():
    global quiz_active, timer_expired, sequence_buffer, prediction_buffer
    quiz_active = False
    timer_expired = False
    sequence_buffer.clear()
    prediction_buffer.clear()
    return jsonify({'status': 'success'})

def generate_frames():
    global quiz_active, timer_expired
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Camera not accessed!")
        return

    try:
        while quiz_active and not timer_expired:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame = detector.findFingers(frame, draw=True)
            landmarks = detector.findPosition(frame)
            sequence_buffer.append(landmarks)

            if len(sequence_buffer) == 30:
                sequence = np.array([list(sequence_buffer)])
                sequence = normalize_data(sequence)
                predictions = model.predict(sequence, verbose=0)
                predicted_class = np.argmax(predictions)
                confidence = predictions[0][predicted_class]

                prediction_buffer.append((predicted_class, confidence))
                if len(prediction_buffer) == prediction_buffer.maxlen:
                    class_counts = {}
                    total_confidence = {}
                    for cls, conf in prediction_buffer:
                        if cls not in class_counts:
                            class_counts[cls] = 0
                            total_confidence[cls] = 0
                        class_counts[cls] += 1
                        total_confidence[cls] += conf

                    smoothed_class = max(class_counts, key=lambda x: (class_counts[x], total_confidence[x] / class_counts[x]))
                    smoothed_confidence = total_confidence[smoothed_class] / class_counts[smoothed_class]

                    if smoothed_confidence > confidence_threshold:
                        predicted_label = labels[smoothed_class]
                        print(f'Predicted Word: {predicted_label}')
                        
                        if predicted_label == target_label:
                            quiz_active = False
                            break

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=True, port=5001)