import cv2
import mediapipe as mp
import time
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense, Dropout #type: ignore
from tensorflow.keras.optimizers import Adam, RMSprop, SGD #type: ignore
from sklearn.preprocessing import StandardScaler
from collections import deque
import random
import tensorflow as tf #type: ignore

# Constants
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
NUM_LANDMARKS = 21 * 3  # 21 landmarks with x, y, z coordinates
NUM_GESTURES = 10  # 0 to 9
SMOOTHING_FRAMES=5

# Adaptive Evolutionary Learning Model (AELM)
class AELM:
    def __init__(self, mutation_rate=0.2):
        self.mutation_rate = mutation_rate
        self.optimizers = ['adam', 'rmsprop', 'sgd']

    def evolve_optimizer(self, current_accuracy):
        if current_accuracy < 0.85 and random.random() < self.mutation_rate:
            return random.choice(self.optimizers)
        return 'adam'  # Default

# Manta Ray Foraging Optimization (MRFO)
class MRFO:
    def optimize_hyperparams(self):
        lr = random.choice([0.001, 0.0005, 0.01])
        dropout = random.choice([0.2, 0.3, 0.4])
        return lr, dropout

# Data Collection (Simplified - You'll need to record your own data)
def collect_data(num_samples=150):  # Increased number of samples
    mp_hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )
    data = []
    labels = []
    current_label = 0  # Start with gesture '0'
    landmark_buffer = deque(maxlen=SMOOTHING_FRAMES)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera for data collection.")
        return np.array(data), np.array(labels)

    print("Press 'n' to switch to the next gesture.")
    print("Press 'q' to finish data collection.")
    print("Try to keep your hand relatively stable for a few seconds while recording each sample.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )
                landmark_list = np.array([[res.x, res.y, res.z] for res in landmarks.landmark]).flatten()
                landmark_buffer.append(landmark_list)

                if len(landmark_buffer) == SMOOTHING_FRAMES:
                    smoothed_landmarks = np.mean(list(landmark_buffer), axis=0)
                    data.append(smoothed_landmarks)
                    labels.append(current_label)
                    cv2.putText(frame, f"Gesture: {current_label}, Samples: {len(data)}/{num_samples * NUM_GESTURES}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    time.sleep(0.2)  # Small delay to avoid rapid sampling

        cv2.imshow('Data Collection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            current_label = (current_label + 1) % NUM_GESTURES
            print(f"Collecting data for gesture: {current_label}")
        elif key == ord('q'):
            break
        elif len(data) >= num_samples * NUM_GESTURES:
            print("Data collection complete.")
            break

    cap.release()
    cv2.destroyAllWindows()
    mp_hands.close()
    return np.array(data), np.array(labels)

# Model Training with AELM and MRFO
def train_gesture_model_optimized(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    aelm = AELM()
    mrfo = MRFO()
    best_model = None
    best_accuracy = 0.0
    optimizer_history = []

    for epoch in range(epochs):
        lr, dropout = mrfo.optimize_hyperparams()
        optimizer_name = aelm.evolve_optimizer(best_accuracy)
        optimizer_history.append(optimizer_name)

        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=lr)
        elif optimizer_name == 'rmsprop':
            optimizer = RMSprop(learning_rate=lr)
        elif optimizer_name == 'sgd':
            optimizer = SGD(learning_rate=lr)

        model = Sequential([
            Dense(64, activation='relu', input_shape=(NUM_LANDMARKS,)),
            Dropout(dropout),
            Dense(128, activation='relu'),
            Dropout(dropout),
            Dense(NUM_GESTURES, activation='softmax')
        ])

        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(X_train, y_train, epochs=1, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0)
        val_accuracy = history.history['val_accuracy'][0]

        model.save('gesture_model.h5')
        print("Model training completed and saved as gesture_model.h5")

        # Evaluate the model on test set
        test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
        print(f"Test accuracy: {test_acc:.4f}")

        print(f"Epoch {epoch+1}/{epochs}, Optimizer: {optimizer_name}, LR: {lr:.4f}, Dropout: {dropout:.2f}, Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model

    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    print(f"Optimizer Evolution History: {optimizer_history}")
    return best_model

# Gesture Recognition
class NewGestureRecognizer:
    def __init__(self, model_path='gesture_model.h5'):
        self.mp_hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils
        try:
            self.model = tf.keras.models.load_model(model_path)
        except FileNotFoundError:
            self.model = None
            print(f"Warning: Model not found at {model_path}. Please train the model first.")
        self.scaler = StandardScaler() # Will be loaded with training data scaling
        try:
            self.scaler.mean_ = np.load('scaler_mean.npy')
            self.scaler.scale_ = np.load('scaler_scale.npy')
        except FileNotFoundError:
            print("Warning: Scaler parameters not found. Please train the model first.")

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(frame_rgb)
        gesture = None

        if self.model and hasattr(self.scaler, 'mean_'):
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    landmark_list = np.array([[res.x, res.y, res.z] for res in landmarks.landmark]).flatten()
                    normalized_landmarks = self.scaler.transform([landmark_list])
                    prediction = self.model.predict(normalized_landmarks, verbose=0)[0]
                    predicted_label = np.argmax(prediction)
                    confidence = np.max(prediction)
                    gesture = str(predicted_label)

        return frame, gesture

def main_new_gesture():
    # 1. Collect Data (Uncomment and run once to collect your gesture data)
    print("Hello world")
    collect_new_data = False# Set to True to collect new data
    if collect_new_data:
        data, labels = collect_data(num_samples=200)
        if data.size > 0:
            X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            best_gesture_model = train_gesture_model_optimized(X_train_scaled, y_train, X_val_scaled, y_val, epochs=50)
            best_gesture_model.save('gesture_model.h5')
            # Save the scaler's parameters for later use
            np.save('scaler_mean.npy', scaler.mean_)
            np.save('scaler_scale.npy', scaler.scale_)
        else:
            print("No data collected. Cannot train model.")
            return

    # 2. Load Trained Model and Scaler
    scaler=StandardScaler()
    recognizer = NewGestureRecognizer()
    scaler.mean_ = np.load('scaler_mean.npy')
    scaler.scale_ = np.load('scaler_scale.npy')

    print("Attempting to open camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            processed_frame, gesture = recognizer.process_frame(frame)

            if gesture:
                cv2.putText(processed_frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Hand Gesture Recognition', processed_frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main_new_gesture()
