import cv2
import mediapipe as mp
import numpy as np
import random
import time
from collections import deque
import math

# Constants
MAX_FRAMES_WITHOUT_GESTURE = 30
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

class AELM:
    """Adaptive Evolutionary Learning Model for optimizer selection"""
    def __init__(self, mutation_rate=0.2, population_size=5):
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.optimizers = ['adam', 'rmsprop', 'sgd', 'nadam']
    
    def evolve_optimizer(self):
        best_optimizer = random.choice(self.optimizers)
        for _ in range(self.population_size):
            if random.random() < self.mutation_rate:
                best_optimizer = random.choice(self.optimizers)
        return best_optimizer

class MRFO:
    """Manta Ray Foraging Optimization for parameter tuning"""
    def __init__(self, max_iter=10, population_size=5):
        self.max_iter = max_iter
        self.population_size = population_size
    
    def optimize_parameters(self):
        confidences = np.linspace(0.3, 0.7, 5)
        max_hands = [1, 2]
        best_confidence = random.choice(confidences)
        best_max_hands = random.choice(max_hands)
        
        for _ in range(self.max_iter):
            if random.random() < 0.5:  # Chain foraging
                best_confidence = min(0.9, best_confidence + 0.1)
            else:  # Cyclone foraging
                best_confidence = max(0.3, best_confidence - 0.1)
        
        return best_confidence, best_max_hands

class GestureRecognizer:
    """Custom gesture recognition using hand landmarks"""
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def _get_hand_angles(self, landmarks):
        """Calculate key angles between finger joints"""
        # Get vectors between key points
        vectors = []
        for i in range(21):  # 21 landmarks per hand
            vectors.append([landmarks[i].x, landmarks[i].y, landmarks[i].z])
        
        vectors = np.array(vectors)
        
        # Calculate angles between important joints
        thumb_angle = self._angle_between(
            vectors[2]-vectors[1], vectors[3]-vectors[2])
        index_angle = self._angle_between(
            vectors[6]-vectors[5], vectors[7]-vectors[6])
        middle_angle = self._angle_between(
            vectors[10]-vectors[9], vectors[11]-vectors[10])
        
        return thumb_angle, index_angle, middle_angle
    
    def _angle_between(self, v1, v2):
        """Returns the angle in degrees between vectors v1 and v2"""
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    
    def recognize_gesture(self, landmarks):
        """Recognize specific gestures from hand landmarks"""
        thumb_angle, index_angle, middle_angle = self._get_hand_angles(landmarks)
        
        # Thumbs up detection
        if thumb_angle < 30 and index_angle > 90 and middle_angle > 90:
            return "Thumbs Up"
        
        # Thumbs down detection
        if thumb_angle > 150 and index_angle > 90 and middle_angle > 90:
            return "Thumbs Down"
        
        # Victory sign (peace sign)
        if index_angle < 30 and middle_angle < 30 and thumb_angle > 90:
            return "Victory"
        
        # OK sign
        if thumb_angle < 30 and index_angle < 30 and middle_angle > 90:
            return "OK"
        
        # Pointing
        if index_angle < 30 and middle_angle > 90 and thumb_angle > 90:
            return "Pointing"
        
        return "Unknown Gesture"

def main():
    # Initialize optimizers
    aelm = AELM()
    mrfo = MRFO()
    best_confidence, best_max_hands = mrfo.optimize_parameters()
    selected_optimizer = aelm.evolve_optimizer()
    
    print(f"Optimized Parameters - Confidence: {best_confidence:.2f}, Max Hands: {best_max_hands}")
    print(f"Selected Optimizer: {selected_optimizer}")
    
    # Initialize gesture recognition
    gesture_recognizer = GestureRecognizer()
    cap = cv2.VideoCapture(0)
    
    # Variables for termination
    frames_without_gesture = 0
    last_detection_time = time.time()
    gesture_history = deque(maxlen=10)
    
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = gesture_recognizer.hands.process(image)
            
            # Process results
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            gesture = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    gesture_recognizer.mp_drawing.draw_landmarks(
                        image, hand_landmarks, gesture_recognizer.mp_hands.HAND_CONNECTIONS)
                    
                    # Recognize gesture
                    gesture = gesture_recognizer.recognize_gesture(hand_landmarks.landmark)
                    
                    # Display gesture
                    if gesture:
                        cv2.putText(image, f"Gesture: {gesture}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Update termination conditions
            if gesture and gesture != "Unknown Gesture":
                frames_without_gesture = 0
                last_detection_time = time.time()
                gesture_history.append(gesture)
            else:
                frames_without_gesture += 1
            
            # Show frame
            cv2.imshow('Gesture Recognition', image)
            
            # Check termination conditions
            current_time = time.time()
            if (frames_without_gesture >= MAX_FRAMES_WITHOUT_GESTURE or 
                (current_time - last_detection_time) > 30):
                print("\nTermination conditions met:")
                print(f"- {frames_without_gesture} consecutive frames without recognized gesture")
                print(f"- {current_time - last_detection_time:.1f} seconds since last detection")
                print("\nGesture statistics:")
                if gesture_history:
                    print(f"Most common gesture: {max(set(gesture_history), key=gesture_history.count)}")
                    print(f"All gestures detected: {list(gesture_history)}")
                break
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        gesture_recognizer.hands.close()

if __name__ == "__main__":
    main()
