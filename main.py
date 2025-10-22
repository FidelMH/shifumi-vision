import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import mediapipe as mp

class HandDetector(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dessiner les points de la main
                self.mp_drawing.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # Détecter le geste
                gesture = self.detect_gesture(hand_landmarks)
                
                # AFFICHER LE TEXTE
                # Titre principal
                cv2.putText(img, f"Geste: {gesture}", 
                           (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1.5, (0, 255, 0), 3)
                
                # Score de confiance
                # cv2.putText(img, "Confiance: 95%", 
                #            (10, 100), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 
                #            0.8, (255, 255, 0), 2)
        else:
            # Message si pas de main détectée
            cv2.putText(img, "Montrez votre main", 
                       (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1.2, (0, 0, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def detect_gesture(self, landmarks):
        # Compter les doigts levés
        fingers_up = self.count_fingers(landmarks)
        
        if fingers_up == 0:
            return "PIERRE ✊"
        elif fingers_up == 2:
            return "CISEAUX ✌️"
        elif fingers_up >= 4:
            return "FEUILLE ✋"
        else:
            return "???"
    
    def count_fingers(self, landmarks):
        # Logique simplifiée pour compter les doigts
        # Index des points : pouce=4, index=8, majeur=12, annulaire=16, auriculaire=20
        finger_tips = [4, 8, 12, 16, 20]
        finger_bases = [2, 6, 10, 14, 18]
        
        count = 0
        for tip, base in zip(finger_tips, finger_bases):
            if landmarks.landmark[tip].y < landmarks.landmark[base].y:
                count += 1
        
        return count

st.title("🎮 Pierre-Feuille-Ciseaux")
webrtc_streamer(key="hand-game", video_processor_factory=HandDetector)