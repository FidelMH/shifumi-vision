import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import mediapipe as mp

import numpy as np
import cv2
import mediapipe as mp
from collections import deque
import random
import time
import logging

# Suppress ScriptRunContext warnings from streamlit-webrtc
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

class ImprovedHandDetector(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Historique pour stabiliser la détection
        self.gesture_history = deque(maxlen=10)  # Garde les 10 derniers gestes

        # Variables d'etat pour le jeu
        self.state = "WAITING"  # WAITING ou LOCKED
        self.stable_frames_count = 0  # Compteur de frames stables
        self.last_stable_gesture = None  # Dernier geste stable detecte
        self.player_gesture = None  # Geste capture du joueur
        self.computer_gesture = None  # Choix de l'ordinateur
        self.frozen_frame = None  # Image figee
        self.game_result = None  # Resultat du jeu
        self.stable_threshold = 90  # 90 frames = 3 secondes a 30fps
        self.lock_timestamp = None  # Timestamp du moment ou le jeu est fige
        self.auto_reset_delay = 5.0  # Delai en secondes avant reset automatique
        self.ft5_end_frame = None  # Frame de fin de partie FT5
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # Si la partie FT5 est terminee, afficher l'ecran de fin
        if 'player_score' in st.session_state and 'computer_score' in st.session_state:
            if st.session_state.player_score >= 5 or st.session_state.computer_score >= 5:
                # Creer ou retourner l'ecran de fin FT5
                if self.ft5_end_frame is None:
                    self.ft5_end_frame = self.create_ft5_end_frame(img.copy(), w, h)
                return av.VideoFrame.from_ndarray(self.ft5_end_frame, format="bgr24")

        # Si le jeu est fige, verifier le timer pour reset automatique
        if self.state == "LOCKED" and self.frozen_frame is not None:
            # Verifier le timer pour reset automatique
            if self.lock_timestamp is not None:
                elapsed_time = time.time() - self.lock_timestamp
                if elapsed_time >= self.auto_reset_delay:
                    # Reset automatique apres 5 secondes
                    self.reset_game()
                else:
                    # Afficher la frame figee avec compte a rebours
                    display_frame = self.frozen_frame.copy()
                    remaining_time = int(self.auto_reset_delay - elapsed_time) + 1
                    cv2.putText(display_frame, f"Prochaine manche dans {remaining_time}s",
                               (w // 2 - 250, h - 50),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               1.2, (255, 255, 255), 3)
                    return av.VideoFrame.from_ndarray(display_frame, format="bgr24")
            else:
                return av.VideoFrame.from_ndarray(self.frozen_frame, format="bgr24")

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dessiner la main
                self.mp_drawing.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                # Detecter le geste ameliore
                gesture, confidence = self.detect_gesture_improved(hand_landmarks, w, h)
                self.gesture_history.append(gesture)

                # Geste stabilise (le plus frequent dans l'historique)
                stable_gesture = max(set(self.gesture_history), key=self.gesture_history.count)

                # Extraire le nom du geste sans emoji
                gesture_name = stable_gesture.split()[0] if stable_gesture else None

                # Logique de capture automatique
                if self.state == "WAITING" and gesture_name in ["PIERRE", "FEUILLE", "CISEAUX"] and confidence > 0.8:
                    if self.last_stable_gesture == gesture_name:
                        self.stable_frames_count += 1
                    else:
                        self.stable_frames_count = 0
                        self.last_stable_gesture = gesture_name

                    # Si stable pendant 3 secondes, capturer
                    if self.stable_frames_count >= self.stable_threshold:
                        self.player_gesture = gesture_name
                        self.computer_gesture = random.choice(["PIERRE", "FEUILLE", "CISEAUX"])
                        self.game_result = self.determine_winner(self.player_gesture, self.computer_gesture)
                        self.state = "LOCKED"
                        self.lock_timestamp = time.time()  # Enregistrer le moment du lock

                        # Creer la frame figee avec les resultats
                        self.frozen_frame = self.create_result_frame(img.copy(), w, h)

                        # Mettre a jour le score dans session_state
                        if 'player_score' not in st.session_state:
                            st.session_state.player_score = 0
                        if 'computer_score' not in st.session_state:
                            st.session_state.computer_score = 0

                        if self.game_result == "VICTOIRE":
                            st.session_state.player_score += 1
                        elif self.game_result == "DEFAITE":
                            st.session_state.computer_score += 1

                        return av.VideoFrame.from_ndarray(self.frozen_frame, format="bgr24")

                # Affichage en mode WAITING
                color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)

                cv2.putText(img, f"{stable_gesture}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                           2, color, 4)

                cv2.putText(img, f"Confiance: {confidence:.0%}",
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, (255, 255, 0), 2)

                # Afficher la progression de stabilite
                if self.last_stable_gesture and gesture_name == self.last_stable_gesture:
                    progress = int((self.stable_frames_count / self.stable_threshold) * 100)
                    cv2.putText(img, f"Stabilite: {progress}%",
                               (10, 160), cv2.FONT_HERSHEY_SIMPLEX,
                               0.8, (255, 255, 255), 2)
        else:
            cv2.putText(img, "Montrez votre main",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       1.2, (0, 0, 255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def detect_gesture_improved(self, landmarks, w, h):
        """Détection améliorée avec angles et distances"""
        
        # Convertir les landmarks en coordonnées pixels
        points = []
        for lm in landmarks.landmark:
            points.append([lm.x * w, lm.y * h])
        points = np.array(points)
        
        # Indices des points clés
        wrist = points[0]
        thumb_tip = points[4]
        index_tip = points[8]
        middle_tip = points[12]
        ring_tip = points[16]
        pinky_tip = points[20]
        
        # Compter les doigts levés avec angles
        fingers_up = self.count_fingers_improved(points)
        
        # Distance entre les doigts (pour ciseaux)
        index_middle_dist = np.linalg.norm(index_tip - middle_tip)
        
        # Vérifier si la main est fermée (pierre)
        palm_closed = self.is_palm_closed(points)
        
        # LOGIQUE DE DÉTECTION
        
        # PIERRE : Tous les doigts fermés
        if palm_closed or fingers_up == 0:
            return "PIERRE ✊", 0.95
        
        # CISEAUX : Exactement 2 doigts levés (index et majeur)
        elif fingers_up == 2:
            # Vérifier que c'est bien index et majeur
            index_up = self.is_finger_extended(points, 8, 6, 5)
            middle_up = self.is_finger_extended(points, 12, 10, 9)
            
            if index_up and middle_up and (index_middle_dist < 100 and index_middle_dist > 40) :
                return "CISEAUX ✌️", 0.9
            else:
                return "??? 🤔", 0.5
        
        # FEUILLE : 4 ou 5 doigts levés
        elif fingers_up >= 4:
            return "FEUILLE ✋", 0.9
        
        # Geste incertain
        else:
            return "??? 🤔", 0.3
    
    def count_fingers_improved(self, points):
        """Compte les doigts en vérifiant les angles"""
        fingers = 0
        
        # Pouce (logique spéciale car orientation différente)
        if self.is_thumb_extended(points):
            fingers += 1
        
        # Autres doigts : index, majeur, annulaire, auriculaire
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        finger_mcps = [5, 9, 13, 17]
        
        for tip, pip, mcp in zip(finger_tips, finger_pips, finger_mcps):
            if self.is_finger_extended(points, tip, pip, mcp):
                fingers += 1
        
        return fingers
    
    def is_finger_extended(self, points, tip_id, pip_id, mcp_id):
        """Vérifie si un doigt est levé en calculant l'angle"""
        tip = points[tip_id]
        pip = points[pip_id]
        mcp = points[mcp_id]
        
        # Le bout du doigt doit être au-dessus du PIP
        # ET la distance tip-MCP doit être grande
        distance_tip_mcp = np.linalg.norm(tip - mcp)
        distance_pip_mcp = np.linalg.norm(pip - mcp)
        
        return tip[1] < pip[1] and distance_tip_mcp > distance_pip_mcp * 1.2
    
    def is_thumb_extended(self, points):
        """Détection spéciale pour le pouce"""
        thumb_tip = points[4]
        thumb_mcp = points[2]
        index_mcp = points[5]
        
        # Distance pouce-index
        distance = np.linalg.norm(thumb_tip - index_mcp)
        
        return distance > 50  # Seuil à ajuster
    
    def is_palm_closed(self, points):
        """Vérifie si la paume est fermée (pierre)"""
        wrist = points[0]
        finger_tips = [4, 8, 12, 16, 20]

        # Toutes les pointes doivent être proches du poignet
        distances = [np.linalg.norm(points[tip] - wrist) for tip in finger_tips]
        avg_distance = np.mean(distances)

        return avg_distance < 120  # Seuil à ajuster selon la taille de main

    def determine_winner(self, player, computer):
        """Determine le gagnant de la partie"""
        if player == computer:
            return "EGALITE"

        winning_combinations = {
            ("PIERRE", "CISEAUX"): "VICTOIRE",
            ("CISEAUX", "FEUILLE"): "VICTOIRE",
            ("FEUILLE", "PIERRE"): "VICTOIRE"
        }

        result = winning_combinations.get((player, computer))
        if result:
            return result
        else:
            return "DEFAITE"

    def create_result_frame(self, img, w, h):
        """Cree la frame avec les resultats affiches"""
        # Creer un overlay semi-transparent
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

        # Couleurs selon le resultat
        if self.game_result == "VICTOIRE":
            result_color = (0, 255, 0)  # Vert
            result_text = "VOUS GAGNEZ !"
        elif self.game_result == "DEFAITE":
            result_color = (0, 0, 255)  # Rouge
            result_text = "VOUS PERDEZ !"
        else:
            result_color = (255, 255, 0)  # Jaune
            result_text = "EGALITE !"

        # Afficher le resultat principal
        cv2.putText(img, result_text,
                   (w // 2 - 250, h // 2 - 100),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   2, result_color, 5)

        # Afficher le geste du joueur
        cv2.putText(img, f"Vous: {self.player_gesture}",
                   (w // 2 - 200, h // 2),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1.5, (255, 255, 255), 3)

        # Afficher le geste de l'ordinateur
        cv2.putText(img, f"Ordi: {self.computer_gesture}",
                   (w // 2 - 200, h // 2 + 60),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1.5, (255, 255, 255), 3)

        # Afficher le score actuel FT5 en haut
        if 'player_score' in st.session_state and 'computer_score' in st.session_state:
            cv2.putText(img, f"Score: {st.session_state.player_score} - {st.session_state.computer_score}",
                       (w // 2 - 150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1.5, (255, 255, 255), 3)

        return img

    def create_ft5_end_frame(self, img, w, h):
        """Cree l'ecran de fin de partie FT5"""
        # Creer un overlay noir complet
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

        # Determiner le gagnant de la partie FT5
        if 'player_score' in st.session_state and 'computer_score' in st.session_state:
            if st.session_state.player_score >= 5:
                result_color = (0, 255, 0)  # Vert
                result_text = "VICTOIRE FT5 !"
                sub_text = "Felicitations !"
            else:
                result_color = (0, 0, 255)  # Rouge
                result_text = "DEFAITE FT5"
                sub_text = "L'ordinateur gagne..."

            # Afficher le resultat principal
            cv2.putText(img, result_text,
                       (w // 2 - 300, h // 2 - 100),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       2.5, result_color, 6)

            cv2.putText(img, sub_text,
                       (w // 2 - 200, h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1.5, (255, 255, 255), 3)

            # Afficher le score final
            cv2.putText(img, f"Score final: {st.session_state.player_score} - {st.session_state.computer_score}",
                       (w // 2 - 250, h // 2 + 80),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1.3, (255, 255, 255), 3)

            # Message pour rejouer
            cv2.putText(img, "Cliquez sur 'Nouvelle Partie' pour rejouer",
                       (w // 2 - 380, h - 50),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1, (255, 255, 0), 2)

        return img

    def reset_game(self):
        """Reinitialise l'etat du jeu pour une nouvelle manche"""
        self.state = "WAITING"
        self.stable_frames_count = 0
        self.last_stable_gesture = None
        self.player_gesture = None
        self.computer_gesture = None
        self.frozen_frame = None
        self.game_result = None
        self.lock_timestamp = None
        self.ft5_end_frame = None

# Interface Streamlit
st.title("Pierre-Feuille-Ciseaux - First to 5")

# Initialiser les scores dans session_state
if 'player_score' not in st.session_state:
    st.session_state.player_score = 0
if 'computer_score' not in st.session_state:
    st.session_state.computer_score = 0
if 'game_counter' not in st.session_state:
    st.session_state.game_counter = 0

# Afficher le score
col1, col2 = st.columns(2)
with col1:
    st.metric("Votre score", st.session_state.player_score)
with col2:
    st.metric("Score ordinateur", st.session_state.computer_score)

st.write("---")

# Verifier si la partie FT5 est terminee
ft5_finished = st.session_state.player_score >= 5 or st.session_state.computer_score >= 5

# Instructions
if ft5_finished:
    st.success("La partie est terminee ! Cliquez sur 'Nouvelle Partie' ci-dessous pour recommencer.")
else:
    st.markdown("""
    ### Comment jouer :
    1. Montrez votre geste a la camera (Pierre, Feuille ou Ciseaux)
    2. Maintenez le geste stable pendant 3 secondes
    3. Le jeu se declenchera automatiquement
    4. Apres chaque manche, le jeu reprend automatiquement apres 5 secondes
    5. Premier a 5 victoires gagne la partie !
    """)

st.write("---")

# Stream video
ctx = webrtc_streamer(key=f"hand-game-{st.session_state.game_counter}", video_processor_factory=ImprovedHandDetector)

st.write("---")

# Bouton Nouvelle Partie (seulement visible si FT5 termine)
if ft5_finished:
    if st.button("Nouvelle Partie", type="primary", use_container_width=True, key="new_game_btn"):
        st.session_state.player_score = 0
        st.session_state.computer_score = 0
        st.session_state.game_counter += 1
        st.rerun()