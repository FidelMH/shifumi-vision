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
        self.current_stable_gesture = None  # Geste stable actuel (avec hysteresis)
        self.current_handedness = None  # Store current hand laterality (Left/Right)

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
            # Store handedness information from MediaPipe
            # Note: MediaPipe's "Left" and "Right" are from the perspective of the person in the image
            if results.multi_handedness:
                self.current_handedness = results.multi_handedness[0].classification[0].label
            else:
                self.current_handedness = None

            # Extract MediaPipe hand detection confidence
            # This represents MediaPipe's confidence in detecting the hand
            mediapipe_confidence = 0.0
            if results.multi_handedness:
                # Get classification score (confidence) from MediaPipe
                # This is a value between 0 and 1 indicating hand detection quality
                mediapipe_confidence = results.multi_handedness[0].classification[0].score

            for hand_landmarks in results.multi_hand_landmarks:
                # Dessiner la main
                self.mp_drawing.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                # Detecter le geste ameliore with MediaPipe confidence
                gesture, confidence = self.detect_gesture_improved(hand_landmarks, w, h, mediapipe_confidence)
                self.gesture_history.append(gesture)

                # Geste stabilise avec hysteresis (requiert 6 frames sur 8 pour changer)
                # Seules les frames avec confiance >= 0.6 sont prises en compte
                stable_gesture = self.update_stable_gesture(gesture, confidence)

                # Si pas encore de geste stable, utiliser le geste actuel
                if stable_gesture is None:
                    stable_gesture = gesture

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

    def update_stable_gesture(self, new_gesture, confidence):
        """
        Mise a jour du geste stable avec hysteresis.
        Un nouveau geste doit etre present pour au moins 6 frames sur les 8 dernieres
        avant d'etre accepte comme nouveau geste stable.
        Les frames avec confiance < 0.6 sont ignorees dans les calculs de stabilite.
        """
        # Ignorer les frames avec confiance trop faible
        if confidence < 0.6:
            return self.current_stable_gesture

        # Analyser les 8 dernieres frames de l'historique
        recent_gestures = list(self.gesture_history)[-8:]

        # Compter les occurrences du nouveau geste dans les frames recentes
        # Un geste doit apparaitre au moins 6 fois sur 8 pour etre considere stable
        if recent_gestures.count(new_gesture) >= 6:
            self.current_stable_gesture = new_gesture

        return self.current_stable_gesture

    def calculate_hand_size(self, points):
        """
        Calculate reference hand size for normalization.
        Uses distance from wrist (landmark 0) to middle finger MCP (landmark 9).
        This provides a stable reference that doesn't change with finger positions.
        """
        wrist = points[0]
        middle_mcp = points[9]
        hand_size = np.linalg.norm(middle_mcp - wrist)
        return hand_size

    def calculate_gesture_quality(self, gesture_type, points, hand_size, fingers_up):
        """
        Calculate gesture quality score (0-1) based on how well the detected gesture
        matches ideal characteristics for that gesture type.

        Args:
            gesture_type: Gesture name without emoji ("PIERRE", "FEUILLE", "CISEAUX", or "???")
            points: Array of hand landmark points
            hand_size: Normalized hand size for distance calculations
            fingers_up: Number of fingers detected as extended

        Returns:
            Float between 0 and 1 indicating gesture quality
        """
        wrist = points[0]
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

        if gesture_type == "PIERRE":
            # For ROCK: all fingertips should be close to wrist
            # Calculate average distance of all fingertips to wrist
            distances = [np.linalg.norm(points[tip] - wrist) for tip in finger_tips]
            avg_distance = np.mean(distances)

            # Quality is inversely proportional to distance
            # Perfect rock: avg_distance < 0.30 * hand_size -> quality = 1.0
            # Poor rock: avg_distance > 0.50 * hand_size -> quality = 0.0
            normalized_distance = avg_distance / hand_size
            quality = max(0.0, min(1.0, (0.50 - normalized_distance) / 0.20))
            return quality

        elif gesture_type == "FEUILLE":
            # For PAPER: all fingers should be properly extended
            # Check that all 5 fingers are extended and spread out
            if fingers_up < 4:
                return 0.3  # Not enough fingers extended

            # Check finger spread - fingers should be apart
            index_tip = points[8]
            middle_tip = points[12]
            ring_tip = points[16]
            pinky_tip = points[20]

            # Calculate average distance between adjacent fingers
            spreads = [
                np.linalg.norm(middle_tip - index_tip),
                np.linalg.norm(ring_tip - middle_tip),
                np.linalg.norm(pinky_tip - ring_tip)
            ]
            avg_spread = np.mean(spreads)

            # Quality based on finger spread
            # Good spread: > 0.15 * hand_size -> quality = 1.0
            # Poor spread: < 0.08 * hand_size -> quality = 0.5
            normalized_spread = avg_spread / hand_size
            spread_quality = max(0.5, min(1.0, (normalized_spread - 0.08) / 0.07 * 0.5 + 0.5))

            # Bonus if all 5 fingers are up
            finger_bonus = 1.0 if fingers_up == 5 else 0.9

            return spread_quality * finger_bonus

        elif gesture_type == "CISEAUX":
            # For SCISSORS: exactly index and middle extended, others closed
            if fingers_up != 2:
                return 0.4  # Wrong number of fingers

            # Verify it's index and middle specifically
            index_up = self.is_finger_extended(points, 8, 6, 5)
            middle_up = self.is_finger_extended(points, 12, 10, 9)

            if not (index_up and middle_up):
                return 0.4  # Wrong fingers extended

            # Check that other fingers are closed
            ring_up = self.is_finger_extended(points, 16, 14, 13)
            pinky_up = self.is_finger_extended(points, 20, 18, 17)

            if ring_up or pinky_up:
                return 0.6  # Other fingers should be closed

            # Check distance between index and middle fingers
            index_tip = points[8]
            middle_tip = points[12]
            distance = np.linalg.norm(index_tip - middle_tip)
            normalized_distance = distance / hand_size

            # Quality based on finger separation
            # Ideal separation: 0.20-0.30 * hand_size -> quality = 1.0
            # Too close or too far apart -> lower quality
            if 0.15 <= normalized_distance <= 0.35:
                return 1.0
            elif normalized_distance < 0.15:
                # Fingers too close
                return max(0.7, normalized_distance / 0.15 * 0.3 + 0.7)
            else:
                # Fingers too far apart
                return max(0.7, 1.0 - (normalized_distance - 0.35) / 0.20 * 0.3)

        else:
            # Unknown gesture - low quality
            return 0.3

    def calculate_temporal_stability(self):
        """
        Calculate temporal stability score (0-1) based on gesture history.
        Measures how consistent the gesture has been over recent frames.

        Returns:
            Float between 0 and 1 indicating temporal stability
            1.0 = gesture is very stable (same gesture in most recent frames)
            0.0 = gesture is unstable (rapidly changing)
        """
        if len(self.gesture_history) == 0:
            return 0.0

        # Analyze the gesture history (up to last 10 frames)
        recent_gestures = list(self.gesture_history)

        if len(recent_gestures) < 3:
            # Not enough history yet
            return 0.5

        # Count occurrences of each gesture in history
        from collections import Counter
        gesture_counts = Counter(recent_gestures)

        # Get the most common gesture and its count
        most_common_gesture, most_common_count = gesture_counts.most_common(1)[0]

        # Calculate stability as ratio of most common gesture
        stability = most_common_count / len(recent_gestures)

        return stability

    def detect_gesture_improved(self, landmarks, w, h, mediapipe_confidence=0.0):
        """Détection améliorée avec angles et distances"""

        # Convertir les landmarks en coordonnées pixels
        points = []
        for lm in landmarks.landmark:
            points.append([lm.x * w, lm.y * h])
        points = np.array(points)

        # Calculate reference hand size for normalization
        # This makes all distance thresholds relative to hand size
        hand_size = self.calculate_hand_size(points)

        # Indices des points clés
        wrist = points[0]
        thumb_tip = points[4]
        index_tip = points[8]
        middle_tip = points[12]
        ring_tip = points[16]
        pinky_tip = points[20]

        # Compter les doigts levés avec angles
        fingers_up = self.count_fingers_improved(points, hand_size)

        # Distance entre les doigts (pour ciseaux)
        index_middle_dist = np.linalg.norm(index_tip - middle_tip)

        # Vérifier si la main est fermée (pierre)
        palm_closed = self.is_palm_closed(points, hand_size)
        
        # LOGIQUE DE DÉTECTION

        # Determine gesture type first
        gesture_type = None
        gesture_emoji = ""

        # PIERRE : Tous les doigts fermés
        if palm_closed or fingers_up == 0:
            gesture_type = "PIERRE"
            gesture_emoji = " ✊"

        # CISEAUX : Exactement 2 doigts levés (index et majeur)
        elif fingers_up == 2:
            # Verify that index and middle fingers are extended
            index_up = self.is_finger_extended(points, 8, 6, 5)
            middle_up = self.is_finger_extended(points, 12, 10, 9)

            # Explicitly verify that ring finger and pinky are NOT extended
            ring_up = self.is_finger_extended(points, 16, 14, 13)
            pinky_up = self.is_finger_extended(points, 20, 18, 17)

            # Verify that thumb is not prominently extended
            thumb_up = self.is_thumb_extended(points, hand_size, self.current_handedness)

            # Calculate the V-angle between index and middle fingers
            # Vector from index MCP to index tip
            index_mcp = points[5]
            index_vector = index_tip - index_mcp

            # Vector from middle MCP to middle tip
            middle_mcp = points[9]
            middle_vector = middle_tip - middle_mcp

            # Calculate angle between the two vectors
            norm_index = np.linalg.norm(index_vector)
            norm_middle = np.linalg.norm(middle_vector)

            # Safety check: avoid division by zero
            if norm_index > 0 and norm_middle > 0:
                dot_product = np.dot(index_vector, middle_vector)
                cos_angle = dot_product / (norm_index * norm_middle)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)
            else:
                angle_deg = 0

            # Normalized threshold: distance between fingers should be less than 30% of hand size
            # Previously was hardcoded to 100 pixels
            # All conditions for proper scissors gesture:
            # 1. Index and middle fingers extended
            # 2. Ring finger and pinky NOT extended
            # 3. Thumb not prominently extended
            # 4. V-angle between index and middle > 15 degrees
            # 5. Distance between fingertips within reasonable range
            if (index_up and middle_up and
                not ring_up and not pinky_up and
                not thumb_up and
                angle_deg > 15 and
                index_middle_dist < hand_size * 0.3):
                gesture_type = "CISEAUX"
                gesture_emoji = " ✌️"
            else:
                gesture_type = "???"
                gesture_emoji = " 🤔"

        # FEUILLE : 4 ou 5 doigts levés
        elif fingers_up >= 4:
            gesture_type = "FEUILLE"
            gesture_emoji = " ✋"

        # Geste incertain
        else:
            gesture_type = "???"
            gesture_emoji = " 🤔"

        # Calculate multi-criteria confidence score
        # Formula: confidence = 0.4 * mediapipe + 0.4 * quality + 0.2 * stability

        # Component 1: MediaPipe hand detection confidence (0-1)
        # This comes from MediaPipe's hand tracking confidence
        mediapipe_score = mediapipe_confidence

        # Component 2: Gesture quality score (0-1)
        # Measures how well the detected gesture matches ideal characteristics
        gesture_quality_score = self.calculate_gesture_quality(
            gesture_type, points, hand_size, fingers_up
        )

        # Component 3: Temporal stability score (0-1)
        # Measures consistency of gesture over recent frames
        temporal_stability_score = self.calculate_temporal_stability()

        # Combine scores using weighted formula
        # 40% MediaPipe confidence + 40% gesture quality + 20% temporal stability
        final_confidence = (
            0.4 * mediapipe_score +
            0.4 * gesture_quality_score +
            0.2 * temporal_stability_score
        )

        # Clamp confidence to valid range [0, 1]
        final_confidence = max(0.0, min(1.0, final_confidence))

        return f"{gesture_type}{gesture_emoji}", final_confidence
    
    def count_fingers_improved(self, points, hand_size):
        """Compte les doigts en vérifiant les angles"""
        fingers = 0

        # Pouce (logique spéciale car orientation différente)
        # Pass handedness information for laterality-aware detection
        if self.is_thumb_extended(points, hand_size, self.current_handedness):
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
        """Verifie si un doigt est leve en calculant l'angle reel entre les segments

        Calcule l'angle entre le segment MCP->PIP et le segment PIP->TIP.
        Un doigt est considere etendu si l'angle est > 160 degres (doigt droit).

        Args:
            points: Array de points de la main
            tip_id: Index du point TIP (bout du doigt)
            pip_id: Index du point PIP (articulation intermediaire)
            mcp_id: Index du point MCP (articulation a la base)

        Returns:
            True si le doigt est etendu (angle > 160 degres), False sinon
        """
        tip = points[tip_id]
        pip = points[pip_id]
        mcp = points[mcp_id]

        # Calculer les vecteurs entre les articulations
        # v1: vecteur de MCP vers PIP
        v1 = pip - mcp
        # v2: vecteur de PIP vers TIP
        v2 = tip - pip

        # Calculer les normes (longueurs) des vecteurs
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        # Verification de securite: eviter la division par zero
        if norm_v1 == 0 or norm_v2 == 0:
            return False

        # Calculer le produit scalaire
        dot_product = np.dot(v1, v2)

        # Calculer le cosinus de l'angle
        cos_angle = dot_product / (norm_v1 * norm_v2)

        # Limiter cos_angle a [-1, 1] pour eviter les erreurs de domaine dans arccos
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Calculer l'angle en radians puis convertir en degres
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        # Un doigt est etendu si l'angle est superieur a 160 degres
        return angle_deg > 160
    
    def is_thumb_extended(self, points, hand_size, handedness):
        """
        Precise thumb detection considering hand laterality (left vs right hand).

        Checks:
        1. Lateral position: thumb should be on the correct side based on handedness
           - Right hand: thumb_tip.x > index_mcp.x (thumb is to the right)
           - Left hand: thumb_tip.x < index_mcp.x (thumb is to the left)
        2. Distance check: thumb must be sufficiently far from index MCP
        3. Wrist distance: thumb tip should be away from wrist (not folded in)

        For rock/fist detection, this method returns False when thumb is folded
        on other fingers (thumb_tip close to middle_pip or wrist).

        Args:
            points: Array of hand landmark points
            hand_size: Reference hand size for normalization
            handedness: "Left" or "Right" from MediaPipe (perspective of person in image)

        Returns:
            True if thumb is extended, False if folded
        """
        thumb_tip = points[4]
        thumb_ip = points[3]
        thumb_mcp = points[2]
        index_mcp = points[5]
        middle_pip = points[10]
        wrist = points[0]

        # Check if thumb is folded on other fingers (for rock detection)
        # If thumb tip is very close to middle PIP, it's folded in a fist
        thumb_to_middle_pip = np.linalg.norm(thumb_tip - middle_pip)
        if thumb_to_middle_pip < hand_size * 0.2:
            return False

        # Distance check: thumb must be sufficiently far from index MCP
        # Uses normalized threshold of 15% of hand size
        distance_to_index = np.linalg.norm(thumb_tip - index_mcp)
        if distance_to_index < hand_size * 0.15:
            return False

        # Check distance from wrist: thumb should be extended away from wrist
        # If thumb is close to wrist, it's likely folded
        thumb_to_wrist = np.linalg.norm(thumb_tip - wrist)
        if thumb_to_wrist < hand_size * 0.25:
            return False

        # Laterality-based detection: check if thumb is on the correct side
        # This is the key improvement that accounts for left vs right hand
        if handedness == "Right":
            # For right hand: thumb should be to the right of index MCP
            # thumb_tip.x > index_mcp.x means thumb is extended to the right
            lateral_check = thumb_tip[0] > index_mcp[0]
        elif handedness == "Left":
            # For left hand: thumb should be to the left of index MCP
            # thumb_tip.x < index_mcp.x means thumb is extended to the left
            lateral_check = thumb_tip[0] < index_mcp[0]
        else:
            # Fallback: if handedness is unknown, use distance-only detection
            # This maintains backwards compatibility
            lateral_check = True

        # Thumb is extended if it passes both distance and lateral checks
        return lateral_check and distance_to_index > hand_size * 0.15
    
    def is_palm_closed(self, points, hand_size):
        """
        Vérifie si la paume est fermée (pierre).
        Uses normalized threshold relative to hand size.
        """
        wrist = points[0]
        finger_tips = [4, 8, 12, 16, 20]

        # Toutes les pointes doivent être proches du poignet
        distances = [np.linalg.norm(points[tip] - wrist) for tip in finger_tips]
        avg_distance = np.mean(distances)

        # Normalized threshold: average fingertip distance should be less than 36% of hand size
        # Previously was hardcoded to 120 pixels
        # The ratio 0.36 was calculated from: 120 / typical_hand_size (approximately 330 pixels)
        return avg_distance < hand_size * 0.36

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
ctx = webrtc_streamer(key="hand-game", video_processor_factory=ImprovedHandDetector)

st.write("---")

# Bouton Nouvelle Partie (seulement visible si FT5 termine)
if ft5_finished:
    if st.button("Nouvelle Partie", type="primary", use_container_width=True, key="new_game_btn"):
        st.session_state.player_score = 0
        st.session_state.computer_score = 0
        st.rerun()