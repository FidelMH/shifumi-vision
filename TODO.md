# TODO - Ameliorations de la Detection de Gestes

Ce document liste les ameliorations a apporter au systeme de detection de gestes pour le jeu Pierre-Feuille-Ciseaux.

---

## Ameliorations Prioritaires (Top 3)

1. **Normalisation par taille de main** - Resout les problemes de distance variable
2. **Calcul d'angles reels** - Remplace les comparaisons de coordonnees Y
3. **Stabilisation temporelle stricte** - Systeme d'etat pour eviter les changements de geste erratiques

---

## 1. Normalisation des distances par rapport a la taille de la main

### Probleme actuel
Les seuils sont fixes en pixels (100 pixels pour les ciseaux, 50 pour le pouce, 120 pour la paume fermee). Si l'utilisateur eloigne ou rapproche sa main de la camera, ces seuils ne sont plus adaptes.

### Solution proposee
- Calculer une "taille de reference" de la main (ex: distance poignet-majeur ou distance poignet-index MCP)
- Exprimer tous les seuils comme des ratios de cette taille de reference
- Exemple: au lieu de `distance < 100`, utiliser `distance < hand_size * 0.3`

### Fichiers a modifier
- `main.py:156` - Methode `is_palm_closed`
- `main.py:154` - Methode `is_thumb_extended`
- `main.py:100` - Detection des ciseaux

### Impact
Haute priorite - Permet la detection quelle que soit la distance camera-main

---

## 2. Amelioration de la detection des doigts leves

### Probleme actuel
La methode `is_finger_extended` (ligne 132) verifie seulement si `tip[1] < pip[1]` (coordonnee Y) et un ratio de distances. Cela ne fonctionne que si la main est orientee normalement (paume face camera, doigts vers le haut).

### Solution proposee
- Calculer l'angle reel entre les segments (MCP→PIP→TIP) en utilisant le produit scalaire
- Considerer qu'un doigt est leve si l'angle est superieur a 160 degres (doigt tendu)
- Formule: `angle = arccos((v1 · v2) / (|v1| * |v2|))`
  - `v1 = PIP - MCP`
  - `v2 = TIP - PIP`

### Fichiers a modifier
- `main.py:132-143` - Methode `is_finger_extended`

### Impact
Moyenne priorite - Ameliore la robustesse face aux variations d'orientation

---

## 3. Detection des ciseaux plus robuste

### Probleme actuel
Le code verifie juste que 2 doigts sont leves et que la distance entre index et majeur est < 100 pixels, mais ne verifie pas que ce sont vraiment des ciseaux "ouverts".

### Solution proposee
- Verifier explicitement que l'annulaire et l'auriculaire sont bien fermes
- Calculer l'angle d'ouverture entre index et majeur (ils doivent etre ecartes en V, angle > 15 degres)
- Normaliser la distance par la taille de la main
- Verifier que le pouce est ferme ou neutre

### Fichiers a modifier
- `main.py:95-103` - Logique de detection des ciseaux dans `detect_gesture_improved`

### Impact
Moyenne priorite - Reduit les faux positifs

---

## 4. Meilleure stabilisation temporelle

### Probleme actuel
Utilise juste le geste le plus frequent sur 10 frames (`max(set(self.gesture_history), key=...)`). Si on alterne entre 2 gestes (5 frames de chaque), ca peut etre instable.

### Solution proposee
Implementer un systeme d'etat avec hysteresis:
- Exiger qu'un nouveau geste soit present pendant au moins 7-8 frames consecutives sur 10
- Ajouter un attribut `self.current_stable_gesture` qui ne change que si le nouveau geste est vraiment stable
- Ignorer les frames avec confiance < 0.6 dans le calcul de stabilite
- Possibilite d'ajouter un filtre de Kalman pour lisser les transitions

### Pseudocode
```python
def update_stable_gesture(self, new_gesture, confidence):
    if confidence < 0.6:
        return self.current_stable_gesture

    recent_gestures = list(self.gesture_history)[-8:]
    if recent_gestures.count(new_gesture) >= 6:
        self.current_stable_gesture = new_gesture

    return self.current_stable_gesture
```

### Fichiers a modifier
- `main.py:40-43` - Logique de stabilisation dans `recv`
- `main.py:23` - Ajouter `self.current_stable_gesture = None` dans `__init__`

### Impact
Haute priorite - Ameliore considerablement l'experience utilisateur

---

## 5. Calibration initiale

### Probleme actuel
Pas d'adaptation aux differentes morphologies de mains ou configurations de camera.

### Solution proposee
Ajouter une phase de calibration au demarrage:
- Demander a l'utilisateur de montrer chaque geste (pierre, feuille, ciseaux) pendant 3 secondes
- Calculer les caracteristiques moyennes de chaque geste pour cet utilisateur
- Ajuster automatiquement les seuils en fonction de ces mesures
- Sauvegarder les parametres de calibration pour les sessions futures (fichier JSON)

### Fichiers a modifier
- Nouveau fichier: `calibration.py`
- `main.py` - Ajouter une etape de calibration avant `webrtc_streamer`

### Impact
Basse priorite - Amelioration significative mais necessite refactoring

---

## 6. Verification de l'orientation de la main

### Probleme actuel
Si la main est tournee (paume vers le bas, main inclinee), la detection basee sur les coordonnees Y echoue.

### Solution proposee
- Calculer le vecteur normal de la paume avec les landmarks 0 (poignet), 5 (index MCP), 17 (auriculaire MCP)
- Utiliser la coordonnee Z de MediaPipe pour verifier la profondeur
- S'assurer que la paume est face a la camera (angle normal < 45 degres)
- Afficher un avertissement "Tournez votre main face a la camera" si mauvaise orientation

### Fichiers a modifier
- `main.py:25-60` - Methode `recv`, ajouter verification d'orientation

### Impact
Moyenne priorite - Ameliore la robustesse et guide l'utilisateur

---

## 7. Zone de detection definie

### Probleme actuel
La main peut etre n'importe ou dans le cadre, ce qui cause des variations de distance et d'eclairage.

### Solution proposee
- Definir une zone rectangulaire au centre de l'ecran (par exemple 40% de largeur, 60% de hauteur)
- Dessiner cette zone sur l'image
- N'accepter les gestes que si le poignet (landmark 0) est dans cette zone
- Afficher un message guide "Placez votre main dans la zone" sinon

### Fichiers a modifier
- `main.py:25-60` - Methode `recv`, ajouter verification de zone

### Impact
Basse priorite - Ameliore la consistance mais contraint l'utilisateur

---

## 8. Validation par confiance multi-criteres

### Probleme actuel
La confiance est calculee de maniere arbitraire (0.95 pour pierre, 0.9 pour ciseaux/feuille, 0.5 sinon).

### Solution proposee
Calculer une vraie confiance basee sur plusieurs facteurs:
- Confiance MediaPipe de detection de la main (`results.multi_hand_landmarks[0].landmark[0].visibility`)
- Ecart par rapport aux caracteristiques "ideales" du geste (ex: pour pierre, toutes les distances doigts-poignet doivent etre petites)
- Stabilite temporelle (variance du geste dans l'historique)
- Score final = moyenne ponderee de ces 3 metriques

### Formule proposee
```python
confidence = (
    0.4 * mediapipe_confidence +
    0.4 * gesture_quality_score +
    0.2 * temporal_stability_score
)
```

### Fichiers a modifier
- `main.py:62-111` - Methode `detect_gesture_improved`

### Impact
Moyenne priorite - Donne une meilleure indication de fiabilite

---

## 9. Gestion du pouce plus precise

### Probleme actuel
`is_thumb_extended` (ligne 145) utilise juste une distance > 50 pixels, ce qui est tres approximatif.

### Solution proposee
- Verifier la position relative du pouce par rapport a l'index ET au poignet
- Tenir compte de la lateralite (main gauche vs droite, detectable via `results.multi_handedness`)
- Pour main droite: pouce leve = thumb_tip.x > index_mcp.x
- Pour main gauche: pouce leve = thumb_tip.x < index_mcp.x
- Pour la pierre, verifier que le pouce est replie SUR les autres doigts (thumb_tip proche de middle_pip)

### Fichiers a modifier
- `main.py:145-154` - Methode `is_thumb_extended`
- `main.py:12-20` - Stocker `multi_handedness` dans `__init__` ou `recv`

### Impact
Moyenne priorite - Ameliore la precision pour pierre vs feuille

---

## 10. Detection de deux mains pour un vrai jeu

### Observation
Le code a `max_num_hands=1` (ligne 17) mais l'historique git montre que c'etait a 2 avant (commit d377d7f).

### Solution proposee pour un vrai jeu
- Passer `max_num_hands=2`
- Detecter 2 mains simultanement (joueur 1 vs joueur 2, ou joueur vs IA)
- Identifier quelle main appartient a quel joueur par position (gauche/droite de l'ecran)
- Implementer un compte a rebours (3-2-1-Go!)
- Valider les gestes simultanement apres le compte a rebours
- Determiner le gagnant et afficher le resultat
- Optionnel: Ajouter un mode IA ou l'ordinateur joue aleatoirement

### Fichiers a modifier
- `main.py:17` - Changer `max_num_hands=2`
- `main.py:31-60` - Modifier la boucle pour traiter 2 mains
- Nouveau fichier: `game_logic.py` - Logique de jeu et regles

### Impact
Haute priorite si l'objectif est un vrai jeu jouable - Transforme le POC en application complete

---

## Ordre d'implementation recommande

1. **Phase 1 - Robustesse de base**
   - Normalisation par taille de main (Amelioration 1)
   - Stabilisation temporelle stricte (Amelioration 4)
   - Calcul d'angles reels (Amelioration 2)

2. **Phase 2 - Precision des gestes**
   - Detection des ciseaux robuste (Amelioration 3)
   - Gestion du pouce precise (Amelioration 9)
   - Confiance multi-criteres (Amelioration 8)

3. **Phase 3 - Experience utilisateur**
   - Verification orientation (Amelioration 6)
   - Zone de detection (Amelioration 7)

4. **Phase 4 - Fonctionnalites avancees**
   - Detection deux mains + logique de jeu (Amelioration 10)
   - Calibration initiale (Amelioration 5)

---

## Notes techniques

### Ressources utiles
- Documentation MediaPipe Hands: https://google.github.io/mediapipe/solutions/hands.html
- Landmarks de la main: 21 points indices (0=poignet, 4=pouce, 8=index, 12=majeur, 16=annulaire, 20=auriculaire)
- Calcul d'angles en Python: `np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))`

### Tests a effectuer apres chaque amelioration
- Tester avec differentes distances camera-main
- Tester avec differentes orientations de main
- Tester avec differents types de mains (grande/petite)
- Tester avec differentes conditions d'eclairage
- Mesurer le taux de faux positifs/negatifs pour chaque geste
