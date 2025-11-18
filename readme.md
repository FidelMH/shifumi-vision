# Shifumi Vision

Jeu de pierre-feuille-ciseaux avec reconnaissance de gestes par caméra.

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run main.py
```

## Retour d'expérience

**Problèmes rencontrés :**
- La détection des ciseaux est imprécise
- Problème avec la gestion de la partie

**Améliorations possibles :**
- Créer un modèle personnalisé pour la reconnaissance des gestes (pierre, feuille, ciseaux)
- Entraîner le modèle avec des photos de mains réelles effectuant les gestes