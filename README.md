# 🧬 Hyperparameter Evolution with YOLOv8

Ce projet applique un algorithme génétique pour optimiser les hyperparamètres de YOLOv8. Il comprend :

- `hyperparameters_evolutionner_light.py` : Génère des expériences avec mutation, croisement et évaluation automatique.
- `app_evolve_light.py` : Application Flask pour visualiser l'évolution des hyperparamètres.

## 🚀 Lancer l'optimisation

```bash
python hyperparameters_evolutionner_light.py
```

Les résultats seront enregistrés dans le dossier `experiments/`.

## 🌐 Lancer la visualisation

```bash
python app_evolve_light.py
```

Puis ouvrir [http://localhost:5000](http://localhost:5000) dans le navigateur.

## 📦 Dépendances

Voir `requirements.txt`.

## 📁 Structure

```
.
├── hyperparameters_evolutionner_light.py
├── app_evolve_light.py
├── experiments/                 # Résultats générés
├── unified/images/             # Images source
├── unified/labels/             # Labels YOLO
└── summary.csv                 # Résumé des essais
```
