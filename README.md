# 🧬 Hyperparameter Evolution with YOLOv8

Ce projet implémente un algorithme génétique pour optimiser automatiquement les hyperparamètres du modèle YOLOv8. Il est conçu pour ajuster des paramètres comme le taux d'apprentissage, la taille d'image ou la taille du dataset afin de maximiser la précision de détection d'objets.

---

## 🧠 Objectif

L'objectif est de **trouver les meilleures combinaisons d'hyperparamètres** qui permettent d'obtenir un maximum de performance (`mAP_50:95`) sur une tâche de détection (ex. : détection de moustiques vs négatifs).

Ce processus repose sur une **stratégie d’évolution génétique** :
- Génération de populations d'hyperparamètres aléatoires,
- Sélection des meilleurs modèles,
- Croisement et mutation des paramètres pour créer la génération suivante.

---

## 📂 Contenu du projet

- `hyperparameters_evolutionner_light.py`  
  Script principal de l’algorithme génétique. Il entraîne, évalue, sélectionne, croise et mute les hyperparamètres au fil des générations.

- `app_evolve_light.py`  
  Application Flask interactive. Permet de visualiser dynamiquement l’évolution des runs, leurs parents, mutations et performances.

- `experiments/`  
  Contient les runs générés automatiquement. Chaque run a son propre dossier avec logs, résultats et dataset utilisé.

- `unified/images/` et `unified/labels/`  
  Dossier source pour les images positives (moustiques) et négatives, ainsi que leurs labels YOLO. Ces données sont utilisées pour créer des sous-datasets temporaires.

---

## 🚀 Lancer l'optimisation

Lance simplement le script principal pour démarrer l’évolution :

```bash
python hyperparameters_evolutionner_light.py
```

Cela crée des générations d’entraînements successifs. Les résultats sont stockés dans `experiments/exp_001`.

---

## 🌐 Visualiser les résultats

Pour explorer visuellement l’évolution de la population d’hyperparamètres :

```bash
python app_evolve_light.py
```

Puis ouvre ton navigateur à l’adresse :  
[http://localhost:5000](http://localhost:5000)

Fonctionnalités :
- Visualisation en graphe par génération
- Couleur selon la métrique choisie (fitness, précision, rappel, mAP)
- Infobulle avec paramètres, mutations, scores
- Sélection d’expériences si plusieurs runs sont présents

---

## 📦 Installation des dépendances

Crée un environnement virtuel et installe les dépendances :

```bash
pip install -r requirements.txt
```

---

## ⚙️ Paramètres évolutifs

Les hyperparamètres suivants sont optimisés par l’algorithme :

- `lr0` : taux d’apprentissage initial
- `batch` : taille de batch
- `positive_ratio` : proportion d’images positives dans le dataset
- `dataset_size` : nombre total d’images dans le dataset
- `imgsz` : taille d’entrée des images pour YOLO

Les autres (comme `epochs` ou `model`) sont fixés dans le script.

---

## 📁 Structure type d’un run

```
experiments/exp_001/
├── run_0001_blue-brave-wolf/
│   ├── dataset/
│   └── yolo/
│       ├── results.csv
│       └── ... (poids, logs)
├── summary.csv  <-- toutes les générations
```

---

## ✅ À faire / extensions possibles

- Ajouter d'autres hyperparamètres à l'évolution
- Intégrer d'autres modèles YOLO
- Ajouter support multi-classe
- Exporter les meilleurs runs automatiquement

---

## 📜 Licence

Ce projet est libre de droits pour usage académique, personnel ou expérimental.
