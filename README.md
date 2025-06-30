# M5 Forecasting - Solution simple LightGBM

Ce dépôt propose une solution simple et pédagogique pour le challenge Kaggle **M5 Forecasting - Accuracy**.

## Prérequis
- Python 3.8+
- Un compte Kaggle (pour télécharger les données)

## Installation de l'environnement

1. Clonez ce dépôt et placez-vous dans le dossier :
   ```bash
   git clone <repo_url>
   cd Kaggle_M5_Forecasting
   ```
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Téléchargement des données

1. Configurez votre clé API Kaggle (voir https://www.kaggle.com/docs/api).
2. Le script télécharge automatiquement les données si elles ne sont pas présentes.

## Exécution du script

Lancez le script principal :
```bash
python m5_simple_lightgbm.py
```

Le script est structuré en cellules exécutables (# %%) pour une lecture facile dans VS Code ou Jupyter. Il effectue toutes les étapes du pipeline :
- Chargement et filtrage des données (exemple : état CA)
- Feature engineering (lags, moyennes mobiles, etc.)
- Entraînement d'un modèle LightGBM
- Évaluation et visualisation des résultats

## Remarques
- Le script est conçu pour être simple, autonome et facilement modifiable.
- Pour une soumission Kaggle, adaptez la partie prédiction finale selon le format requis. 