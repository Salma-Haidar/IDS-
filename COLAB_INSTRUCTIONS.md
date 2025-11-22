# Instructions pour Google Colab

## Méthode 1 : Cloner depuis GitHub

1. Créer un nouveau notebook sur [Google Colab](https://colab.research.google.com/)

2. Exécuter ces cellules :
```python
# Cellule 1 : Cloner le repo
!git clone https://github.com/VOTRE-USERNAME/nsl-kdd-intrusion-detection.git
%cd nsl-kdd-intrusion-detection
!ls -la

# Cellule 2 : Installer les dépendances
!pip install -q -r requirements.txt

# Cellule 3 : Vérifier GPU
import tensorflow as tf
print(f"GPU disponible: {len(tf.config.list_physical_devices('GPU')) > 0}")

# Cellule 4 : Télécharger les données (si non incluses dans le repo)
# Option A : Depuis Kaggle
from google.colab import files
uploaded = files.upload()  # Upload kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d hassan06/nslkdd
!unzip -q nslkdd.zip -d data/
!ls data/

# Option B : Upload manuel
from google.colab import files
uploaded = files.upload()  # Upload KDDTrain+.txt et KDDTest+.txt
!mv KDDTrain+.txt data/
!mv KDDTest+.txt data/

# Cellule 5 : Exécuter le projet
!python run_nsl_kdd_experiment.py
```

## Méthode 2 : Via Google Drive
```python
# Monter Drive
from google.colab import drive
drive.mount('/content/drive')

# Copier le projet
!cp -r "/content/drive/MyDrive/nsl_kdd_intrusion_detection" "/content/"
%cd nsl_kdd_intrusion_detection

# Installer et exécuter
!pip install -q -r requirements.txt
!python run_nsl_kdd_experiment.py
```

## Configuration rapide pour test

Pour un test rapide (5-10 minutes), modifiez dans `run_nsl_kdd_experiment.py` :
```python
config = {
    'epochs': 10,
    'batch_size': 128,
    'n_features': 15,
    'sequence_length': 5
}
```

## Télécharger les résultats
```python
from google.colab import files

# Télécharger le modèle
files.download('models/saved_models/nsl_kdd_binary_model.h5')

# Télécharger les résultats
files.download('results/reports/results_binary.csv')
```