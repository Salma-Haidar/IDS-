# NSL-KDD CNN-LSTM Intrusion Detection System

Système de détection d'intrusion réseau utilisant une architecture CNN-LSTM sur le dataset NSL-KDD.

## 🎯 Objectif

Détecter les intrusions réseau en utilisant un modèle hybride CNN-LSTM qui combine :
- **CNN** : Extraction automatique des caractéristiques
- **LSTM** : Capture des dépendances temporelles

## 📊 Dataset

NSL-KDD Dataset (version améliorée de KDD Cup 1999)
- **Training samples** : ~125,000
- **Test samples** : ~22,500
- **Features** : 41
- **Classes** : Normal, DoS, Probe, R2L, U2R

## 🏗️ Architecture
```
Input (10, 20) → CNN Block → LSTM Block → Dense Layers → Output
```

## 📁 Structure du projet
```
nsl_kdd_intrusion_detection/
├── data/
│   ├── KDDTrain+.txt
│   ├── KDDTest+.txt
│   └── processed/
├── models/
│   └── saved_models/
├── results/
│   ├── plots/
│   └── reports/
├── nsl_kdd_cnn_lstm_detector.py
├── nsl_kdd_loader.py
├── run_nsl_kdd_experiment.py
├── requirements.txt
└── README.md
```

## 🚀 Installation
```bash
pip install -r requirements.txt
```

## 💻 Utilisation

### En local
```bash
python run_nsl_kdd_experiment.py
```

### Dans Google Colab

Voir `COLAB_INSTRUCTIONS.md`

## 📈 Résultats attendus

- **Binary Classification** : ~95-97% accuracy
- **Multi-class Classification** : ~92-94% accuracy

## 📝 Auteur

Salma HAIDAR
