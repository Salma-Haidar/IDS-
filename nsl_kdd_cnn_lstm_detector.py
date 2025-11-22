"""
CNN-LSTM Intrusion Detection System for NSL-KDD Dataset (Kaggle)
Optimized implementation for NSL-KDD dataset from Kaggle
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class NSLKDDCNNLSTMDetector:
    def __init__(self, sequence_length=10):
        """
        Initialize the CNN-LSTM Intrusion Detection System for NSL-KDD
        
        Args:
            sequence_length (int): Length of sequences for LSTM
        """
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = None
        self.feature_names = None
        self.n_features = None
        
    def load_nsl_kdd_data(self, train_path, test_path):
        """
        Load NSL-KDD dataset from Kaggle format
        
        Args:
            train_path: Path to training data CSV
            test_path: Path to testing data CSV
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("Loading NSL-KDD dataset from Kaggle...")
        
        # NSL-KDD feature names (41 features + label)
        feature_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class'
        ]
        
        # Load datasets
        try:
            # Try loading with headers first (Kaggle format)
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            # Check if it has the right number of columns
            if train_data.shape[1] == 42:  # 41 features + 1 label
                print("Dataset loaded with headers")
                if 'class' not in train_data.columns:
                    # If last column is not named 'class', rename it
                    train_data.columns = feature_names
                    test_data.columns = feature_names
            else:
                # No headers, add them
                train_data.columns = feature_names
                test_data.columns = feature_names
                
        except:
            # If CSV loading fails, try without headers
            train_data = pd.read_csv(train_path, header=None, names=feature_names)
            test_data = pd.read_csv(test_path, header=None, names=feature_names)
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        # Separate features and labels
        X_train = train_data.drop('class', axis=1)
        y_train = train_data['class']
        X_test = test_data.drop('class', axis=1)
        y_test = test_data['class']
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        print(f"Features: {X_train.shape[1]}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Display class distribution
        print(f"\\nTraining set class distribution:")
        print(y_train.value_counts())
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_data(self, X_train, X_test, y_train, y_test, 
                       classification_type='binary', use_rfe=True, n_features=20):
        """
        Preprocess NSL-KDD data
        
        Args:
            X_train, X_test, y_train, y_test: Data splits
            classification_type: 'binary' or 'multiclass'
            use_rfe: Whether to use Recursive Feature Elimination
            n_features: Number of features to select with RFE
            
        Returns:
            Preprocessed data ready for training
        """
        print("Preprocessing NSL-KDD data...")
        
        # Handle categorical features
        categorical_features = ['protocol_type', 'service', 'flag']
        
        for feature in categorical_features:
            if feature in X_train.columns:
                # Use LabelEncoder for categorical features
                le = LabelEncoder()
                # Fit on combined data to ensure consistency
                combined_data = pd.concat([X_train[feature], X_test[feature]])
                le.fit(combined_data.astype(str))
                
                X_train[feature] = le.transform(X_train[feature].astype(str))
                X_test[feature] = le.transform(X_test[feature].astype(str))
        
        # Handle missing values
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        # Process labels based on classification type
        if classification_type == 'binary':
            # Binary: Normal (0) vs Attack (1)
            y_train_processed = (y_train != 'normal').astype(int)
            y_test_processed = (y_test != 'normal').astype(int)
            print(f"Binary classification - Attack ratio: {y_train_processed.mean():.2%}")
        else:
            # Multi-class: Keep original attack types
            # Encode all labels
            le_labels = LabelEncoder()
            combined_labels = pd.concat([y_train, y_test])
            le_labels.fit(combined_labels)
            
            y_train_processed = le_labels.transform(y_train)
            y_test_processed = le_labels.transform(y_test)
            
            print(f"Multi-class classification - Classes: {len(le_labels.classes_)}")
            print(f"Class names: {le_labels.classes_}")
            
            self.label_encoder = le_labels
        
        # Feature selection using RFE
        if use_rfe:
            print(f"Applying RFE to select {n_features} features...")
            dt_classifier = DecisionTreeClassifier(random_state=42)
            rfe = RFE(estimator=dt_classifier, n_features_to_select=n_features, step=1)
            
            X_train_rfe = rfe.fit_transform(X_train, y_train_processed)
            X_test_rfe = rfe.transform(X_test)
            
            # Store selected feature names
            selected_features = [self.feature_names[i] for i in range(len(self.feature_names)) if rfe.support_[i]]
            print(f"Selected features: {selected_features}")
            
            X_train = pd.DataFrame(X_train_rfe)
            X_test = pd.DataFrame(X_test_rfe)
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create sequences for LSTM
        print("Creating sequences for LSTM...")
        X_train_seq = self.create_sequences(X_train_scaled)
        X_test_seq = self.create_sequences(X_test_scaled)
        
        # Store number of features
        self.n_features = X_train_seq.shape[2]
        
        print(f"Final training shape: {X_train_seq.shape}")
        print(f"Final test shape: {X_test_seq.shape}")
        
        return X_train_seq, X_test_seq, y_train_processed, y_test_processed
    
    def create_sequences(self, X):
        """
        Create sequences for LSTM from feature data
        
        Args:
            X: Feature matrix
            
        Returns:
            Reshaped sequences for LSTM
        """
        n_samples, n_features = X.shape
        
        # If we have fewer samples than sequence length, pad with zeros
        if n_samples < self.sequence_length:
            padding = np.zeros((self.sequence_length - n_samples, n_features))
            X_padded = np.vstack([X, padding])
            return X_padded.reshape(1, self.sequence_length, n_features)
        
        # Create overlapping sequences
        sequences = []
        for i in range(n_samples - self.sequence_length + 1):
            sequences.append(X[i:i + self.sequence_length])
        
        # Create corresponding labels (repeat labels for each sequence)
        if len(sequences) == 0:
            # Fallback: create single sequence with available data
            if n_samples > 0:
                seq = np.zeros((self.sequence_length, n_features))
                seq[:min(n_samples, self.sequence_length)] = X[:min(n_samples, self.sequence_length)]
                sequences = [seq]
            else:
                sequences = [np.zeros((self.sequence_length, n_features))]
        
        return np.array(sequences)
    
    def build_cnn_lstm_model(self, n_classes=2):
        """
        Build optimized CNN-LSTM model for NSL-KDD
        
        Args:
            n_classes: Number of classes
        """
        print("Building CNN-LSTM model for NSL-KDD...")
        
        # Input layer
        input_layer = layers.Input(shape=(self.sequence_length, self.n_features), name='input')
        
        # CNN Block for feature extraction
        cnn = layers.Conv1D(filters=64, kernel_size=3, activation='relu', 
                           padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_layer)
        cnn = layers.BatchNormalization()(cnn)
        cnn = layers.MaxPooling1D(pool_size=2)(cnn)
        cnn = layers.Dropout(0.3)(cnn)
        
        cnn = layers.Conv1D(filters=128, kernel_size=3, activation='relu',
                           padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01))(cnn)
        cnn = layers.BatchNormalization()(cnn)
        cnn = layers.MaxPooling1D(pool_size=2)(cnn)
        cnn = layers.Dropout(0.3)(cnn)
        
        # LSTM Block for temporal patterns
        lstm = layers.LSTM(units=100, dropout=0.2, recurrent_dropout=0.2,
                          return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(cnn)
        lstm = layers.LSTM(units=50, dropout=0.2, recurrent_dropout=0.2)(lstm)
        
        # Dense layers
        dense = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(lstm)
        dense = layers.BatchNormalization()(dense)
        dense = layers.Dropout(0.4)(dense)
        
        dense = layers.Dense(32, activation='relu')(dense)
        dense = layers.Dropout(0.3)(dense)
        
        # Output layer
        if n_classes == 2:
            output = layers.Dense(1, activation='sigmoid', name='output')(dense)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            output = layers.Dense(n_classes, activation='softmax', name='output')(dense)
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        # Create and compile model
        self.model = Model(inputs=input_layer, outputs=output, name='NSL_KDD_CNN_LSTM')
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
            loss=loss,
            metrics=metrics
        )
        
        print("\\nModel Summary:")
        self.model.summary()
        
        return self.model
    
    def train_model(self, X_train, y_train, validation_split=0.15, epochs=100, batch_size=64):
        """
        Train the CNN-LSTM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            validation_split: Validation split ratio
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        print(f"Training CNN-LSTM model on NSL-KDD...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation split: {validation_split}")
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=20, 
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_nsl_kdd_model.h5', 
                monitor='val_loss', 
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=10, 
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model
        print("\\nStarting training...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        print("Training completed!")
        
    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive model evaluation
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with all evaluation metrics
        """
        print("\\nEvaluating model performance...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        
        if len(np.unique(y_test)) == 2:  # Binary classification
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            y_test_eval = y_test
        else:  # Multi-class
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_test_eval = y_test
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_eval, y_pred)
        precision = precision_score(y_test_eval, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test_eval, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_eval, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC for binary classification
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test_eval, y_pred_proba)
        else:
            auc = None
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
        
        # Print results
        print(f"\\n{'='*50}")
        print(f"MODEL EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        if auc:
            print(f"AUC-ROC:   {auc:.4f}")
        print(f"{'='*50}")
        
        # Detailed classification report
        print(f"\\nDetailed Classification Report:")
        if len(np.unique(y_test)) == 2:
            target_names = ['Normal', 'Attack']
        else:
            target_names = [f'Class_{i}' for i in range(len(np.unique(y_test)))]
        
        print(classification_report(y_test_eval, y_pred, target_names=target_names))
        
        return results
    
    def plot_results(self, X_test, y_test):
        """
        Plot comprehensive results
        """
        # Training history
        if self.history:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Accuracy plot
            ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
            ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
            ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Loss plot
            ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
            ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
            ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Confusion Matrix
            y_pred_proba = self.model.predict(X_test, verbose=0)
            if len(np.unique(y_test)) == 2:
                y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                labels = ['Normal', 'Attack']
            else:
                y_pred = np.argmax(y_pred_proba, axis=1)
                labels = [f'Class {i}' for i in range(len(np.unique(y_test)))]
            
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                       xticklabels=labels, yticklabels=labels)
            ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Predicted')
            ax3.set_ylabel('Actual')
            
            # Feature importance (if available)
            ax4.text(0.5, 0.5, f'NSL-KDD CNN-LSTM\\nIntrusion Detection\\n\\nTotal Parameters: {self.model.count_params():,}\\nSequence Length: {self.sequence_length}\\nFeatures: {self.n_features}', 
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            
            plt.tight_layout()
            plt.show()


def main():
    """
    Main function to run NSL-KDD CNN-LSTM experiment
    """
    print("=" * 60)
    print("NSL-KDD CNN-LSTM INTRUSION DETECTION SYSTEM")
    print("=" * 60)
    
    # Initialize detector
    detector = NSLKDDCNNLSTMDetector(sequence_length=10)
    
    # File paths (update these with your Kaggle NSL-KDD paths)
    train_path = "KDDTrain+.txt"  # Update this path
    test_path = "KDDTest+.txt"    # Update this path
    
    try:
        # Load data
        X_train, X_test, y_train, y_test = detector.load_nsl_kdd_data(train_path, test_path)
        
        # Choose classification type
        classification_type = 'binary'  # Change to 'multiclass' for multi-class classification
        
        # Preprocess data
        X_train_processed, X_test_processed, y_train_processed, y_test_processed = detector.preprocess_data(
            X_train, X_test, y_train, y_test,
            classification_type=classification_type,
            use_rfe=True,
            n_features=20
        )
        
        # Build model
        n_classes = 2 if classification_type == 'binary' else len(np.unique(y_train_processed))
        model = detector.build_cnn_lstm_model(n_classes=n_classes)
        
        # Train model
        detector.train_model(
            X_train_processed, y_train_processed,
            validation_split=0.15,
            epochs=100,
            batch_size=64
        )
        
        # Evaluate model
        results = detector.evaluate_model(X_test_processed, y_test_processed)
        
        # Plot results
        detector.plot_results(X_test_processed, y_test_processed)
        
        # Save model
        detector.model.save('nsl_kdd_cnn_lstm_final_model.h5')
        print(f"\\nModel saved as 'nsl_kdd_cnn_lstm_final_model.h5'")
        
        print(f"\\nExperiment completed successfully!")
        print(f"Final Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find dataset files.")
        print(f"Please make sure the NSL-KDD dataset files are in the correct location:")
        print(f"- Training file: {train_path}")
        print(f"- Test file: {test_path}")
        print(f"\\nYou can download NSL-KDD from Kaggle:")
        print(f"https://www.kaggle.com/datasets/hassan06/nslkdd")


if __name__ == "__main__":
    main()
