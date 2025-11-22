"""
NSL-KDD Dataset Loader for Kaggle Format
Handles various formats of NSL-KDD dataset from Kaggle
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class NSLKDDLoader:
    def __init__(self):
        """Initialize NSL-KDD dataset loader"""
        self.feature_names = [
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
        
        self.attack_categories = {
            'normal': 'Normal',
            'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS',
            'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS',
            'udpstorm': 'DoS',
            'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
            'mscan': 'Probe', 'saint': 'Probe',
            'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
            'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
            'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
            'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
            'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
            'httptunnel': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R'
        }
        
    def load_data(self, train_path, test_path=None):
        """
        Load NSL-KDD dataset from various Kaggle formats
        
        Args:
            train_path: Path to training data
            test_path: Path to test data (optional)
            
        Returns:
            X_train, X_test, y_train, y_test (or just X_train, y_train if no test_path)
        """
        print("Loading NSL-KDD dataset...")
        
        # Load training data
        X_train, y_train = self._load_single_file(train_path, "training")
        
        if test_path:
            # Load test data
            X_test, y_test = self._load_single_file(test_path, "test")
            return X_train, X_test, y_train, y_test
        else:
            return X_train, y_train
    
    def _load_single_file(self, file_path, data_type="data"):
        """Load a single NSL-KDD file with proper format handling"""
        
        print(f"Loading {data_type} file: {file_path}")
        
        try:
            # Try loading as CSV first
            try:
                data = pd.read_csv(file_path)
                print(f"Loaded as CSV with headers. Shape: {data.shape}")
            except:
                # Try without headers
                data = pd.read_csv(file_path, header=None)
                print(f"Loaded as CSV without headers. Shape: {data.shape}")
            
            # Handle different column counts
            if data.shape[1] == 42:  # 41 features + 1 label
                if data.columns[0] == 0:  # No headers
                    data.columns = self.feature_names
                print("Dataset has correct number of columns (42)")
                
            elif data.shape[1] == 43:  # Sometimes has extra column
                print("Dataset has 43 columns, removing extra column")
                data = data.iloc[:, :-1]  # Remove last column
                data.columns = self.feature_names
                
            elif data.shape[1] == 41:  # Missing label column
                print("Dataset has 41 columns, assuming all are features")
                feature_cols = self.feature_names[:-1]
                data.columns = feature_cols
                # Create dummy labels
                data['class'] = 'normal'
                
            else:
                raise ValueError(f"Unexpected number of columns: {data.shape[1]}")
            
            # Separate features and labels
            X = data.drop('class', axis=1)
            y = data['class']
            
            # Clean label names (remove trailing dots, spaces)
            y = y.astype(str).str.strip().str.rstrip('.')
            
            print(f"Final {data_type} data shape: X{X.shape}, y{y.shape}")
            print(f"Unique classes in {data_type} data: {sorted(y.unique())}")
            
            return X, y
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            raise
    
    def analyze_dataset(self, X, y, dataset_name="NSL-KDD"):
        """
        Perform comprehensive dataset analysis
        
        Args:
            X: Features
            y: Labels
            dataset_name: Name for plots
        """
        print(f"\n{'='*50}")
        print(f"DATASET ANALYSIS: {dataset_name}")
        print(f"{'='*50}")
        
        # Basic info
        print(f"Dataset shape: {X.shape}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of samples: {X.shape[0]}")
        
        # Class distribution
        print(f"\nClass Distribution:")
        class_counts = y.value_counts()
        for class_name, count in class_counts.items():
            percentage = (count / len(y)) * 100
            print(f"  {class_name}: {count} ({percentage:.2f}%)")
        
        # Attack category distribution
        y_categories = y.map(self.attack_categories)
        print(f"\nAttack Category Distribution:")
        category_counts = y_categories.value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(y)) * 100
            print(f"  {category}: {count} ({percentage:.2f}%)")
        
        # Feature types
        print(f"\nFeature Information:")
        print(f"Numerical features: {X.select_dtypes(include=[np.number]).shape[1]}")
        print(f"Categorical features: {X.select_dtypes(include=['object']).shape[1]}")
        
        # Missing values
        missing_values = X.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\nMissing Values:")
            for feature, missing in missing_values[missing_values > 0].items():
                print(f"  {feature}: {missing}")
        else:
            print(f"\nNo missing values found")
        
        # Basic statistics
        print(f"\nBasic Statistics:")
        print(f"Mean duration: {X['duration'].mean():.2f}")
        print(f"Mean src_bytes: {X['src_bytes'].mean():.2f}")
        print(f"Mean dst_bytes: {X['dst_bytes'].mean():.2f}")
        
    def plot_analysis(self, X, y, dataset_name="NSL-KDD"):
        """
        Create comprehensive plots for dataset analysis
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{dataset_name} Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Class distribution
        class_counts = y.value_counts()
        axes[0, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Class Distribution')
        
        # 2. Attack category distribution
        y_categories = y.map(self.attack_categories)
        category_counts = y_categories.value_counts()
        axes[0, 1].bar(category_counts.index, category_counts.values)
        axes[0, 1].set_title('Attack Categories')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Duration distribution
        normal_mask = y == 'normal'
        attack_mask = y != 'normal'
        
        axes[0, 2].hist([X[normal_mask]['duration'], X[attack_mask]['duration']], 
                       bins=50, alpha=0.7, label=['Normal', 'Attack'])
        axes[0, 2].set_title('Duration Distribution')
        axes[0, 2].set_xlabel('Duration')
        axes[0, 2].legend()
        axes[0, 2].set_yscale('log')
        
        # 4. Bytes distribution
        axes[1, 0].hist([X[normal_mask]['src_bytes'], X[attack_mask]['src_bytes']], 
                       bins=50, alpha=0.7, label=['Normal', 'Attack'])
        axes[1, 0].set_title('Source Bytes Distribution')
        axes[1, 0].set_xlabel('Source Bytes')
        axes[1, 0].legend()
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')
        
        # 5. Protocol type distribution
        protocol_counts = X['protocol_type'].value_counts()
        axes[1, 1].bar(protocol_counts.index, protocol_counts.values)
        axes[1, 1].set_title('Protocol Type Distribution')
        
        # 6. Feature correlation heatmap (selected features)
        selected_features = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count']
        if all(feat in X.columns for feat in selected_features):
            correlation_matrix = X[selected_features].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[1, 2])
            axes[1, 2].set_title('Feature Correlations')
        
        plt.tight_layout()
        plt.show()
        
    def prepare_binary_classification(self, y):
        """
        Prepare labels for binary classification (Normal vs Attack)
        
        Args:
            y: Original labels
            
        Returns:
            Binary labels (0=Normal, 1=Attack)
        """
        return (y != 'normal').astype(int)
    
    def prepare_multiclass_classification(self, y, use_categories=True):
        """
        Prepare labels for multi-class classification
        
        Args:
            y: Original labels
            use_categories: If True, use attack categories; if False, use specific attacks
            
        Returns:
            Multi-class labels and label encoder
        """
        if use_categories:
            # Use attack categories (Normal, DoS, Probe, R2L, U2R)
            y_processed = y.map(self.attack_categories)
            print(f"Using attack categories: {sorted(y_processed.unique())}")
        else:
            # Use specific attack types
            y_processed = y
            print(f"Using specific attacks: {sorted(y_processed.unique())}")
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_processed)
        
        print(f"Label encoding mapping:")
        for i, class_name in enumerate(le.classes_):
            print(f"  {i}: {class_name}")
        
        return y_encoded, le
    
    def get_feature_info(self):
        """Get information about NSL-KDD features"""
        feature_info = {
            'basic_features': self.feature_names[:9],
            'content_features': self.feature_names[9:22],
            'traffic_features': self.feature_names[22:41],
            'categorical_features': ['protocol_type', 'service', 'flag'],
            'binary_features': ['land', 'logged_in', 'root_shell', 'su_attempted', 
                               'is_host_login', 'is_guest_login']
        }
        return feature_info


# Example usage and testing
if __name__ == "__main__":
    # Initialize loader
    loader = NSLKDDLoader()
    
    print("NSL-KDD Dataset Loader")
    print("=" * 50)
    
    # Example file paths (update these with your actual paths)
    train_path = "KDDTrain+.txt"  # Update this
    test_path = "KDDTest+.txt"    # Update this
    
    try:
        # Load data
        print(f"Attempting to load NSL-KDD dataset...")
        print(f"Training file: {train_path}")
        print(f"Test file: {test_path}")
        
        X_train, X_test, y_train, y_test = loader.load_data(train_path, test_path)
        
        # Analyze training data
        loader.analyze_dataset(X_train, y_train, "NSL-KDD Training")
        
        # Analyze test data
        loader.analyze_dataset(X_test, y_test, "NSL-KDD Test")
        
        # Create visualizations
        loader.plot_analysis(X_train, y_train, "NSL-KDD Training")
        
        # Prepare different label formats
        print(f"\n" + "="*50)
        print("LABEL PREPARATION")
        print("="*50)
        
        # Binary classification
        y_train_binary = loader.prepare_binary_classification(y_train)
        y_test_binary = loader.prepare_binary_classification(y_test)
        print(f"Binary classification - Training attack ratio: {y_train_binary.mean():.2%}")
        print(f"Binary classification - Test attack ratio: {y_test_binary.mean():.2%}")
        
        # Multi-class classification (categories)
        y_train_multi, le_multi = loader.prepare_multiclass_classification(y_train, use_categories=True)
        y_test_multi = le_multi.transform(y_test.map(loader.attack_categories))
        print(f"Multi-class (categories) - Classes: {len(le_multi.classes_)}")
        
        # Multi-class classification (specific attacks)
        y_train_specific, le_specific = loader.prepare_multiclass_classification(y_train, use_categories=False)
        y_test_specific = le_specific.transform(y_test)
        print(f"Multi-class (specific) - Classes: {len(le_specific.classes_)}")
        
        # Feature information
        feature_info = loader.get_feature_info()
        print(f"\nFeature Information:")
        for category, features in feature_info.items():
            print(f"  {category}: {len(features)} features")
        
        print(f"\nDataset loaded and analyzed successfully!")
        print(f"Ready for CNN-LSTM training...")
        
    except FileNotFoundError:
        print(f"\nError: Dataset files not found!")
        print(f"Please download NSL-KDD from Kaggle and update the file paths:")
        print(f"1. Visit: https://www.kaggle.com/datasets/hassan06/nslkdd")
        print(f"2. Download the dataset")
        print(f"3. Update the file paths in this script")
        print(f"4. Run again")
        
    except Exception as e:
        print(f"\nError loading dataset: {str(e)}")
        print(f"Please check the file format and paths")
