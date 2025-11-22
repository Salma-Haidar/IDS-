"""
Complete NSL-KDD CNN-LSTM Intrusion Detection System
Main execution script that combines all components
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from nsl_kdd_loader import NSLKDDLoader
from nsl_kdd_cnn_lstm_detector import NSLKDDCNNLSTMDetector

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def main():
    """
    Main execution function for NSL-KDD CNN-LSTM experiment
    """
    print("=" * 80)
    print("NSL-KDD CNN-LSTM INTRUSION DETECTION SYSTEM")
    print("Complete Implementation for University AI Project")
    print("=" * 80)
    
    # Configuration
    config = {
        'train_path': 'KDDTrain+.txt',  # Update with your path
        'test_path': 'KDDTest+.txt',    # Update with your path
        'classification_type': 'binary',  # 'binary' or 'multiclass'
        'sequence_length': 10,
        'use_rfe': True,
        'n_features': 20,
        'epochs': 100,
        'batch_size': 64,
        'validation_split': 0.15
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # Step 1: Load and analyze dataset
        print("STEP 1: Loading and Analyzing Dataset")
        print("-" * 50)
        
        loader = NSLKDDLoader()
        X_train, X_test, y_train, y_test = loader.load_data(
            config['train_path'], config['test_path']
        )
        
        # Perform dataset analysis
        loader.analyze_dataset(X_train, y_train, "NSL-KDD Training Set")
        loader.plot_analysis(X_train, y_train, "NSL-KDD Training Set")
        
        # Step 2: Initialize detector
        print(f"\nSTEP 2: Initializing CNN-LSTM Detector")
        print("-" * 50)
        
        detector = NSLKDDCNNLSTMDetector(sequence_length=config['sequence_length'])
        
        # Step 3: Preprocess data
        print(f"\nSTEP 3: Data Preprocessing")
        print("-" * 50)
        
        X_train_processed, X_test_processed, y_train_processed, y_test_processed = detector.preprocess_data(
            X_train, X_test, y_train, y_test,
            classification_type=config['classification_type'],
            use_rfe=config['use_rfe'],
            n_features=config['n_features']
        )
        
        print(f"Preprocessed data shapes:")
        print(f"  X_train: {X_train_processed.shape}")
        print(f"  X_test: {X_test_processed.shape}")
        print(f"  y_train: {y_train_processed.shape}")
        print(f"  y_test: {y_test_processed.shape}")
        
        # Step 4: Build model
        print(f"\nSTEP 4: Building CNN-LSTM Model")
        print("-" * 50)
        
        n_classes = 2 if config['classification_type'] == 'binary' else len(np.unique(y_train_processed))
        model = detector.build_cnn_lstm_model(n_classes=n_classes)
        
        # Step 5: Train model
        print(f"\nSTEP 5: Training Model")
        print("-" * 50)
        
        detector.train_model(
            X_train_processed, y_train_processed,
            validation_split=config['validation_split'],
            epochs=config['epochs'],
            batch_size=config['batch_size']
        )
        
        # Step 6: Evaluate model
        print(f"\nSTEP 6: Model Evaluation")
        print("-" * 50)
        
        results = detector.evaluate_model(X_test_processed, y_test_processed)
        
        # Step 7: Visualize results
        print(f"\nSTEP 7: Result Visualization")
        print("-" * 50)
        
        detector.plot_results(X_test_processed, y_test_processed)
        
        # Step 8: Save model and results
        print(f"\nSTEP 8: Saving Results")
        print("-" * 50)
        
        # Save model
        model_filename = f'nsl_kdd_cnn_lstm_{config["classification_type"]}_model.h5'
        detector.model.save(model_filename)
        print(f"Model saved as: {model_filename}")
        
        # Save results
        results_df = pd.DataFrame([results])
        results_df['classification_type'] = config['classification_type']
        results_df['sequence_length'] = config['sequence_length']
        results_df['n_features'] = config['n_features']
        
        results_filename = f'nsl_kdd_results_{config["classification_type"]}.csv'
        results_df.to_csv(results_filename, index=False)
        print(f"Results saved as: {results_filename}")
        
        # Final summary
        print(f"\n" + "=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Final Results Summary:")
        print(f"  Classification Type: {config['classification_type'].upper()}")
        print(f"  Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"  Test Precision: {results['precision']:.4f}")
        print(f"  Test Recall: {results['recall']:.4f}")
        print(f"  Test F1-Score: {results['f1_score']:.4f}")
        if results['auc']:
            print(f"  Test AUC-ROC: {results['auc']:.4f}")
        
        # Compare with research benchmarks
        print(f"\nComparison with Research Benchmarks:")
        if config['classification_type'] == 'binary':
            benchmark_acc = 0.93  # From research papers
            improvement = ((results['accuracy'] - benchmark_acc) / benchmark_acc) * 100
            print(f"  Research Benchmark: 93-95% accuracy")
            print(f"  Our Result: {results['accuracy']*100:.2f}% accuracy")
            if improvement > 0:
                print(f"  Improvement: +{improvement:.1f}% above benchmark!")
            else:
                print(f"  Performance: {improvement:.1f}% from benchmark")
        else:
            benchmark_acc = 0.91
            improvement = ((results['accuracy'] - benchmark_acc) / benchmark_acc) * 100
            print(f"  Research Benchmark: 90-93% accuracy")
            print(f"  Our Result: {results['accuracy']*100:.2f}% accuracy")
            if improvement > 0:
                print(f"  Improvement: +{improvement:.1f}% above benchmark!")
            else:
                print(f"  Performance: {improvement:.1f}% from benchmark")
        
        print(f"\nFiles Generated:")
        print(f"  📊 Model: {model_filename}")
        print(f"  📈 Results: {results_filename}")
        print(f"  📋 Training history available in memory")
        
        print(f"\nProject Status: ✅ SUCCESS")
        print(f"Ready for university presentation!")
        
        return detector, results
        
    except FileNotFoundError:
        print("❌ ERROR: Dataset files not found!")
        print(f"\nPlease follow these steps:")
        print(f"1. Download NSL-KDD from Kaggle:")
        print(f"   https://www.kaggle.com/datasets/hassan06/nslkdd")
        print(f"2. Place files in your project directory:")
        print(f"   - KDDTrain+.txt")
        print(f"   - KDDTest+.txt")
        print(f"3. Update file paths in config if needed")
        print(f"4. Run this script again")
        
        return None, None
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print(f"\nTroubleshooting tips:")
        print(f"1. Check file paths and formats")
        print(f"2. Ensure all dependencies are installed")
        print(f"3. Check available memory (recommend 8GB+)")
        print(f"4. Try reducing batch_size or sequence_length")
        
        return None, None


def run_comparative_analysis():
    """
    Run comparative analysis between binary and multiclass classification
    """
    print("Running Comparative Analysis...")
    print("This will train both binary and multiclass models")
    
    configs = [
        {'classification_type': 'binary', 'epochs': 50},
        {'classification_type': 'multiclass', 'epochs': 50}
    ]
    
    results_comparison = []
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Training {config['classification_type'].upper()} Classification Model")
        print(f"{'='*60}")
        
        # Update main config
        main_config = {
            'train_path': 'KDDTrain+.txt',
            'test_path': 'KDDTest+.txt',
            'classification_type': config['classification_type'],
            'sequence_length': 10,
            'use_rfe': True,
            'n_features': 20,
            'epochs': config['epochs'],
            'batch_size': 64,
            'validation_split': 0.15
        }
        
        try:
            detector, results = main()
            if results:
                results['type'] = config['classification_type']
                results_comparison.append(results)
        except Exception as e:
            print(f"Error in {config['classification_type']} classification: {str(e)}")
    
    # Compare results
    if len(results_comparison) == 2:
        print(f"\n{'='*80}")
        print("COMPARATIVE ANALYSIS RESULTS")
        print(f"{'='*80}")
        
        comparison_df = pd.DataFrame(results_comparison)
        print(comparison_df.to_string(index=False))
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        x = np.arange(len(metrics))
        width = 0.35
        
        binary_scores = [comparison_df[comparison_df['type']=='binary'][metric].iloc[0] for metric in metrics]
        multi_scores = [comparison_df[comparison_df['type']=='multiclass'][metric].iloc[0] for metric in metrics]
        
        plt.bar(x - width/2, binary_scores, width, label='Binary Classification', alpha=0.8)
        plt.bar(x + width/2, multi_scores, width, label='Multiclass Classification', alpha=0.8)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Binary vs Multiclass Classification Performance')
        plt.xticks(x, metrics)
        plt.legend()
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for i, (binary_score, multi_score) in enumerate(zip(binary_scores, multi_scores)):
            plt.text(i - width/2, binary_score + 0.01, f'{binary_score:.3f}', ha='center')
            plt.text(i + width/2, multi_score + 0.01, f'{multi_score:.3f}', ha='center')
        
        plt.tight_layout()
        plt.show()
        
        comparison_df.to_csv('nsl_kdd_comparison_results.csv', index=False)
        print(f"\nComparison results saved to: nsl_kdd_comparison_results.csv")


if __name__ == "__main__":
    print("NSL-KDD CNN-LSTM Intrusion Detection System")
    print("Choose execution mode:")
    print("1. Single experiment (recommended for first run)")
    print("2. Comparative analysis (binary vs multiclass)")
    print("3. Quick test with reduced epochs")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        detector, results = main()
    elif choice == "2":
        run_comparative_analysis()
    elif choice == "3":
        print("Running quick test...")
        # Override config for quick testing
        import __main__
        __main__.config = {
            'train_path': 'KDDTrain+.txt',
            'test_path': 'KDDTest+.txt',
            'classification_type': 'binary',
            'sequence_length': 5,
            'use_rfe': True,
            'n_features': 15,
            'epochs': 10,
            'batch_size': 128,
            'validation_split': 0.1
        }
        detector, results = main()
    else:
        print("Invalid choice. Running default single experiment...")
        detector, results = main()
