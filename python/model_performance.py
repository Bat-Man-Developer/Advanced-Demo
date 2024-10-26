# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

class Performance:
    def __init__(self, data_paths):
        # Initialize class attributes
        self.data = self.load_data(data_paths)
        self.data = self.preprocess_data(self.data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(self.data)
        
        # Initialize Random Forest model
        self.model = RandomForestClassifier(n_estimators=1000, random_state=42)
        
        # Perform cross-validation
        self.cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        # Get feature importance
        self.feature_importance = self.model.feature_importances_

    def load_data(self, data_paths):
        # Load data from multiple CSV files
        data_frames = []
        for path in data_paths:
            df = pd.read_csv(path)
            data_frames.append(df)
        return pd.concat(data_frames, ignore_index=True)

    def preprocess_data(self, data):
        # Convert 'time' column to datetime and other columns to numeric
        data['time'] = pd.to_datetime(data['time'])
        for col in data.columns:
            if col != 'time':
                data[col] = pd.to_numeric(data[col], errors='coerce')
        return data

    def split_data(self, data):
        # Split data into features (X) and target (y)
        X = data.drop(['time', 'label'], axis=1)
        y = data['label']
        return train_test_split(X, y, train_size=0.70, test_size=0.30, random_state=42, shuffle=True)

    def predict(self, X):
        # Make predictions using Random Forest
        return self.model.predict_proba(X)[:, 1]

    def evaluate_model(self):
        # Evaluate the model
        y_pred_proba = self.predict(self.X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        return accuracy, precision, recall, f1, tp, fp, fn, tn, y_pred_proba, self.cv_scores

    def print_results(self):
        print("Uploaded Model Statistics Successfully.")
        print(f"Cross-validation scores: {self.cv_scores}")
        print(f"Mean CV score: {self.cv_scores.mean():.3f} (+/- {self.cv_scores.std() * 2:.3f})")
        metrics = self.evaluate_model()
        self.plot_results(metrics)

    def plot_bias_variance(self):
        train_sizes = np.linspace(0.1, 0.99, 10)
        
        plt.figure(figsize=(12, 8))
        
        train_sizes_abs, train_scores, test_scores = learning_curve(
            self.model, self.X_train, self.y_train,
            cv=5, n_jobs=-1,
            train_sizes=train_sizes,
            random_state=42)
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.plot(train_sizes_abs, train_mean, 'o-', label='Training score')
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.plot(train_sizes_abs, test_mean, 's-', label='Cross-validation score')
        plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1)
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title('Learning Curves (Random Forest)')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('C:/Xampp/htdocs/ai-demo-website/model_plots/learning_curves.png')
        plt.close()

    def plot_results(self, metrics):
        # Learning Curves
        self.plot_bias_variance()

        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        y_pred = (metrics[8] > 0.5).astype(int)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('C:/Xampp/htdocs/ai-demo-website/model_plots/confusion_matrix.png')
        plt.close()

        # Feature Importance
        plt.figure(figsize=(12, 6))
        feature_names = [f'Feature {i}' for i in range(len(self.feature_importance))]
        sorted_idx = np.argsort(self.feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.barh(pos, self.feature_importance[sorted_idx], align='center')
        plt.yticks(pos, np.array(feature_names)[sorted_idx])
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('C:/Xampp/htdocs/ai-demo-website/model_plots/feature_importance.png')
        plt.close()

        # Performance Metrics
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = metrics[:4]
        plt.figure(figsize=(10, 6))
        plt.bar(metrics_names, metrics_values)
        plt.title('Random Forest Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        for i, v in enumerate(metrics_values):
            plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig('C:/Xampp/htdocs/ai-demo-website/model_plots/performance_metrics.png')
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, metrics[8])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('C:/Xampp/htdocs/ai-demo-website/model_plots/roc_curve.png')
        plt.close()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.y_test, metrics[8])
        average_precision = average_precision_score(self.y_test, metrics[8])
        plt.figure(figsize=(10, 8))
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')
        plt.savefig('C:/Xampp/htdocs/ai-demo-website/model_plots/precision_recall_curve.png')
        plt.close()

        # Calibration Plot
        fraction_of_positives, mean_predicted_value = calibration_curve(self.y_test, metrics[8], n_bins=10)
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Random Forest")
        plt.ylabel("Fraction of positives")
        plt.xlabel("Mean predicted value")
        plt.title("Calibration Plot")
        plt.legend(loc="best")
        plt.savefig('C:/Xampp/htdocs/ai-demo-website/model_plots/calibration_plot.png')
        plt.close()

# Usage
data_paths = [
    'C:/Xampp/htdocs/ai-demo-website/dataset/ICMP_FLOOD.csv',
    'C:/Xampp/htdocs/ai-demo-website/dataset/ICMP_BENIGN.csv'
]

# Create an instance of the Performance class and print results
performance = Performance(data_paths)
performance.print_results()