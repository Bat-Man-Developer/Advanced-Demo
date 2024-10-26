import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

class Performance:
    def __init__(self, data_paths, n_iterations=2):
        self.data = self.load_data(data_paths)
        self.data = self.preprocess_data(self.data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(self.data)
        self.n_iterations = n_iterations
        
        # Optimize Random Forest parameters
        self.rf_model = self.optimize_random_forest()
        
        # Perform feature selection and model training
        self.feature_importance = self.stability_selection()
        self.trained_model = self.train_model()

    def load_data(self, data_paths):
        data_frames = []
        for path in data_paths:
            df = pd.read_csv(path)
            data_frames.append(df)
        return pd.concat(data_frames, ignore_index=True)

    def preprocess_data(self, data):
        data['time'] = pd.to_datetime(data['time'])
        for col in data.columns:
            if col != 'time':
                data[col] = pd.to_numeric(data[col], errors='coerce')
        return data

    def split_data(self, data):
        X = data.drop(['time', 'label'], axis=1)
        y = data['label']
        return train_test_split(X, y, train_size=0.70, test_size=0.30, random_state=42, shuffle=True)

    def optimize_random_forest(self):
        param_grid = {
            'n_estimators': [100, 500, 1000],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, scoring='f1')
        grid_search.fit(self.X_train, self.y_train)
        
        return RandomForestClassifier(**grid_search.best_params_, random_state=42)

    def stability_selection(self):
        feature_importance = np.zeros(self.X_train.shape[1])
        for _ in range(self.n_iterations):
            X_sample, y_sample = resample(self.X_train, self.y_train)
            selector = SelectFromModel(estimator=self.rf_model)
            selector.fit(X_sample, y_sample)
            feature_importance += selector.get_support()
        return feature_importance / self.n_iterations

    def train_model(self):
        self.rf_model.fit(self.X_train, self.y_train)
        return self.rf_model

    def predict(self, X):
        return self.trained_model.predict_proba(X)[:, 1]

    def evaluate_model(self):
        y_pred_proba = self.predict(self.X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        
        return accuracy, precision, recall, f1, tp, fp, fn, tn, y_pred_proba

    def print_results(self):
        print("Optimized Random Forest Model Statistics:")
        print(f"Best parameters: {self.rf_model.get_params()}")
        metrics = self.evaluate_model()
        self.plot_results(metrics)

    def plot_results(self, metrics):
        # All the plotting functions remain the same, but only for Random Forest
        # [Previous plotting code remains unchanged]
        # Remove the individual model comparisons and model weights plots
        
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        y_pred = (metrics[8] > 0.5).astype(int)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Random Forest Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('C:/Xampp/htdocs/net-cure-website/model_plots/confusion_matrix.png')
        plt.close()

        # Feature Importance
        plt.figure(figsize=(12, 6))
        feature_importance = self.trained_model.feature_importances_
        feature_names = [f'Feature {i}' for i in range(len(feature_importance))]
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, np.array(feature_names)[sorted_idx])
        plt.title('Random Forest Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('C:/Xampp/htdocs/net-cure-website/model_plots/feature_importance.png')
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
        plt.savefig('C:/Xampp/htdocs/net-cure-website/model_plots/performance_metrics.png')
        plt.close()

# Usage
data_paths = [
    'C:/Xampp/htdocs/net-cure-website/dataset/ICMP_FLOOD.csv',
    'C:/Xampp/htdocs/net-cure-website/dataset/ICMP_BENIGN.csv'
]

performance = Performance(data_paths)
performance.print_results()