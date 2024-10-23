# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import logsumexp
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

class Performance:
    def __init__(self, data_paths, n_subspaces=2, subspace_size=0.35, n_iterations=2, threshold=0.5):
        # Initialize class attributes
        self.data = self.load_data(data_paths)
        self.data = self.preprocess_data(self.data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(self.data)
        
        self.n_subspaces = n_subspaces
        self.subspace_size = subspace_size
        self.n_iterations = n_iterations
        self.threshold = threshold
        
        # Define dictionary of models
        self.models = {
            'Linear Regression': LinearRegression(),
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'SVM': SVC(probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=1000, random_state=42),
            'Naive Bayes': GaussianNB(),
            'KNN': KNeighborsClassifier(n_neighbors=1000)
        }
        
        # Perform feature selection and model training
        self.feature_importance = self.stability_selection()
        self.trained_models = self.train_models()
        self.model_weights = self.calculate_model_weights()

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

    def stability_selection(self):
        # Perform stability selection for feature importance
        feature_importance = np.zeros(self.X_train.shape[1])
        for _ in range(self.n_iterations):
            X_sample, y_sample = resample(self.X_train, self.y_train)
            selector = SelectFromModel(estimator=RandomForestClassifier(n_estimators=1000, random_state=42))
            selector.fit(X_sample, y_sample)
            feature_importance += selector.get_support()
        return feature_importance / self.n_iterations

    def random_subspace(self, X):
        # Generate a random subspace of features
        n_features = X.shape[1]
        subspace_size = int(n_features * self.subspace_size)
        selected_features = np.random.choice(n_features, subspace_size, replace=False)
        return X[:, selected_features], selected_features

    def train_models(self):
        # Train models on random subspaces
        trained_models = {}
        for name, model in self.models.items():
            subspace_models = []
            for _ in range(self.n_subspaces):
                X_subspace, selected_features = self.random_subspace(self.X_train.values)
                subspace_model = model.__class__(**model.get_params())
                subspace_model.fit(X_subspace, self.y_train)
                subspace_models.append((subspace_model, selected_features))
            trained_models[name] = subspace_models
        return trained_models

    def calculate_model_weights(self):
        # Calculate weights for each model based on log-likelihood
        weights = {}
        epsilon = 1e-15  # Small value to prevent log(0)
        for name, subspace_models in self.trained_models.items():
            log_likelihood = 0
            for model, selected_features in subspace_models:
                X_test_subspace = self.X_test.values[:, selected_features]
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test_subspace)
                    log_likelihood += np.sum(np.log(y_pred_proba[np.arange(len(self.y_test)), self.y_test] + epsilon))
                else:
                    y_pred = model.predict(X_test_subspace)
                    log_likelihood += np.sum(np.log((y_pred == self.y_test).astype(float) + epsilon))
            
            weights[name] = log_likelihood
        
        # Normalize weights
        total_weight = logsumexp(list(weights.values()))
        for name in weights:
            weights[name] = np.exp(weights[name] - total_weight)
        
        return weights

    def predict(self, X):
        # Make predictions using the ensemble of models
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, (name, subspace_models) in enumerate(self.trained_models.items()):
            subspace_predictions = np.zeros((X.shape[0], len(subspace_models)))
            for j, (model, selected_features) in enumerate(subspace_models):
                X_subspace = X[:, selected_features]
                if hasattr(model, 'predict_proba'):
                    subspace_predictions[:, j] = model.predict_proba(X_subspace)[:, 1]
                else:
                    subspace_predictions[:, j] = model.predict(X_subspace)
            predictions[:, i] = np.mean(subspace_predictions, axis=1)
        
        self.individual_predictions = predictions  # Store individual model predictions
        weighted_predictions = np.sum(predictions * np.array(list(self.model_weights.values())), axis=1)
        return weighted_predictions

    def evaluate_model(self):
        # Evaluate the ensemble model
        y_pred_proba = self.predict(self.X_test.values)
        y_pred = (y_pred_proba > self.threshold).astype(int)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        return accuracy, precision, recall, f1, tp, fp, fn, tn, y_pred_proba

    def evaluate_individual_models(self):
        # Evaluate each individual model
        individual_metrics = {}
        for i, model_name in enumerate(self.models.keys()):
            y_pred = (self.individual_predictions[:, i] > self.threshold).astype(int)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            individual_metrics[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        return individual_metrics

    def print_results(self):
        # Print results and generate plots
        print("Uploaded Model Statistics Successfully.")
        metrics = self.evaluate_model()
        individual_metrics = self.evaluate_individual_models()
        self.plot_results(metrics, individual_metrics)

    def plot_bias_variance(self):
        n_samples = len(self.X_train)
        train_sizes = np.linspace(0.1, 0.99, 10)
        
        plt.figure(figsize=(12, 8))
        
        for name, model in self.models.items():
            train_scores, test_scores = [], []
            for train_size in train_sizes:
                X_subset, _, y_subset, _ = train_test_split(self.X_train, self.y_train, train_size=train_size, random_state=42)
                
                # Train scores
                train_score = cross_val_score(model, X_subset, y_subset, cv=2).mean()  # Changed to 2 folds
                train_scores.append(train_score)
                
                # Test scores
                test_score = cross_val_score(model, self.X_test, self.y_test, cv=2).mean()  # Changed to 2 folds
                test_scores.append(test_score)
            
            plt.plot(train_sizes, np.array(train_scores), 'o-', label=f'{name} (train)')
            plt.plot(train_sizes, np.array(test_scores), 's-', label=f'{name} (test)')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title('Bias-Variance Tradeoff')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('C:/Xampp/htdocs/net-cure-website/model_plots/bias_variance_tradeoff.png')
        plt.close()

        # Learning Curves
        plt.figure(figsize=(12, 8))
        
        for name, model in self.models.items():
            train_sizes_abs, train_scores, test_scores = learning_curve(
                model, self.X_train, self.y_train, cv=2,  # Changed to 2 folds
                n_jobs=-1, 
                train_sizes=np.linspace(0.1, 0.99, 10),
                random_state=42)
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            plt.plot(train_sizes_abs, train_mean, 'o-', label=f'{name} (train)')
            plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1)
            plt.plot(train_sizes_abs, test_mean, 's-', label=f'{name} (test)')
            plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1)
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('C:/Xampp/htdocs/net-cure-website/model_plots/learning_curves.png')
        plt.close()

    def plot_results(self, metrics, individual_metrics):
        # Bias-Variance Tradeoff Plot
        self.plot_bias_variance()

        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        y_pred_proba = self.predict(self.X_test.values)
        y_pred = (y_pred_proba > self.threshold).astype(int)  # Apply threshold
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('C:/Xampp/htdocs/net-cure-website/model_plots/confusion_matrix.png')
        plt.close()

        # Model Weights
        plt.figure(figsize=(10, 6))
        plt.bar(self.model_weights.keys(), self.model_weights.values())
        plt.title('Model Weights')
        plt.xlabel('Models')
        plt.ylabel('Weight')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('C:/Xampp/htdocs/net-cure-website/model_plots/model_weights.png')
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
        plt.savefig('C:/Xampp/htdocs/net-cure-website/model_plots/feature_importance.png')
        plt.close()

        # Performance Metrics
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = metrics[:4]
        plt.figure(figsize=(10, 6))
        plt.bar(metrics_names, metrics_values)
        plt.title('Proposed Model Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        for i, v in enumerate(metrics_values):
            plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig('C:/Xampp/htdocs/net-cure-website/model_plots/performance_metrics.png')
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
        plt.savefig('C:/Xampp/htdocs/net-cure-website/model_plots/roc_curve.png')
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
        plt.savefig('C:/Xampp/htdocs/net-cure-website/model_plots/precision_recall_curve.png')
        plt.close()

        # Calibration Plot
        fraction_of_positives, mean_predicted_value = calibration_curve(self.y_test, metrics[8], n_bins=10)
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Proposed model")
        plt.ylabel("Fraction of positives")
        plt.xlabel("Mean predicted value")
        plt.title("Calibration Plot")
        plt.legend(loc="best")
        plt.savefig('C:/Xampp/htdocs/net-cure-website/model_plots/calibration_plot.png')
        plt.close()

        # Boxplots of predictions
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=pd.DataFrame(self.individual_predictions, columns=self.models.keys()))
        plt.title('Boxplots of Individual Model Predictions')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('C:/Xampp/htdocs/net-cure-website/model_plots/boxplots_predictions.png')
        plt.close()

        # Comparing proposed model vs individual models
        proposed_metrics = {
            'accuracy': metrics[0],
            'precision': metrics[1],
            'recall': metrics[2],
            'f1': metrics[3]
        }

        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            plt.figure(figsize=(10, 6))
            model_names = list(individual_metrics.keys()) + ['Proposed Model']
            metric_values = [individual_metrics[model][metric] for model in individual_metrics] + [proposed_metrics[metric]]
            
            plt.bar(model_names, metric_values)
            plt.title(f'Proposed Model vs Individual Models: {metric.capitalize()}')
            plt.xlabel('Models')
            plt.ylabel(metric.capitalize())
            plt.ylim(0, 1)
            plt.xticks(rotation=45, ha='right')
            
            for i, v in enumerate(metric_values):
                plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f'C:/Xampp/htdocs/net-cure-website/model_plots/{metric}_comparison.png')
            plt.close()

# Usage
data_paths = [
    'C:/Xampp/htdocs/net-cure-website/dataset/ICMP_FLOOD.csv',
    'C:/Xampp/htdocs/net-cure-website/dataset/ICMP_BENIGN.csv'
]

# Create an instance of the Statistic class and print results
performance = Performance(data_paths)
performance.print_results()