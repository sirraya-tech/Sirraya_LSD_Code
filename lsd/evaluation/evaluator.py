import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, accuracy_score, 
    confusion_matrix, roc_curve, precision_recall_curve, auc,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from ..core.config import LayerwiseSemanticDynamicsConfig
from ..utils.helpers import logger, SEED

class ComprehensiveEvaluator:
    """Comprehensive evaluation with multiple strategies."""
    
    def __init__(self, config: LayerwiseSemanticDynamicsConfig):
        self.config = config
    
    def compute_composite_score(self, metrics: Dict[str, float]) -> float:
        """Compute weighted composite detection score from multiple metrics."""
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in self.config.composite_score_weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            score /= total_weight
        
        return float(score)
    
    def evaluate_supervised(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive supervised evaluation using multiple classifiers."""
        logger.log("Starting supervised evaluation...")
        
        # Prepare features and labels
        feature_columns = [
            'final_alignment', 'mean_alignment', 'max_alignment', 'convergence_layer',
            'stability', 'alignment_gain', 'mean_velocity', 'mean_acceleration',
            'oscillation_count'
        ]
        
        # Filter available features
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].values
        
        # Convert labels properly
        y = np.array([1 if label == 'factual' else 0 for label in df['label']])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define classifiers
        classifiers = {
            'LogisticRegression': LogisticRegression(random_state=SEED, max_iter=1000),
            'RandomForest': RandomForestClassifier(random_state=SEED, n_estimators=100),
            'GradientBoosting': GradientBoostingClassifier(random_state=SEED, n_estimators=100),
        }
        
        results = {}
        
        for clf_name, clf in classifiers.items():
            logger.log(f"Training {clf_name}...")
            
            # Train classifier
            clf.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = clf.predict(X_test_scaled)
            y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
            
            # Comprehensive metrics
            metrics = self._compute_comprehensive_metrics(y_test, y_pred, y_pred_proba)
            
            # Store test data for visualization
            metrics['y_true'] = [int(y) for y in y_test.tolist()]
            metrics['y_pred'] = [int(y) for y in y_pred.tolist()]
            metrics['y_pred_proba'] = [float(y) for y in y_pred_proba.tolist()]
            
            # Cross-validation scores
            try:
                cv_scores = cross_val_score(clf, X_train_scaled, y_train, 
                                          cv=min(self.config.cross_validation_folds, 5), 
                                          scoring='f1')
                metrics['cv_f1_mean'] = float(cv_scores.mean())
                metrics['cv_f1_std'] = float(cv_scores.std())
            except:
                metrics['cv_f1_mean'] = 0.0
                metrics['cv_f1_std'] = 0.0
            
            # Composite score
            metrics['composite_score'] = self.compute_composite_score(metrics)
            
            results[clf_name] = metrics
            
            logger.log(f"{clf_name} - F1: {metrics['f1']:.4f}, AUC-ROC: {metrics['auroc']:.4f}, "
                      f"Composite: {metrics['composite_score']:.4f}")
        
        return results
    
    def evaluate_unsupervised(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Unsupervised evaluation using clustering and anomaly detection."""
        logger.log("Starting unsupervised evaluation...")
        
        # Prepare features for clustering
        feature_columns = [
            'final_alignment', 'mean_alignment', 'stability', 'alignment_gain',
            'mean_velocity', 'mean_acceleration'
        ]
        
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].values
        
        # Handle NaN values
        X = np.nan_to_num(X)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = {}
        
        # K-means clustering
        try:
            kmeans = KMeans(n_clusters=2, random_state=SEED)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Evaluate clustering against ground truth
            true_labels = np.array([1 if label == 'factual' else 0 for label in df['label']])
            
            # Try both cluster mappings
            accuracy1 = accuracy_score(true_labels, cluster_labels)
            accuracy2 = accuracy_score(true_labels, 1 - cluster_labels)
            clustering_accuracy = float(max(accuracy1, accuracy2))
            
            results['clustering_accuracy'] = clustering_accuracy
        except Exception as e:
            logger.log(f"Clustering failed: {e}", "WARNING")
            results['clustering_accuracy'] = 0.0
        
        # Gaussian Mixture Model for anomaly detection
        try:
            gmm = GaussianMixture(n_components=2, random_state=SEED)
            gmm.fit(X_scaled)
            anomaly_scores = gmm.score_samples(X_scaled)
            
            results['anomaly_scores_mean'] = float(np.mean(anomaly_scores))
            results['anomaly_scores_std'] = float(np.std(anomaly_scores))
        except Exception as e:
            logger.log(f"GMM failed: {e}", "WARNING")
            results['anomaly_scores_mean'] = 0.0
            results['anomaly_scores_std'] = 0.0
        
        logger.log(f"Unsupervised analysis - Clustering accuracy: {results.get('clustering_accuracy', 0.0):.4f}")
        
        return results
    
    def evaluate_hybrid(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Hybrid evaluation combining supervised and unsupervised approaches."""
        supervised_results = self.evaluate_supervised(df)
        unsupervised_results = self.evaluate_unsupervised(df)
        
        # Combine results
        hybrid_results = {
            'supervised': supervised_results,
            'unsupervised': unsupervised_results,
            'hybrid_metrics': {}
        }
        
        # Compute hybrid metrics
        if supervised_results:
            best_supervised_score = float(max(
                result['composite_score'] for result in supervised_results.values()
            ))
            hybrid_results['hybrid_metrics']['best_supervised_score'] = best_supervised_score
            
            # Overall hybrid score
            hybrid_results['hybrid_metrics']['overall_hybrid_score'] = float(
                0.7 * best_supervised_score +
                0.3 * unsupervised_results.get('clustering_accuracy', 0.0)
            )
        
        return hybrid_results
    
    def _compute_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        try:
            # Basic metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            accuracy = accuracy_score(y_true, y_pred)
            
            # Confusion matrix components
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Advanced metrics
            auroc = roc_auc_score(y_true, y_pred_proba)
            
            # Precision-recall curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
            prauc = auc(recall_curve, precision_curve)
            
            # Additional metrics
            mcc = matthews_corrcoef(y_true, y_pred)
            kappa = cohen_kappa_score(y_true, y_pred)
            
            # F2 score (emphasizes recall)
            f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
            
            return {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'f2': float(f2),
                'accuracy': float(accuracy),
                'specificity': float(specificity),
                'auroc': float(auroc),
                'prauc': float(prauc),
                'mcc': float(mcc),
                'kappa': float(kappa),
                'confusion_matrix': {
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'true_positives': int(tp)
                }
            }
        except Exception as e:
            logger.log(f"Error computing metrics: {e}", "WARNING")
            # Return default metrics
            return {
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'f2': 0.0,
                'accuracy': 0.0, 'specificity': 0.0, 'auroc': 0.0, 'prauc': 0.0,
                'mcc': 0.0, 'kappa': 0.0,
                'confusion_matrix': {'true_negatives': 0, 'false_positives': 0, 
                                   'false_negatives': 0, 'true_positives': 0}
            }