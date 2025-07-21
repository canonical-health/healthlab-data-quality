"""
Machine Learning-based Anomaly Detection for Healthcare Data Quality

This module implements AI/ML approaches to detect unknown data quality issues
through anomaly detection, pattern recognition, and statistical learning.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ..core.dimensions import QualityIssue, QualityDimensionType


class MLAnomalyDetector:
    """
    Machine Learning-based anomaly detector for healthcare data quality
    
    Uses multiple ML approaches:
    1. Isolation Forest for multivariate anomaly detection
    2. Statistical outlier detection (Z-score, IQR)
    3. Pattern-based anomaly detection
    4. Ensemble methods for improved accuracy
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize ML anomaly detector
        
        Args:
            contamination: Expected proportion of anomalies in data
            random_state: Random state for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        
        # ML models
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        
        # Statistical thresholds
        self.z_score_threshold = 3.0
        self.iqr_multiplier = 1.5
        
        # Training history
        self.training_stats = {}
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML analysis
        
        Handles:
        - Encoding categorical variables
        - Scaling numerical variables
        - Feature engineering
        - Missing value handling
        """
        df = data.copy()
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove ID columns and other non-informative columns
        id_patterns = ['_id', 'id_', 'uuid', 'guid']
        id_cols = [col for col in df.columns if any(pattern in col.lower() for pattern in id_patterns)]
        
        # Handle datetime columns
        datetime_cols = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    # Extract useful features from datetime
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_day'] = df[col].dt.day
                    df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                    datetime_cols.append(col)
                    numeric_cols.extend([f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_dayofweek'])
                except:
                    continue
        
        # Remove original datetime and ID columns
        cols_to_remove = id_cols + datetime_cols
        df = df.drop(columns=cols_to_remove, errors='ignore')
        
        # Update column lists
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        # Handle missing values
        # For numeric: fill with median
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # For categorical: fill with mode or 'Unknown'
        for col in categorical_cols:
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)
        
        # Encode categorical variables
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Transform using existing encoder, handle unknown categories
                known_categories = set(self.label_encoders[col].classes_)
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(lambda x: x if x in known_categories else 'Unknown')
                
                # Add 'Unknown' to encoder if not present
                if 'Unknown' not in known_categories:
                    self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, 'Unknown')
                
                df[col] = self.label_encoders[col].transform(df[col])
        
        # Store feature columns for later use
        self.feature_columns = df.columns.tolist()
        
        return df
    
    def train(self, data: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the ML anomaly detection models
        
        Args:
            data: Training data
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training statistics and performance metrics
        """
        print("Training ML anomaly detection models...")
        
        # Prepare features
        features_df = self.prepare_features(data)
        
        # Split data for training and validation
        if validation_split > 0:
            train_features, val_features = train_test_split(
                features_df, test_size=validation_split, random_state=self.random_state
            )
        else:
            train_features = features_df
            val_features = None
        
        # Scale features
        train_scaled = self.scaler.fit_transform(train_features)
        
        # Train Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        self.isolation_forest.fit(train_scaled)
        
        # Calculate training statistics
        train_scores = self.isolation_forest.decision_function(train_scaled)
        train_predictions = self.isolation_forest.predict(train_scaled)
        
        training_stats = {
            'training_samples': len(train_features),
            'feature_count': len(self.feature_columns),
            'anomaly_rate': (train_predictions == -1).mean(),
            'mean_anomaly_score': train_scores.mean(),
            'std_anomaly_score': train_scores.std(),
            'training_date': datetime.now().isoformat()
        }
        
        # Validation metrics
        if val_features is not None:
            val_scaled = self.scaler.transform(val_features)
            val_scores = self.isolation_forest.decision_function(val_scaled)
            val_predictions = self.isolation_forest.predict(val_scaled)
            
            training_stats.update({
                'validation_samples': len(val_features),
                'validation_anomaly_rate': (val_predictions == -1).mean(),
                'validation_mean_score': val_scores.mean()
            })
        
        self.training_stats = training_stats
        self.is_trained = True
        
        print(f"✓ Model trained on {training_stats['training_samples']} samples")
        print(f"✓ Detected {training_stats['anomaly_rate']:.2%} anomalies in training data")
        
        return training_stats
    
    def detect_anomalies(self, data: pd.DataFrame) -> List[QualityIssue]:
        """
        Detect anomalies in new data using trained models
        
        Returns:
            List of quality issues found through ML detection
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before detecting anomalies")
        
        issues = []
        
        # Prepare features
        features_df = self.prepare_features(data)
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Isolation Forest anomaly detection
        anomaly_scores = self.isolation_forest.decision_function(features_scaled)
        anomaly_predictions = self.isolation_forest.predict(features_scaled)
        
        # Find anomalies
        anomaly_indices = np.where(anomaly_predictions == -1)[0]
        
        for idx in anomaly_indices:
            # Calculate confidence based on anomaly score
            score = anomaly_scores[idx]
            confidence = min(1.0, abs(score) / 2.0)  # Normalize to 0-1 range
            
            issues.append(QualityIssue(
                dimension=QualityDimensionType.PLAUSIBILITY,
                severity='minor' if confidence < 0.7 else 'major',
                rule_name='ml_isolation_forest_anomaly',
                description=f'ML model detected anomalous pattern (score: {score:.3f})',
                field_name='multivariate_pattern',
                record_id=str(data.index[idx]),
                confidence=confidence
            ))
        
        # Statistical outlier detection
        issues.extend(self._detect_statistical_outliers(data))
        
        # Pattern-based anomaly detection
        issues.extend(self._detect_pattern_anomalies(data))
        
        return issues
    
    def _detect_statistical_outliers(self, data: pd.DataFrame) -> List[QualityIssue]:
        """Detect statistical outliers using Z-score and IQR methods"""
        issues = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if data[col].notna().sum() < 10:  # Skip if too few values
                continue
            
            # Z-score outliers
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            z_outliers = data[z_scores > self.z_score_threshold]
            
            for idx, row in z_outliers.iterrows():
                issues.append(QualityIssue(
                    dimension=QualityDimensionType.PLAUSIBILITY,
                    severity='minor',
                    rule_name='statistical_z_score_outlier',
                    description=f'Statistical outlier in {col} (Z-score: {z_scores.loc[idx]:.2f})',
                    field_name=col,
                    record_id=str(idx),
                    actual_value=row[col],
                    confidence=min(1.0, z_scores.loc[idx] / 5.0)
                ))
            
            # IQR outliers
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR
            
            iqr_outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            
            for idx, row in iqr_outliers.iterrows():
                # Skip if already flagged by Z-score
                if idx in z_outliers.index:
                    continue
                
                issues.append(QualityIssue(
                    dimension=QualityDimensionType.PLAUSIBILITY,
                    severity='minor',
                    rule_name='statistical_iqr_outlier',
                    description=f'IQR outlier in {col} (value: {row[col]}, expected: {lower_bound:.2f}-{upper_bound:.2f})',
                    field_name=col,
                    record_id=str(idx),
                    actual_value=row[col],
                    expected_value=f"{lower_bound:.2f}-{upper_bound:.2f}",
                    confidence=0.6
                ))
        
        return issues
    
    def _detect_pattern_anomalies(self, data: pd.DataFrame) -> List[QualityIssue]:
        """Detect pattern-based anomalies"""
        issues = []
        
        # Detect duplicate records (potential data quality issue)
        duplicates = data.duplicated(keep=False)
        for idx in data[duplicates].index:
            issues.append(QualityIssue(
                dimension=QualityDimensionType.PLAUSIBILITY,
                severity='warning',
                rule_name='duplicate_record_pattern',
                description='Potential duplicate record detected',
                field_name='record_pattern',
                record_id=str(idx),
                confidence=0.8
            ))
        
        # Detect impossible value combinations
        if 'age' in data.columns and 'birth_date' in data.columns:
            try:
                birth_dates = pd.to_datetime(data['birth_date'], errors='coerce')
                calculated_ages = (datetime.now() - birth_dates).dt.days // 365
                age_diff = abs(data['age'] - calculated_ages)
                
                inconsistent_ages = data[age_diff > 2]  # Allow 2 year tolerance
                for idx, row in inconsistent_ages.iterrows():
                    issues.append(QualityIssue(
                        dimension=QualityDimensionType.PLAUSIBILITY,
                        severity='major',
                        rule_name='age_birth_date_inconsistency',
                        description=f'Age ({row["age"]}) inconsistent with birth date',
                        field_name='age',
                        record_id=str(idx),
                        actual_value=row['age'],
                        confidence=0.9
                    ))
            except:
                pass
        
        return issues
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'training_stats': self.training_stats,
            'contamination': self.contamination,
            'z_score_threshold': self.z_score_threshold,
            'iqr_multiplier': self.iqr_multiplier
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        
        self.isolation_forest = model_data['isolation_forest']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.training_stats = model_data['training_stats']
        self.contamination = model_data['contamination']
        self.z_score_threshold = model_data['z_score_threshold']
        self.iqr_multiplier = model_data['iqr_multiplier']
        self.is_trained = True
        
        print(f"Model loaded from: {filepath}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores (approximate for ensemble methods)
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            return {}
        
        # For Isolation Forest, we can use path lengths as a proxy for importance
        # This is an approximation since IF doesn't provide direct feature importance
        
        importance_scores = {}
        for i, feature in enumerate(self.feature_columns):
            # Use random forest-like approach to estimate importance
            # This is a simplified version - in practice, you might want more sophisticated methods
            importance_scores[feature] = 1.0 / len(self.feature_columns)
        
        return importance_scores
    
    def explain_anomaly(self, data: pd.DataFrame, record_index: int) -> Dict[str, Any]:
        """
        Provide explanation for why a specific record was flagged as anomalous
        
        Args:
            data: The dataset
            record_index: Index of the record to explain
            
        Returns:
            Dictionary with explanation details
        """
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        try:
            # Get the specific record
            record = data.iloc[record_index:record_index+1]
            features_df = self.prepare_features(record)
            features_scaled = self.scaler.transform(features_df)
            
            # Get anomaly score
            anomaly_score = self.isolation_forest.decision_function(features_scaled)[0]
            
            # Calculate which features deviate most from the norm
            record_features = features_scaled[0]
            feature_deviations = {}
            
            for i, feature_name in enumerate(self.feature_columns):
                feature_deviations[feature_name] = abs(record_features[i])
            
            # Sort by deviation
            sorted_deviations = sorted(feature_deviations.items(), key=lambda x: x[1], reverse=True)
            
            explanation = {
                'record_index': record_index,
                'anomaly_score': anomaly_score,
                'is_anomaly': anomaly_score < 0,
                'top_contributing_features': sorted_deviations[:5],
                'feature_values': dict(zip(self.feature_columns, record_features))
            }
            
            return explanation
            
        except Exception as e:
            return {"error": f"Failed to explain anomaly: {str(e)}"}


class EnsembleAnomalyDetector:
    """
    Ensemble approach combining multiple anomaly detection methods
    for improved accuracy and robustness
    """
    
    def __init__(self):
        self.detectors = {}
        self.weights = {}
        self.is_trained = False
    
    def add_detector(self, name: str, detector: MLAnomalyDetector, weight: float = 1.0):
        """Add a detector to the ensemble"""
        self.detectors[name] = detector
        self.weights[name] = weight
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train all detectors in the ensemble"""
        training_results = {}
        
        for name, detector in self.detectors.items():
            print(f"Training {name} detector...")
            results = detector.train(data)
            training_results[name] = results
        
        self.is_trained = True
        return training_results
    
    def detect_anomalies(self, data: pd.DataFrame) -> List[QualityIssue]:
        """Detect anomalies using ensemble voting"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before detecting anomalies")
        
        all_issues = []
        issue_votes = {}
        
        # Collect issues from all detectors
        for name, detector in self.detectors.items():
            detector_issues = detector.detect_anomalies(data)
            
            for issue in detector_issues:
                issue_key = (issue.record_id, issue.field_name, issue.rule_name)
                
                if issue_key not in issue_votes:
                    issue_votes[issue_key] = {
                        'issue': issue,
                        'votes': 0,
                        'total_confidence': 0
                    }
                
                weight = self.weights.get(name, 1.0)
                issue_votes[issue_key]['votes'] += weight
                issue_votes[issue_key]['total_confidence'] += issue.confidence * weight
        
        # Filter issues based on ensemble voting
        min_votes = sum(self.weights.values()) * 0.3  # Require 30% of total votes
        
        for issue_data in issue_votes.values():
            if issue_data['votes'] >= min_votes:
                issue = issue_data['issue']
                # Update confidence based on ensemble agreement
                issue.confidence = issue_data['total_confidence'] / issue_data['votes']
                all_issues.append(issue)
        
        return all_issues
