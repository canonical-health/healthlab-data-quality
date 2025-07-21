"""
Core Data Quality Dimensions based on Kahn Framework

This module implements the three core dimensions of data quality:
1. Completeness - presence of data values
2. Conformance - adherence to format, type, and domain constraints  
3. Plausibility - believability and reasonableness of data values
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


class QualityDimensionType(Enum):
    """Types of data quality dimensions"""
    COMPLETENESS = "completeness"
    CONFORMANCE = "conformance"
    PLAUSIBILITY = "plausibility"


@dataclass
class QualityIssue:
    """Represents a data quality issue found during assessment"""
    dimension: QualityDimensionType
    severity: str  # 'critical', 'major', 'minor', 'warning'
    rule_name: str
    description: str
    field_name: str
    record_id: Optional[str] = None
    confidence: float = 1.0
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    suggestion: Optional[str] = None


@dataclass
class DimensionResult:
    """Results from assessing a single quality dimension"""
    dimension: QualityDimensionType
    score: float  # 0-100
    issues: List[QualityIssue]
    total_records: int
    failed_records: int
    rules_evaluated: int
    rules_passed: int


class QualityDimension(ABC):
    """Abstract base class for data quality dimensions"""
    
    def __init__(self, name: str, dimension_type: QualityDimensionType):
        self.name = name
        self.dimension_type = dimension_type
        self.rules = []
    
    @abstractmethod
    def assess(self, data: pd.DataFrame) -> DimensionResult:
        """Assess data quality for this dimension"""
        pass
    
    def add_rule(self, rule):
        """Add a validation rule to this dimension"""
        self.rules.append(rule)


class CompletenessDimension(QualityDimension):
    """
    Completeness Dimension - assesses presence of data values
    
    Evaluates:
    - Missing required fields
    - Null values in critical columns
    - Empty strings where values expected
    - Incomplete record structures
    """
    
    def __init__(self):
        super().__init__("Completeness", QualityDimensionType.COMPLETENESS)
        self.required_fields = set()
        self.critical_fields = set()
    
    def set_required_fields(self, fields: List[str]):
        """Set fields that must be present"""
        self.required_fields = set(fields)
    
    def set_critical_fields(self, fields: List[str]):
        """Set fields that should not be null/empty"""
        self.critical_fields = set(fields)
    
    def assess(self, data: pd.DataFrame) -> DimensionResult:
        """Assess completeness of the dataset"""
        issues = []
        total_records = len(data)
        failed_records = 0
        
        # Check required fields presence
        missing_fields = self.required_fields - set(data.columns)
        for field in missing_fields:
            issues.append(QualityIssue(
                dimension=QualityDimensionType.COMPLETENESS,
                severity='critical',
                rule_name='required_field_missing',
                description=f'Required field {field} is missing from dataset',
                field_name=field
            ))
        
        # Check for null/empty values in critical fields
        for field in self.critical_fields:
            if field in data.columns:
                null_count = data[field].isnull().sum()
                empty_count = (data[field] == '').sum() if data[field].dtype == 'object' else 0
                total_missing = null_count + empty_count
                
                if total_missing > 0:
                    issues.append(QualityIssue(
                        dimension=QualityDimensionType.COMPLETENESS,
                        severity='major' if total_missing > total_records * 0.1 else 'minor',
                        rule_name='critical_field_missing_values',
                        description=f'Field {field} has {total_missing} missing values ({total_missing/total_records:.1%})',
                        field_name=field,
                        actual_value=f"{total_missing} missing"
                    ))
        
        # Calculate overall completeness score
        failed_records = len(set(issue.record_id for issue in issues if issue.record_id))
        
        # Simple scoring: 100 - (percentage of fields with issues * severity multiplier)
        major_issues = sum(1 for issue in issues if issue.severity in ['critical', 'major'])
        minor_issues = sum(1 for issue in issues if issue.severity == 'minor')
        
        score = max(0, 100 - (major_issues * 20 + minor_issues * 5))
        
        return DimensionResult(
            dimension=QualityDimensionType.COMPLETENESS,
            score=score,
            issues=issues,
            total_records=total_records,
            failed_records=failed_records,
            rules_evaluated=len(self.required_fields) + len(self.critical_fields),
            rules_passed=len(self.required_fields) + len(self.critical_fields) - len(issues)
        )


class ConformanceDimension(QualityDimension):
    """
    Conformance Dimension - assesses adherence to format and domain constraints
    
    Evaluates:
    - Data type conformity
    - Format validation (dates, IDs, codes)
    - Domain value constraints
    - Pattern matching (regex)
    - Range constraints
    """
    
    def __init__(self):
        super().__init__("Conformance", QualityDimensionType.CONFORMANCE)
        self.type_constraints = {}
        self.format_patterns = {}
        self.domain_values = {}
        self.range_constraints = {}
    
    def add_type_constraint(self, field: str, expected_type: type):
        """Add type constraint for a field"""
        self.type_constraints[field] = expected_type
    
    def add_format_pattern(self, field: str, pattern: str, description: str = ""):
        """Add regex pattern constraint for a field"""
        self.format_patterns[field] = (pattern, description)
    
    def add_domain_values(self, field: str, valid_values: List[Any]):
        """Add valid domain values for a field"""
        self.domain_values[field] = set(valid_values)
    
    def add_range_constraint(self, field: str, min_val: Any = None, max_val: Any = None):
        """Add range constraint for numeric fields"""
        self.range_constraints[field] = (min_val, max_val)
    
    def assess(self, data: pd.DataFrame) -> DimensionResult:
        """Assess conformance of the dataset"""
        issues = []
        total_records = len(data)
        failed_record_ids = set()
        
        # Check type constraints
        for field, expected_type in self.type_constraints.items():
            if field in data.columns:
                if expected_type == int:
                    non_numeric = data[~pd.to_numeric(data[field], errors='coerce').notna()]
                    for idx, row in non_numeric.iterrows():
                        issues.append(QualityIssue(
                            dimension=QualityDimensionType.CONFORMANCE,
                            severity='major',
                            rule_name='type_constraint_violation',
                            description=f'Field {field} should be numeric',
                            field_name=field,
                            record_id=str(idx),
                            actual_value=row[field],
                            expected_value="numeric value"
                        ))
                        failed_record_ids.add(str(idx))
        
        # Check format patterns
        import re
        for field, (pattern, description) in self.format_patterns.items():
            if field in data.columns:
                invalid_rows = data[~data[field].astype(str).str.match(pattern, na=False)]
                for idx, row in invalid_rows.iterrows():
                    issues.append(QualityIssue(
                        dimension=QualityDimensionType.CONFORMANCE,
                        severity='major',
                        rule_name='format_pattern_violation',
                        description=f'Field {field} does not match expected format: {description}',
                        field_name=field,
                        record_id=str(idx),
                        actual_value=row[field]
                    ))
                    failed_record_ids.add(str(idx))
        
        # Check domain values
        for field, valid_values in self.domain_values.items():
            if field in data.columns:
                invalid_rows = data[~data[field].isin(valid_values)]
                for idx, row in invalid_rows.iterrows():
                    issues.append(QualityIssue(
                        dimension=QualityDimensionType.CONFORMANCE,
                        severity='major',
                        rule_name='domain_value_violation',
                        description=f'Field {field} contains invalid value',
                        field_name=field,
                        record_id=str(idx),
                        actual_value=row[field],
                        expected_value=f"One of: {list(valid_values)[:5]}{'...' if len(valid_values) > 5 else ''}"
                    ))
                    failed_record_ids.add(str(idx))
        
        # Check range constraints
        for field, (min_val, max_val) in self.range_constraints.items():
            if field in data.columns:
                numeric_data = pd.to_numeric(data[field], errors='coerce')
                if min_val is not None:
                    below_min = data[numeric_data < min_val]
                    for idx, row in below_min.iterrows():
                        issues.append(QualityIssue(
                            dimension=QualityDimensionType.CONFORMANCE,
                            severity='major',
                            rule_name='range_constraint_violation',
                            description=f'Field {field} value below minimum ({min_val})',
                            field_name=field,
                            record_id=str(idx),
                            actual_value=row[field],
                            expected_value=f">= {min_val}"
                        ))
                        failed_record_ids.add(str(idx))
                
                if max_val is not None:
                    above_max = data[numeric_data > max_val]
                    for idx, row in above_max.iterrows():
                        issues.append(QualityIssue(
                            dimension=QualityDimensionType.CONFORMANCE,
                            severity='major',
                            rule_name='range_constraint_violation',
                            description=f'Field {field} value above maximum ({max_val})',
                            field_name=field,
                            record_id=str(idx),
                            actual_value=row[field],
                            expected_value=f"<= {max_val}"
                        ))
                        failed_record_ids.add(str(idx))
        
        # Calculate score
        rules_count = (len(self.type_constraints) + len(self.format_patterns) + 
                      len(self.domain_values) + len(self.range_constraints))
        rules_passed = rules_count - len(issues)
        
        score = max(0, 100 - (len(issues) / max(total_records, 1)) * 100)
        
        return DimensionResult(
            dimension=QualityDimensionType.CONFORMANCE,
            score=score,
            issues=issues,
            total_records=total_records,
            failed_records=len(failed_record_ids),
            rules_evaluated=rules_count,
            rules_passed=rules_passed
        )


class PlausibilityDimension(QualityDimension):
    """
    Plausibility Dimension - assesses believability and reasonableness of data
    
    Evaluates:
    - Clinical logic violations (e.g., pregnancy in males)
    - Temporal inconsistencies
    - Statistical outliers
    - Cross-field dependencies
    - Medical knowledge violations
    """
    
    def __init__(self):
        super().__init__("Plausibility", QualityDimensionType.PLAUSIBILITY)
        self.clinical_rules = []
        self.temporal_rules = []
        self.dependency_rules = []
        self.outlier_thresholds = {}
    
    def add_clinical_rule(self, rule_func, name: str, description: str):
        """Add clinical logic rule"""
        self.clinical_rules.append((rule_func, name, description))
    
    def add_temporal_rule(self, rule_func, name: str, description: str):
        """Add temporal consistency rule"""
        self.temporal_rules.append((rule_func, name, description))
    
    def add_dependency_rule(self, rule_func, name: str, description: str):
        """Add cross-field dependency rule"""
        self.dependency_rules.append((rule_func, name, description))
    
    def add_outlier_threshold(self, field: str, z_threshold: float = 3.0):
        """Add statistical outlier detection for field"""
        self.outlier_thresholds[field] = z_threshold
    
    def assess(self, data: pd.DataFrame) -> DimensionResult:
        """Assess plausibility of the dataset"""
        issues = []
        total_records = len(data)
        failed_record_ids = set()
        
        # Apply clinical rules
        for rule_func, name, description in self.clinical_rules:
            try:
                violations = rule_func(data)
                for idx in violations:
                    issues.append(QualityIssue(
                        dimension=QualityDimensionType.PLAUSIBILITY,
                        severity='critical',
                        rule_name=name,
                        description=description,
                        field_name='clinical_logic',
                        record_id=str(idx)
                    ))
                    failed_record_ids.add(str(idx))
            except Exception as e:
                print(f"Error applying clinical rule {name}: {e}")
        
        # Apply temporal rules
        for rule_func, name, description in self.temporal_rules:
            try:
                violations = rule_func(data)
                for idx in violations:
                    issues.append(QualityIssue(
                        dimension=QualityDimensionType.PLAUSIBILITY,
                        severity='major',
                        rule_name=name,
                        description=description,
                        field_name='temporal_logic',
                        record_id=str(idx)
                    ))
                    failed_record_ids.add(str(idx))
            except Exception as e:
                print(f"Error applying temporal rule {name}: {e}")
        
        # Apply dependency rules
        for rule_func, name, description in self.dependency_rules:
            try:
                violations = rule_func(data)
                for idx in violations:
                    issues.append(QualityIssue(
                        dimension=QualityDimensionType.PLAUSIBILITY,
                        severity='major',
                        rule_name=name,
                        description=description,
                        field_name='dependency_logic',
                        record_id=str(idx)
                    ))
                    failed_record_ids.add(str(idx))
            except Exception as e:
                print(f"Error applying dependency rule {name}: {e}")
        
        # Statistical outlier detection
        for field, z_threshold in self.outlier_thresholds.items():
            if field in data.columns:
                numeric_data = pd.to_numeric(data[field], errors='coerce')
                if numeric_data.notna().sum() > 0:
                    z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())
                    outliers = data[z_scores > z_threshold]
                    for idx, row in outliers.iterrows():
                        issues.append(QualityIssue(
                            dimension=QualityDimensionType.PLAUSIBILITY,
                            severity='minor',
                            rule_name='statistical_outlier',
                            description=f'Field {field} is a statistical outlier (z-score > {z_threshold})',
                            field_name=field,
                            record_id=str(idx),
                            actual_value=row[field],
                            confidence=min(1.0, z_scores.iloc[idx] / 10)  # Confidence based on z-score
                        ))
        
        # Calculate score
        rules_count = (len(self.clinical_rules) + len(self.temporal_rules) + 
                      len(self.dependency_rules) + len(self.outlier_thresholds))
        
        critical_issues = sum(1 for issue in issues if issue.severity == 'critical')
        major_issues = sum(1 for issue in issues if issue.severity == 'major')
        minor_issues = sum(1 for issue in issues if issue.severity == 'minor')
        
        score = max(0, 100 - (critical_issues * 30 + major_issues * 15 + minor_issues * 5))
        
        return DimensionResult(
            dimension=QualityDimensionType.PLAUSIBILITY,
            score=score,
            issues=issues,
            total_records=total_records,
            failed_records=len(failed_record_ids),
            rules_evaluated=rules_count,
            rules_passed=rules_count - len(issues)
        )
