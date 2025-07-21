"""
Healthcare Data Quality Assessment Framework

This package provides a comprehensive framework for assessing healthcare data quality
based on Kahn's framework dimensions: Completeness, Conformance, and Plausibility.

The framework includes:
- Rule-based validation for known data quality issues
- AI/ML-based anomaly detection for unknown issues
- Synthetic healthcare data generation for testing
- Comprehensive reporting and scorecards
"""

__version__ = "0.1.0"
__author__ = "HealthLab Research Team"

from .core.framework import DataQualityFramework
from .core.dimensions import CompletenessDimension, ConformanceDimension, PlausibilityDimension
from .validators.rule_based import RuleBasedValidator
from .validators.ml_based import MLAnomalyDetector
from .data.synthetic_generator import SyntheticFHIRDataGenerator
from .reporting.scorecard import DataQualityScorecard

__all__ = [
    "DataQualityFramework",
    "CompletenessDimension",
    "ConformanceDimension", 
    "PlausibilityDimension",
    "RuleBasedValidator",
    "MLAnomalyDetector",
    "SyntheticFHIRDataGenerator",
    "DataQualityScorecard"
]
