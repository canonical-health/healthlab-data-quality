"""
Core Data Quality Assessment Framework

This module provides the main framework class that orchestrates the assessment
of healthcare data quality across all three Kahn dimensions.
"""

from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
import json

from .dimensions import (
    QualityDimension, DimensionResult, QualityDimensionType,
    CompletenessDimension, ConformanceDimension, PlausibilityDimension
)


class DataQualityFramework:
    """
    Main framework class for healthcare data quality assessment
    
    Orchestrates assessment across all three Kahn dimensions:
    - Completeness
    - Conformance  
    - Plausibility
    """
    
    def __init__(self):
        self.dimensions: Dict[QualityDimensionType, QualityDimension] = {
            QualityDimensionType.COMPLETENESS: CompletenessDimension(),
            QualityDimensionType.CONFORMANCE: ConformanceDimension(),
            QualityDimensionType.PLAUSIBILITY: PlausibilityDimension()
        }
        self.assessment_history = []
    
    def get_dimension(self, dimension_type: QualityDimensionType) -> QualityDimension:
        """Get a specific quality dimension"""
        return self.dimensions[dimension_type]
    
    def assess_all_dimensions(self, data: pd.DataFrame, 
                            assessment_name: str = None) -> Dict[QualityDimensionType, DimensionResult]:
        """
        Assess data quality across all dimensions
        
        Args:
            data: DataFrame containing healthcare data to assess
            assessment_name: Optional name for this assessment
            
        Returns:
            Dictionary mapping dimension types to their results
        """
        if assessment_name is None:
            assessment_name = f"Assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        results = {}
        
        # Assess each dimension
        for dim_type, dimension in self.dimensions.items():
            try:
                result = dimension.assess(data)
                results[dim_type] = result
                print(f"✓ {dim_type.value.title()} assessment completed - Score: {result.score:.1f}")
            except Exception as e:
                print(f"✗ Error assessing {dim_type.value}: {e}")
                # Create error result
                from .dimensions import DimensionResult, QualityIssue
                error_issue = QualityIssue(
                    dimension=dim_type,
                    severity='critical',
                    rule_name='assessment_error',
                    description=f'Error during assessment: {str(e)}',
                    field_name='system'
                )
                results[dim_type] = DimensionResult(
                    dimension=dim_type,
                    score=0.0,
                    issues=[error_issue],
                    total_records=len(data),
                    failed_records=len(data),
                    rules_evaluated=0,
                    rules_passed=0
                )
        
        # Store assessment in history
        assessment_record = {
            'name': assessment_name,
            'timestamp': datetime.now(),
            'total_records': len(data),
            'results': results,
            'overall_score': self._calculate_overall_score(results)
        }
        self.assessment_history.append(assessment_record)
        
        return results
    
    def _calculate_overall_score(self, results: Dict[QualityDimensionType, DimensionResult]) -> float:
        """Calculate weighted overall data quality score"""
        # Weight the dimensions (can be customized)
        weights = {
            QualityDimensionType.COMPLETENESS: 0.3,
            QualityDimensionType.CONFORMANCE: 0.4,
            QualityDimensionType.PLAUSIBILITY: 0.3
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for dim_type, result in results.items():
            if dim_type in weights:
                weighted_sum += result.score * weights[dim_type]
                total_weight += weights[dim_type]
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def get_latest_assessment(self) -> Optional[Dict]:
        """Get the most recent assessment results"""
        return self.assessment_history[-1] if self.assessment_history else None
    
    def get_assessment_summary(self) -> Dict:
        """Get summary statistics across all assessments"""
        if not self.assessment_history:
            return {"message": "No assessments performed yet"}
        
        latest = self.assessment_history[-1]
        
        summary = {
            'total_assessments': len(self.assessment_history),
            'latest_assessment': {
                'name': latest['name'],
                'timestamp': latest['timestamp'].isoformat(),
                'overall_score': latest['overall_score'],
                'total_records': latest['total_records']
            },
            'dimension_scores': {},
            'total_issues': 0,
            'critical_issues': 0,
            'major_issues': 0,
            'minor_issues': 0
        }
        
        # Aggregate dimension scores and issues
        for dim_type, result in latest['results'].items():
            summary['dimension_scores'][dim_type.value] = result.score
            summary['total_issues'] += len(result.issues)
            
            for issue in result.issues:
                if issue.severity == 'critical':
                    summary['critical_issues'] += 1
                elif issue.severity == 'major':
                    summary['major_issues'] += 1
                elif issue.severity == 'minor':
                    summary['minor_issues'] += 1
        
        return summary
    
    def configure_completeness_rules(self, required_fields: List[str], 
                                   critical_fields: List[str] = None):
        """Configure completeness dimension rules"""
        completeness_dim = self.get_dimension(QualityDimensionType.COMPLETENESS)
        completeness_dim.set_required_fields(required_fields)
        if critical_fields:
            completeness_dim.set_critical_fields(critical_fields)
    
    def configure_conformance_rules(self, type_constraints: Dict = None,
                                  format_patterns: Dict = None,
                                  domain_values: Dict = None,
                                  range_constraints: Dict = None):
        """Configure conformance dimension rules"""
        conformance_dim = self.get_dimension(QualityDimensionType.CONFORMANCE)
        
        if type_constraints:
            for field, expected_type in type_constraints.items():
                conformance_dim.add_type_constraint(field, expected_type)
        
        if format_patterns:
            for field, (pattern, description) in format_patterns.items():
                conformance_dim.add_format_pattern(field, pattern, description)
        
        if domain_values:
            for field, values in domain_values.items():
                conformance_dim.add_domain_values(field, values)
        
        if range_constraints:
            for field, (min_val, max_val) in range_constraints.items():
                conformance_dim.add_range_constraint(field, min_val, max_val)
    
    def configure_plausibility_rules(self, clinical_rules: List = None,
                                   temporal_rules: List = None,
                                   dependency_rules: List = None,
                                   outlier_fields: List[str] = None):
        """Configure plausibility dimension rules"""
        plausibility_dim = self.get_dimension(QualityDimensionType.PLAUSIBILITY)
        
        if clinical_rules:
            for rule_func, name, description in clinical_rules:
                plausibility_dim.add_clinical_rule(rule_func, name, description)
        
        if temporal_rules:
            for rule_func, name, description in temporal_rules:
                plausibility_dim.add_temporal_rule(rule_func, name, description)
        
        if dependency_rules:
            for rule_func, name, description in dependency_rules:
                plausibility_dim.add_dependency_rule(rule_func, name, description)
        
        if outlier_fields:
            for field in outlier_fields:
                plausibility_dim.add_outlier_threshold(field)
    
    def export_assessment_results(self, filepath: str, format: str = 'json'):
        """Export assessment results to file"""
        if not self.assessment_history:
            raise ValueError("No assessment results to export")
        
        latest = self.assessment_history[-1]
        
        # Convert results to serializable format
        export_data = {
            'assessment_name': latest['name'],
            'timestamp': latest['timestamp'].isoformat(),
            'overall_score': latest['overall_score'],
            'total_records': latest['total_records'],
            'dimensions': {}
        }
        
        for dim_type, result in latest['results'].items():
            export_data['dimensions'][dim_type.value] = {
                'score': result.score,
                'total_records': result.total_records,
                'failed_records': result.failed_records,
                'rules_evaluated': result.rules_evaluated,
                'rules_passed': result.rules_passed,
                'issues': [
                    {
                        'severity': issue.severity,
                        'rule_name': issue.rule_name,
                        'description': issue.description,
                        'field_name': issue.field_name,
                        'record_id': issue.record_id,
                        'confidence': issue.confidence,
                        'actual_value': str(issue.actual_value) if issue.actual_value is not None else None,
                        'expected_value': str(issue.expected_value) if issue.expected_value is not None else None
                    }
                    for issue in result.issues
                ]
            }
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        print(f"Assessment results exported to: {filepath}")


def create_healthcare_framework() -> DataQualityFramework:
    """
    Factory function to create a pre-configured healthcare data quality framework
    with common healthcare validation rules
    """
    framework = DataQualityFramework()
    
    # Configure common healthcare completeness rules
    framework.configure_completeness_rules(
        required_fields=['patient_id', 'encounter_id'],
        critical_fields=['patient_id', 'encounter_id', 'birth_date', 'gender']
    )
    
    # Configure common healthcare conformance rules
    framework.configure_conformance_rules(
        type_constraints={
            'age': int,
            'weight_kg': float,
            'height_cm': float,
            'temperature_c': float,
            'heart_rate': int,
            'systolic_bp': int,
            'diastolic_bp': int
        },
        format_patterns={
            'patient_id': (r'^PAT\d{6}$', 'Patient ID format: PAT followed by 6 digits'),
            'encounter_id': (r'^ENC\d{8}$', 'Encounter ID format: ENC followed by 8 digits'),
            'icd10_code': (r'^[A-Z]\d{2}(\.\d{1,2})?$', 'ICD-10 code format'),
            'cpt_code': (r'^\d{5}$', 'CPT code format: 5 digits')
        },
        domain_values={
            'gender': ['M', 'F', 'O', 'U'],  # Male, Female, Other, Unknown
            'encounter_type': ['inpatient', 'outpatient', 'emergency', 'observation'],
            'vital_status': ['alive', 'deceased', 'unknown']
        },
        range_constraints={
            'age': (0, 150),
            'weight_kg': (0, 500),
            'height_cm': (0, 300),
            'temperature_c': (30, 45),
            'heart_rate': (30, 200),
            'systolic_bp': (50, 300),
            'diastolic_bp': (30, 200)
        }
    )
    
    # Add some basic plausibility rules
    def pregnancy_male_check(data):
        """Check for pregnancy diagnosis in male patients"""
        violations = []
        if 'gender' in data.columns and 'icd10_code' in data.columns:
            # Pregnancy ICD-10 codes start with O00-O9A
            pregnancy_codes = data['icd10_code'].str.startswith('O', na=False)
            male_patients = data['gender'] == 'M'
            violations = data[pregnancy_codes & male_patients].index.tolist()
        return violations
    
    def age_birth_date_consistency(data):
        """Check age consistency with birth date"""
        violations = []
        if 'age' in data.columns and 'birth_date' in data.columns:
            try:
                birth_dates = pd.to_datetime(data['birth_date'], errors='coerce')
                calculated_ages = (datetime.now() - birth_dates).dt.days // 365
                age_diff = abs(data['age'] - calculated_ages)
                violations = data[age_diff > 1].index.tolist()  # Allow 1 year tolerance
            except:
                pass
        return violations
    
    framework.configure_plausibility_rules(
        clinical_rules=[
            (pregnancy_male_check, 'pregnancy_in_male', 'Pregnancy diagnosis in male patient'),
        ],
        temporal_rules=[
            (age_birth_date_consistency, 'age_birth_date_mismatch', 'Age does not match birth date'),
        ],
        outlier_fields=['weight_kg', 'height_cm', 'heart_rate', 'systolic_bp', 'diastolic_bp']
    )
    
    return framework
