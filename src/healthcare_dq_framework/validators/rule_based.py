"""
Rule-based Validators for Healthcare Data Quality

This module implements traditional rule-based validation approaches
for known data quality issues in healthcare data.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Callable, Any, Optional
from datetime import datetime, timedelta

from ..core.dimensions import QualityIssue, QualityDimensionType


class RuleBasedValidator:
    """
    Rule-based validator for healthcare data quality issues
    
    Implements deterministic rules for known data quality problems
    including format validation, range checks, and clinical logic rules.
    """
    
    def __init__(self):
        self.rules = {}
        self.rule_categories = {
            'format': [],
            'range': [],
            'clinical': [],
            'temporal': [],
            'reference': []
        }
    
    def add_rule(self, rule_name: str, rule_func: Callable, category: str = 'general',
                 severity: str = 'major', description: str = ""):
        """
        Add a validation rule
        
        Args:
            rule_name: Unique identifier for the rule
            rule_func: Function that takes DataFrame and returns list of issues
            category: Category of the rule (format, range, clinical, etc.)
            severity: Severity level (critical, major, minor, warning)
            description: Human-readable description of the rule
        """
        self.rules[rule_name] = {
            'function': rule_func,
            'category': category,
            'severity': severity,
            'description': description
        }
        
        if category in self.rule_categories:
            self.rule_categories[category].append(rule_name)
    
    def validate_all(self, data: pd.DataFrame) -> List[QualityIssue]:
        """Run all validation rules against the dataset"""
        all_issues = []
        
        for rule_name, rule_config in self.rules.items():
            try:
                issues = rule_config['function'](data)
                
                # Ensure issues have proper metadata
                for issue in issues:
                    if not hasattr(issue, 'rule_name') or not issue.rule_name:
                        issue.rule_name = rule_name
                    if not hasattr(issue, 'severity') or not issue.severity:
                        issue.severity = rule_config['severity']
                
                all_issues.extend(issues)
                
            except Exception as e:
                # Create error issue if rule fails
                error_issue = QualityIssue(
                    dimension=QualityDimensionType.CONFORMANCE,
                    severity='critical',
                    rule_name=f'{rule_name}_error',
                    description=f'Rule {rule_name} failed: {str(e)}',
                    field_name='system'
                )
                all_issues.append(error_issue)
        
        return all_issues
    
    def validate_category(self, data: pd.DataFrame, category: str) -> List[QualityIssue]:
        """Run validation rules for a specific category"""
        issues = []
        
        if category not in self.rule_categories:
            raise ValueError(f"Unknown category: {category}")
        
        for rule_name in self.rule_categories[category]:
            if rule_name in self.rules:
                try:
                    rule_issues = self.rules[rule_name]['function'](data)
                    issues.extend(rule_issues)
                except Exception as e:
                    print(f"Error running rule {rule_name}: {e}")
        
        return issues


class HealthcareRuleLibrary:
    """
    Library of common healthcare data quality validation rules
    """
    
    @staticmethod
    def create_patient_id_format_rule(pattern: str = r'^PAT\d{6}$'):
        """Create patient ID format validation rule"""
        def validate_patient_id_format(data: pd.DataFrame) -> List[QualityIssue]:
            issues = []
            if 'patient_id' in data.columns:
                invalid_ids = data[~data['patient_id'].str.match(pattern, na=False)]
                for idx, row in invalid_ids.iterrows():
                    issues.append(QualityIssue(
                        dimension=QualityDimensionType.CONFORMANCE,
                        severity='major',
                        rule_name='patient_id_format',
                        description=f'Patient ID does not match required format: {pattern}',
                        field_name='patient_id',
                        record_id=str(idx),
                        actual_value=row['patient_id']
                    ))
            return issues
        return validate_patient_id_format
    
    @staticmethod
    def create_age_range_rule(min_age: int = 0, max_age: int = 150):
        """Create age range validation rule"""
        def validate_age_range(data: pd.DataFrame) -> List[QualityIssue]:
            issues = []
            if 'age' in data.columns:
                invalid_ages = data[(data['age'] < min_age) | (data['age'] > max_age)]
                for idx, row in invalid_ages.iterrows():
                    issues.append(QualityIssue(
                        dimension=QualityDimensionType.PLAUSIBILITY,
                        severity='critical',
                        rule_name='age_range_violation',
                        description=f'Age {row["age"]} is outside valid range ({min_age}-{max_age})',
                        field_name='age',
                        record_id=str(idx),
                        actual_value=row['age'],
                        expected_value=f'{min_age}-{max_age}'
                    ))
            return issues
        return validate_age_range
    
    @staticmethod
    def create_vital_signs_range_rules():
        """Create vital signs range validation rules"""
        vital_ranges = {
            'temperature_c': (30.0, 45.0),
            'heart_rate': (30, 200),
            'systolic_bp': (50, 300),
            'diastolic_bp': (30, 200),
            'respiratory_rate': (5, 50),
            'oxygen_saturation': (50, 100)
        }
        
        def validate_vital_signs_ranges(data: pd.DataFrame) -> List[QualityIssue]:
            issues = []
            
            for vital, (min_val, max_val) in vital_ranges.items():
                if vital in data.columns:
                    invalid_vitals = data[(data[vital] < min_val) | (data[vital] > max_val)]
                    for idx, row in invalid_vitals.iterrows():
                        issues.append(QualityIssue(
                            dimension=QualityDimensionType.PLAUSIBILITY,
                            severity='major',
                            rule_name=f'{vital}_range_violation',
                            description=f'{vital} value {row[vital]} is outside plausible range ({min_val}-{max_val})',
                            field_name=vital,
                            record_id=str(idx),
                            actual_value=row[vital],
                            expected_value=f'{min_val}-{max_val}'
                        ))
            
            return issues
        return validate_vital_signs_ranges
    
    @staticmethod
    def create_pregnancy_male_rule():
        """Create rule to detect pregnancy diagnosis in male patients"""
        def validate_pregnancy_male(data: pd.DataFrame) -> List[QualityIssue]:
            issues = []
            
            if 'gender' in data.columns and 'icd10_code' in data.columns:
                # Pregnancy ICD-10 codes typically start with O (O00-O9A)
                pregnancy_pattern = r'^O[0-9][0-9A-Z]'
                
                pregnancy_codes = data['icd10_code'].str.match(pregnancy_pattern, na=False)
                male_patients = data['gender'] == 'M'
                
                violations = data[pregnancy_codes & male_patients]
                
                for idx, row in violations.iterrows():
                    issues.append(QualityIssue(
                        dimension=QualityDimensionType.PLAUSIBILITY,
                        severity='critical',
                        rule_name='pregnancy_in_male',
                        description=f'Pregnancy-related diagnosis ({row["icd10_code"]}) in male patient',
                        field_name='icd10_code',
                        record_id=str(idx),
                        actual_value=f"Gender: {row['gender']}, Code: {row['icd10_code']}"
                    ))
            
            return issues
        return validate_pregnancy_male
    
    @staticmethod
    def create_pediatric_adult_procedure_rule():
        """Create rule to detect adult procedures in pediatric patients"""
        adult_procedures = [
            '33361',  # Mammography
            '55700',  # Prostate biopsy
            '58150',  # Hysterectomy
            '19120'   # Breast excision
        ]
        
        def validate_pediatric_adult_procedures(data: pd.DataFrame) -> List[QualityIssue]:
            issues = []
            
            if 'age' in data.columns and 'cpt_code' in data.columns:
                pediatric_patients = data['age'] < 18
                adult_procedure_codes = data['cpt_code'].isin(adult_procedures)
                
                violations = data[pediatric_patients & adult_procedure_codes]
                
                for idx, row in violations.iterrows():
                    issues.append(QualityIssue(
                        dimension=QualityDimensionType.PLAUSIBILITY,
                        severity='critical',
                        rule_name='adult_procedure_in_pediatric',
                        description=f'Adult procedure ({row["cpt_code"]}) performed on pediatric patient (age {row["age"]})',
                        field_name='cpt_code',
                        record_id=str(idx),
                        actual_value=f"Age: {row['age']}, Procedure: {row['cpt_code']}"
                    ))
            
            return issues
        return validate_pediatric_adult_procedures
    
    @staticmethod
    def create_temporal_consistency_rule():
        """Create rule to validate temporal consistency"""
        def validate_temporal_consistency(data: pd.DataFrame) -> List[QualityIssue]:
            issues = []
            
            # Check admission/discharge date consistency
            if 'admission_date' in data.columns and 'discharge_date' in data.columns:
                try:
                    admission_dates = pd.to_datetime(data['admission_date'], errors='coerce')
                    discharge_dates = pd.to_datetime(data['discharge_date'], errors='coerce')
                    
                    # Find records where discharge is before admission
                    invalid_dates = data[discharge_dates < admission_dates]
                    
                    for idx, row in invalid_dates.iterrows():
                        issues.append(QualityIssue(
                            dimension=QualityDimensionType.PLAUSIBILITY,
                            severity='critical',
                            rule_name='discharge_before_admission',
                            description=f'Discharge date ({row["discharge_date"]}) is before admission date ({row["admission_date"]})',
                            field_name='discharge_date',
                            record_id=str(idx),
                            actual_value=row['discharge_date'],
                            expected_value=f'After {row["admission_date"]}'
                        ))
                except:
                    pass
            
            # Check birth date vs current age
            if 'birth_date' in data.columns and 'age' in data.columns:
                try:
                    birth_dates = pd.to_datetime(data['birth_date'], errors='coerce')
                    calculated_ages = (datetime.now() - birth_dates).dt.days // 365
                    
                    # Allow 2-year tolerance for age calculation differences
                    age_mismatches = data[abs(data['age'] - calculated_ages) > 2]
                    
                    for idx, row in age_mismatches.iterrows():
                        calculated_age = calculated_ages.iloc[idx]
                        issues.append(QualityIssue(
                            dimension=QualityDimensionType.PLAUSIBILITY,
                            severity='major',
                            rule_name='age_birth_date_mismatch',
                            description=f'Recorded age ({row["age"]}) does not match birth date (calculated: {calculated_age})',
                            field_name='age',
                            record_id=str(idx),
                            actual_value=row['age'],
                            expected_value=calculated_age
                        ))
                except:
                    pass
            
            return issues
        return validate_temporal_consistency
    
    @staticmethod
    def create_icd10_format_rule():
        """Create ICD-10 code format validation rule"""
        def validate_icd10_format(data: pd.DataFrame) -> List[QualityIssue]:
            issues = []
            
            if 'icd10_code' in data.columns:
                # ICD-10 pattern: Letter followed by 2 digits, optionally followed by decimal and 1-2 more digits
                icd10_pattern = r'^[A-Z]\d{2}(\.\d{1,2})?$'
                
                invalid_codes = data[~data['icd10_code'].str.match(icd10_pattern, na=False)]
                
                for idx, row in invalid_codes.iterrows():
                    issues.append(QualityIssue(
                        dimension=QualityDimensionType.CONFORMANCE,
                        severity='major',
                        rule_name='icd10_format_violation',
                        description=f'ICD-10 code does not match expected format: {icd10_pattern}',
                        field_name='icd10_code',
                        record_id=str(idx),
                        actual_value=row['icd10_code']
                    ))
            
            return issues
        return validate_icd10_format
    
    @staticmethod
    def create_required_fields_rule(required_fields: List[str]):
        """Create rule to check for required fields"""
        def validate_required_fields(data: pd.DataFrame) -> List[QualityIssue]:
            issues = []
            
            for field in required_fields:
                if field not in data.columns:
                    issues.append(QualityIssue(
                        dimension=QualityDimensionType.COMPLETENESS,
                        severity='critical',
                        rule_name='required_field_missing',
                        description=f'Required field {field} is missing from dataset',
                        field_name=field
                    ))
                else:
                    # Check for null/empty values
                    null_records = data[data[field].isnull()]
                    empty_records = data[data[field] == ''] if data[field].dtype == 'object' else pd.DataFrame()
                    
                    missing_records = pd.concat([null_records, empty_records]).drop_duplicates()
                    
                    for idx, row in missing_records.iterrows():
                        issues.append(QualityIssue(
                            dimension=QualityDimensionType.COMPLETENESS,
                            severity='major',
                            rule_name='required_field_empty',
                            description=f'Required field {field} is empty',
                            field_name=field,
                            record_id=str(idx)
                        ))
            
            return issues
        return validate_required_fields
    
    @staticmethod
    def create_reference_integrity_rule():
        """Create rule to check reference integrity between related records"""
        def validate_reference_integrity(data: pd.DataFrame) -> List[QualityIssue]:
            issues = []
            
            # Check if encounter references valid patient
            if 'patient_id' in data.columns and 'encounter_id' in data.columns:
                # This is a simplified check - in practice you'd check against a patient table
                patient_ids = data['patient_id'].dropna().unique()
                
                for idx, row in data.iterrows():
                    if pd.notna(row['patient_id']) and row['patient_id'] not in patient_ids:
                        issues.append(QualityIssue(
                            dimension=QualityDimensionType.CONFORMANCE,
                            severity='critical',
                            rule_name='invalid_patient_reference',
                            description=f'Invalid patient_id reference: {row["patient_id"]}',
                            field_name='patient_id',
                            record_id=str(idx),
                            actual_value=row['patient_id']
                        ))
            
            return issues
        return validate_reference_integrity


def create_healthcare_validator() -> RuleBasedValidator:
    """
    Factory function to create a pre-configured healthcare rule validator
    """
    validator = RuleBasedValidator()
    library = HealthcareRuleLibrary()
    
    # Add format validation rules
    validator.add_rule(
        'patient_id_format',
        library.create_patient_id_format_rule(),
        'format',
        'major',
        'Validate patient ID format'
    )
    
    validator.add_rule(
        'icd10_format',
        library.create_icd10_format_rule(),
        'format',
        'major',
        'Validate ICD-10 code format'
    )
    
    # Add range validation rules
    validator.add_rule(
        'age_range',
        library.create_age_range_rule(),
        'range',
        'critical',
        'Validate age is within reasonable range'
    )
    
    validator.add_rule(
        'vital_signs_ranges',
        library.create_vital_signs_range_rules(),
        'range',
        'major',
        'Validate vital signs are within plausible ranges'
    )
    
    # Add clinical logic rules
    validator.add_rule(
        'pregnancy_male',
        library.create_pregnancy_male_rule(),
        'clinical',
        'critical',
        'Detect pregnancy diagnosis in male patients'
    )
    
    validator.add_rule(
        'pediatric_adult_procedures',
        library.create_pediatric_adult_procedure_rule(),
        'clinical',
        'critical',
        'Detect adult procedures in pediatric patients'
    )
    
    # Add temporal consistency rules
    validator.add_rule(
        'temporal_consistency',
        library.create_temporal_consistency_rule(),
        'temporal',
        'major',
        'Validate temporal consistency of dates'
    )
    
    # Add completeness rules
    validator.add_rule(
        'required_fields',
        library.create_required_fields_rule(['patient_id', 'encounter_id']),
        'completeness',
        'critical',
        'Check for required fields'
    )
    
    # Add reference integrity rules
    validator.add_rule(
        'reference_integrity',
        library.create_reference_integrity_rule(),
        'reference',
        'critical',
        'Validate reference integrity'
    )
    
    return validator
