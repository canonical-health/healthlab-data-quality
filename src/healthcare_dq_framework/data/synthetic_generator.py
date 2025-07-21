"""
Synthetic FHIR Data Generator

Generates synthetic healthcare data that resembles FHIR resources for testing
the data quality framework. Includes both clean data and data with intentional
quality issues to test the framework's detection capabilities.
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from typing import Dict, List, Optional, Tuple
import uuid


class SyntheticFHIRDataGenerator:
    """
    Generator for synthetic healthcare data resembling FHIR resources
    
    Generates:
    - Patient demographics
    - Encounters
    - Observations (vital signs, lab results)
    - Conditions (diagnoses)
    - Procedures
    """
    
    def __init__(self, seed: int = 42):
        """Initialize generator with optional seed for reproducibility"""
        self.fake = Faker()
        Faker.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Common medical data
        self.icd10_codes = [
            'I10', 'E11.9', 'Z00.00', 'I25.10', 'J44.1', 'N18.6', 'I50.9',
            'F41.9', 'M79.3', 'K21.9', 'J45.9', 'E78.5', 'I48.91', 'G93.1',
            'M25.50', 'R06.02', 'Z51.11', 'I35.0', 'N39.0', 'K59.00'
        ]
        
        self.cpt_codes = [
            '99213', '99214', '99232', '99233', '99291', '99292', '36415',
            '80053', '85025', '80061', '93000', '71020', '73030', '76700',
            '99396', '99397', '90471', '90686', '99204', '99205'
        ]
        
        self.encounter_types = ['inpatient', 'outpatient', 'emergency', 'observation']
        self.genders = ['M', 'F']
        self.vital_statuses = ['alive', 'deceased', 'unknown']
        
        # For introducing data quality issues
        self.quality_issue_rate = 0.1  # 10% of records will have issues
    
    def generate_patients(self, num_patients: int, 
                         introduce_issues: bool = True) -> pd.DataFrame:
        """Generate synthetic patient data"""
        patients = []
        
        for i in range(num_patients):
            # Generate base patient data
            birth_date = self.fake.date_of_birth(minimum_age=0, maximum_age=100)
            age = (datetime.now().date() - birth_date).days // 365
            
            patient = {
                'patient_id': f"PAT{i+1:06d}",
                'first_name': self.fake.first_name(),
                'last_name': self.fake.last_name(),
                'birth_date': birth_date.strftime('%Y-%m-%d'),
                'age': age,
                'gender': random.choice(self.genders),
                'address': self.fake.address().replace('\n', ', '),
                'phone': self.fake.phone_number(),
                'email': self.fake.email(),
                'vital_status': random.choice(self.vital_statuses),
                'mrn': f"MRN{random.randint(100000, 999999)}"
            }
            
            # Introduce data quality issues
            if introduce_issues and random.random() < self.quality_issue_rate:
                issue_type = random.choice(['missing', 'format', 'logic'])
                
                if issue_type == 'missing':
                    # Remove required field
                    field_to_remove = random.choice(['first_name', 'last_name', 'birth_date'])
                    patient[field_to_remove] = None
                
                elif issue_type == 'format':
                    # Introduce format violations
                    if random.random() < 0.5:
                        patient['patient_id'] = f"INVALID{i}"  # Wrong format
                    else:
                        patient['phone'] = "invalid-phone"  # Invalid phone
                
                elif issue_type == 'logic':
                    # Introduce logical inconsistencies
                    patient['age'] = random.randint(150, 200)  # Impossible age
            
            patients.append(patient)
        
        return pd.DataFrame(patients)
    
    def generate_encounters(self, patients_df: pd.DataFrame, 
                          encounters_per_patient: Tuple[int, int] = (1, 5),
                          introduce_issues: bool = True) -> pd.DataFrame:
        """Generate synthetic encounter data"""
        encounters = []
        encounter_counter = 1
        
        for _, patient in patients_df.iterrows():
            num_encounters = random.randint(*encounters_per_patient)
            
            for j in range(num_encounters):
                admission_date = self.fake.date_between(start_date='-2y', end_date='today')
                
                encounter = {
                    'encounter_id': f"ENC{encounter_counter:08d}",
                    'patient_id': patient['patient_id'],
                    'encounter_type': random.choice(self.encounter_types),
                    'admission_date': admission_date.strftime('%Y-%m-%d'),
                    'discharge_date': (admission_date + timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d'),
                    'primary_diagnosis': random.choice(self.icd10_codes),
                    'attending_physician': self.fake.name(),
                    'department': random.choice(['Cardiology', 'Emergency', 'Internal Medicine', 'Surgery', 'Pediatrics']),
                    'length_of_stay': random.randint(0, 30)
                }
                
                # Introduce data quality issues
                if introduce_issues and random.random() < self.quality_issue_rate:
                    issue_type = random.choice(['missing', 'format', 'temporal'])
                    
                    if issue_type == 'missing':
                        encounter['patient_id'] = None  # Missing reference
                    
                    elif issue_type == 'format':
                        encounter['encounter_id'] = f"WRONG{j}"  # Wrong format
                    
                    elif issue_type == 'temporal':
                        # Discharge before admission
                        encounter['discharge_date'] = (admission_date - timedelta(days=1)).strftime('%Y-%m-%d')
                
                encounters.append(encounter)
                encounter_counter += 1
        
        return pd.DataFrame(encounters)
    
    def generate_vital_signs(self, encounters_df: pd.DataFrame,
                           vitals_per_encounter: Tuple[int, int] = (1, 3),
                           introduce_issues: bool = True) -> pd.DataFrame:
        """Generate synthetic vital signs data"""
        vitals = []
        vital_counter = 1
        
        for _, encounter in encounters_df.iterrows():
            num_vitals = random.randint(*vitals_per_encounter)
            
            for k in range(num_vitals):
                # Generate realistic vital signs
                vital = {
                    'vital_id': f"VIT{vital_counter:08d}",
                    'encounter_id': encounter['encounter_id'],
                    'patient_id': encounter['patient_id'],
                    'measurement_date': encounter['admission_date'],
                    'temperature_c': round(random.uniform(36.0, 38.5), 1),
                    'heart_rate': random.randint(60, 100),
                    'systolic_bp': random.randint(90, 140),
                    'diastolic_bp': random.randint(60, 90),
                    'respiratory_rate': random.randint(12, 20),
                    'oxygen_saturation': random.randint(95, 100),
                    'weight_kg': round(random.uniform(40, 120), 1),
                    'height_cm': round(random.uniform(150, 200), 1)
                }
                
                # Introduce data quality issues
                if introduce_issues and random.random() < self.quality_issue_rate:
                    issue_type = random.choice(['outlier', 'impossible', 'missing'])
                    
                    if issue_type == 'outlier':
                        # Statistical outliers
                        vital['heart_rate'] = random.randint(300, 400)
                    
                    elif issue_type == 'impossible':
                        # Impossible values
                        vital['temperature_c'] = random.uniform(50, 60)
                        vital['systolic_bp'] = random.randint(300, 400)
                    
                    elif issue_type == 'missing':
                        vital['heart_rate'] = None
                        vital['systolic_bp'] = None
                
                vitals.append(vital)
                vital_counter += 1
        
        return pd.DataFrame(vitals)
    
    def generate_lab_results(self, encounters_df: pd.DataFrame,
                           labs_per_encounter: Tuple[int, int] = (0, 5),
                           introduce_issues: bool = True) -> pd.DataFrame:
        """Generate synthetic lab results"""
        labs = []
        lab_counter = 1
        
        lab_tests = {
            'glucose': (70, 100, 'mg/dL'),
            'hemoglobin': (12, 16, 'g/dL'),
            'white_blood_cells': (4.5, 11.0, 'K/uL'),
            'creatinine': (0.6, 1.2, 'mg/dL'),
            'sodium': (136, 145, 'mmol/L'),
            'potassium': (3.5, 5.0, 'mmol/L'),
            'cholesterol_total': (100, 200, 'mg/dL')
        }
        
        for _, encounter in encounters_df.iterrows():
            num_labs = random.randint(*labs_per_encounter)
            
            for k in range(num_labs):
                test_name = random.choice(list(lab_tests.keys()))
                min_val, max_val, unit = lab_tests[test_name]
                
                lab = {
                    'lab_id': f"LAB{lab_counter:08d}",
                    'encounter_id': encounter['encounter_id'],
                    'patient_id': encounter['patient_id'],
                    'test_date': encounter['admission_date'],
                    'test_name': test_name,
                    'result_value': round(random.uniform(min_val, max_val), 2),
                    'reference_range': f"{min_val}-{max_val}",
                    'unit': unit,
                    'abnormal_flag': random.choice(['N', 'H', 'L', 'C'])  # Normal, High, Low, Critical
                }
                
                # Introduce data quality issues
                if introduce_issues and random.random() < self.quality_issue_rate:
                    issue_type = random.choice(['extreme', 'negative', 'missing'])
                    
                    if issue_type == 'extreme':
                        lab['result_value'] = random.uniform(max_val * 10, max_val * 20)
                    
                    elif issue_type == 'negative':
                        lab['result_value'] = random.uniform(-10, -1)  # Negative values where impossible
                    
                    elif issue_type == 'missing':
                        lab['result_value'] = None
                        lab['unit'] = None
                
                labs.append(lab)
                lab_counter += 1
        
        return pd.DataFrame(labs)
    
    def generate_conditions(self, encounters_df: pd.DataFrame,
                          conditions_per_encounter: Tuple[int, int] = (1, 3),
                          introduce_issues: bool = True) -> pd.DataFrame:
        """Generate synthetic condition/diagnosis data"""
        conditions = []
        condition_counter = 1
        
        for _, encounter in encounters_df.iterrows():
            num_conditions = random.randint(*conditions_per_encounter)
            
            for k in range(num_conditions):
                condition = {
                    'condition_id': f"CON{condition_counter:08d}",
                    'encounter_id': encounter['encounter_id'],
                    'patient_id': encounter['patient_id'],
                    'icd10_code': random.choice(self.icd10_codes),
                    'condition_name': self.fake.text(max_nb_chars=50),
                    'onset_date': encounter['admission_date'],
                    'severity': random.choice(['mild', 'moderate', 'severe']),
                    'status': random.choice(['active', 'resolved', 'chronic'])
                }
                
                # Introduce plausibility issues
                if introduce_issues and random.random() < self.quality_issue_rate:
                    # Get patient gender from encounters (assuming we have it)
                    issue_type = random.choice(['pregnancy_male', 'pediatric_adult', 'format'])
                    
                    if issue_type == 'pregnancy_male':
                        # Pregnancy-related codes for potential male patients
                        condition['icd10_code'] = 'O80.1'  # Normal delivery
                    
                    elif issue_type == 'format':
                        condition['icd10_code'] = 'INVALID_CODE'
                
                conditions.append(condition)
                condition_counter += 1
        
        return pd.DataFrame(conditions)
    
    def generate_complete_dataset(self, num_patients: int = 100,
                                encounters_per_patient: Tuple[int, int] = (1, 3),
                                introduce_issues: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Generate a complete synthetic healthcare dataset
        
        Returns:
            Dictionary containing DataFrames for patients, encounters, vitals, labs, conditions
        """
        print(f"Generating synthetic healthcare data for {num_patients} patients...")
        
        # Generate patients
        patients_df = self.generate_patients(num_patients, introduce_issues)
        print(f"✓ Generated {len(patients_df)} patients")
        
        # Generate encounters
        encounters_df = self.generate_encounters(patients_df, encounters_per_patient, introduce_issues)
        print(f"✓ Generated {len(encounters_df)} encounters")
        
        # Generate vital signs
        vitals_df = self.generate_vital_signs(encounters_df, (1, 2), introduce_issues)
        print(f"✓ Generated {len(vitals_df)} vital sign records")
        
        # Generate lab results
        labs_df = self.generate_lab_results(encounters_df, (0, 3), introduce_issues)
        print(f"✓ Generated {len(labs_df)} lab results")
        
        # Generate conditions
        conditions_df = self.generate_conditions(encounters_df, (1, 2), introduce_issues)
        print(f"✓ Generated {len(conditions_df)} condition records")
        
        dataset = {
            'patients': patients_df,
            'encounters': encounters_df,
            'vital_signs': vitals_df,
            'lab_results': labs_df,
            'conditions': conditions_df
        }
        
        if introduce_issues:
            print(f"⚠️  Data quality issues introduced at ~{self.quality_issue_rate:.0%} rate for testing")
        
        return dataset
    
    def save_dataset_to_csv(self, dataset: Dict[str, pd.DataFrame], output_dir: str):
        """Save generated dataset to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for table_name, df in dataset.items():
            filepath = os.path.join(output_dir, f"{table_name}.csv")
            df.to_csv(filepath, index=False)
            print(f"Saved {table_name} to {filepath}")
    
    def create_merged_dataset(self, dataset: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create a merged dataset suitable for quality assessment
        
        Merges all tables into a single DataFrame for comprehensive quality assessment
        """
        # Start with encounters as the base
        merged = dataset['encounters'].copy()
        
        # Merge with patients
        if 'patients' in dataset:
            merged = merged.merge(
                dataset['patients'], 
                on='patient_id', 
                how='left',
                suffixes=('_encounter', '_patient')
            )
        
        # Merge with latest vital signs per encounter
        if 'vital_signs' in dataset:
            latest_vitals = dataset['vital_signs'].groupby('encounter_id').last().reset_index()
            merged = merged.merge(
                latest_vitals[['encounter_id', 'temperature_c', 'heart_rate', 'systolic_bp', 
                              'diastolic_bp', 'weight_kg', 'height_cm']], 
                on='encounter_id', 
                how='left'
            )
        
        # Add condition count per encounter
        if 'conditions' in dataset:
            condition_counts = dataset['conditions'].groupby('encounter_id').size().reset_index(name='condition_count')
            merged = merged.merge(condition_counts, on='encounter_id', how='left')
            merged['condition_count'] = merged['condition_count'].fillna(0)
        
        # Add lab result count per encounter  
        if 'lab_results' in dataset:
            lab_counts = dataset['lab_results'].groupby('encounter_id').size().reset_index(name='lab_count')
            merged = merged.merge(lab_counts, on='encounter_id', how='left')
            merged['lab_count'] = merged['lab_count'].fillna(0)
        
        return merged
