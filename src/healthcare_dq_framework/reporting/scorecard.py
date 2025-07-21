"""
Data Quality Scorecard and Reporting System

This module provides comprehensive reporting capabilities for healthcare
data quality assessments including scorecards, detailed reports, and
visualizations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from ..core.dimensions import QualityDimensionType, DimensionResult, QualityIssue


class DataQualityScorecard:
    """
    Generate comprehensive data quality scorecards and reports
    """
    
    def __init__(self):
        self.assessment_results = None
        self.overall_score = 0.0
        self.dimension_scores = {}
        self.issue_summary = {}
        
    def generate_scorecard(self, assessment_results: Dict[QualityDimensionType, DimensionResult],
                          overall_score: float) -> Dict[str, Any]:
        """
        Generate a comprehensive data quality scorecard
        
        Args:
            assessment_results: Results from framework assessment
            overall_score: Overall quality score
            
        Returns:
            Dictionary containing scorecard data
        """
        self.assessment_results = assessment_results
        self.overall_score = overall_score
        
        scorecard = {
            'assessment_metadata': {
                'timestamp': datetime.now().isoformat(),
                'overall_score': overall_score,
                'assessment_status': self._get_assessment_status(overall_score)
            },
            'dimension_summary': self._create_dimension_summary(),
            'issue_analysis': self._create_issue_analysis(),
            'recommendations': self._generate_recommendations(),
            'detailed_metrics': self._create_detailed_metrics()
        }
        
        return scorecard
    
    def _get_assessment_status(self, score: float) -> str:
        """Determine assessment status based on score"""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Good"
        elif score >= 70:
            return "Acceptable"
        elif score >= 60:
            return "Poor"
        else:
            return "Critical"
    
    def _create_dimension_summary(self) -> Dict[str, Any]:
        """Create summary of dimension scores and status"""
        summary = {}
        
        for dim_type, result in self.assessment_results.items():
            dimension_name = dim_type.value.title()
            
            summary[dimension_name] = {
                'score': result.score,
                'status': self._get_assessment_status(result.score),
                'total_records': result.total_records,
                'failed_records': result.failed_records,
                'failure_rate': (result.failed_records / result.total_records * 100) if result.total_records > 0 else 0,
                'rules_evaluated': result.rules_evaluated,
                'rules_passed': result.rules_passed,
                'issue_count': len(result.issues)
            }
            
            # Store for later use
            self.dimension_scores[dimension_name] = result.score
        
        return summary
    
    def _create_issue_analysis(self) -> Dict[str, Any]:
        """Analyze issues across all dimensions"""
        all_issues = []
        for result in self.assessment_results.values():
            all_issues.extend(result.issues)
        
        # Count issues by severity
        severity_counts = {'critical': 0, 'major': 0, 'minor': 0, 'warning': 0}
        rule_counts = {}
        field_counts = {}
        
        for issue in all_issues:
            # Count by severity
            if issue.severity in severity_counts:
                severity_counts[issue.severity] += 1
            
            # Count by rule
            rule_counts[issue.rule_name] = rule_counts.get(issue.rule_name, 0) + 1
            
            # Count by field
            field_counts[issue.field_name] = field_counts.get(issue.field_name, 0) + 1
        
        # Find top problematic rules and fields
        top_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_fields = sorted(field_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        analysis = {
            'total_issues': len(all_issues),
            'severity_breakdown': severity_counts,
            'top_failing_rules': [{'rule': rule, 'count': count} for rule, count in top_rules],
            'top_problematic_fields': [{'field': field, 'count': count} for field, count in top_fields],
            'issue_distribution': self._calculate_issue_distribution()
        }
        
        self.issue_summary = analysis
        return analysis
    
    def _calculate_issue_distribution(self) -> Dict[str, float]:
        """Calculate distribution of issues across dimensions"""
        dimension_issue_counts = {}
        
        for dim_type, result in self.assessment_results.items():
            dimension_issue_counts[dim_type.value] = len(result.issues)
        
        total_issues = sum(dimension_issue_counts.values())
        
        if total_issues == 0:
            return {dim: 0.0 for dim in dimension_issue_counts.keys()}
        
        return {dim: (count / total_issues * 100) for dim, count in dimension_issue_counts.items()}
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on assessment results"""
        recommendations = []
        
        # Score-based recommendations
        if self.overall_score < 60:
            recommendations.append({
                'priority': 'High',
                'category': 'Overall Quality',
                'recommendation': 'Immediate data quality intervention required. Consider comprehensive data cleansing and validation.',
                'impact': 'Critical'
            })
        
        # Dimension-specific recommendations
        for dim_name, score in self.dimension_scores.items():
            if score < 70:
                recommendations.append({
                    'priority': 'High' if score < 50 else 'Medium',
                    'category': f'{dim_name} Issues',
                    'recommendation': self._get_dimension_recommendation(dim_name, score),
                    'impact': 'High' if score < 50 else 'Medium'
                })
        
        # Issue-specific recommendations
        if self.issue_summary:
            critical_issues = self.issue_summary['severity_breakdown'].get('critical', 0)
            major_issues = self.issue_summary['severity_breakdown'].get('major', 0)
            
            if critical_issues > 0:
                recommendations.append({
                    'priority': 'Critical',
                    'category': 'Critical Issues',
                    'recommendation': f'Address {critical_issues} critical data quality issues immediately. These may indicate systematic data collection problems.',
                    'impact': 'Critical'
                })
            
            if major_issues > 10:
                recommendations.append({
                    'priority': 'High',
                    'category': 'Major Issues',
                    'recommendation': f'Investigate and resolve {major_issues} major data quality issues. Consider implementing additional validation rules.',
                    'impact': 'High'
                })
        
        return recommendations
    
    def _get_dimension_recommendation(self, dimension: str, score: float) -> str:
        """Get specific recommendation for a dimension"""
        recommendations = {
            'Completeness': {
                'low': 'Implement required field validation and improve data collection processes.',
                'medium': 'Review data entry procedures and add completeness checks.'
            },
            'Conformance': {
                'low': 'Establish data format standards and implement strict validation rules.',
                'medium': 'Review and update data validation rules, add format checking.'
            },
            'Plausibility': {
                'low': 'Implement clinical decision support and cross-field validation rules.',
                'medium': 'Add plausibility checks and clinical logic validation.'
            }
        }
        
        level = 'low' if score < 50 else 'medium'
        return recommendations.get(dimension, {}).get(level, 'Review and improve data quality processes.')
    
    def _create_detailed_metrics(self) -> Dict[str, Any]:
        """Create detailed metrics for technical analysis"""
        metrics = {}
        
        for dim_type, result in self.assessment_results.items():
            dimension_name = dim_type.value.title()
            
            # Calculate metrics per rule
            rule_metrics = {}
            for issue in result.issues:
                rule_name = issue.rule_name
                if rule_name not in rule_metrics:
                    rule_metrics[rule_name] = {
                        'count': 0,
                        'severity_distribution': {'critical': 0, 'major': 0, 'minor': 0, 'warning': 0},
                        'average_confidence': 0.0,
                        'affected_fields': set()
                    }
                
                rule_metrics[rule_name]['count'] += 1
                rule_metrics[rule_name]['severity_distribution'][issue.severity] += 1
                rule_metrics[rule_name]['average_confidence'] += issue.confidence
                rule_metrics[rule_name]['affected_fields'].add(issue.field_name)
            
            # Calculate averages and convert sets to lists
            for rule_name, metrics_data in rule_metrics.items():
                if metrics_data['count'] > 0:
                    metrics_data['average_confidence'] /= metrics_data['count']
                metrics_data['affected_fields'] = list(metrics_data['affected_fields'])
            
            metrics[dimension_name] = {
                'summary': {
                    'score': result.score,
                    'total_records': result.total_records,
                    'failed_records': result.failed_records,
                    'rules_evaluated': result.rules_evaluated,
                    'rules_passed': result.rules_passed
                },
                'rule_metrics': rule_metrics
            }
        
        return metrics
    
    def export_scorecard(self, scorecard: Dict[str, Any], filepath: str, format: str = 'json'):
        """Export scorecard to file"""
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(scorecard, f, indent=2, default=str)
        elif format.lower() == 'csv':
            # Create flattened CSV version
            self._export_scorecard_csv(scorecard, filepath)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        print(f"Scorecard exported to: {filepath}")
    
    def _export_scorecard_csv(self, scorecard: Dict[str, Any], filepath: str):
        """Export scorecard data to CSV format"""
        # Create summary CSV
        summary_data = []
        
        # Overall metrics
        summary_data.append({
            'Metric': 'Overall Score',
            'Value': scorecard['assessment_metadata']['overall_score'],
            'Status': scorecard['assessment_metadata']['assessment_status']
        })
        
        # Dimension scores
        for dimension, metrics in scorecard['dimension_summary'].items():
            summary_data.append({
                'Metric': f'{dimension} Score',
                'Value': metrics['score'],
                'Status': metrics['status']
            })
            
            summary_data.append({
                'Metric': f'{dimension} Failed Records',
                'Value': metrics['failed_records'],
                'Status': f"{metrics['failure_rate']:.1f}%"
            })
        
        # Issue counts
        for severity, count in scorecard['issue_analysis']['severity_breakdown'].items():
            summary_data.append({
                'Metric': f'{severity.title()} Issues',
                'Value': count,
                'Status': 'Count'
            })
        
        # Save to CSV
        df = pd.DataFrame(summary_data)
        df.to_csv(filepath, index=False)
    
    def generate_html_report(self, scorecard: Dict[str, Any], output_path: str = "data_quality_report.html"):
        """Generate an HTML report with visualizations"""
        html_content = self._create_html_report(scorecard)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {output_path}")
        return output_path
    
    def _create_html_report(self, scorecard: Dict[str, Any]) -> str:
        """Create HTML content for the report"""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Healthcare Data Quality Assessment Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .overall-score {{ font-size: 48px; font-weight: bold; color: {self._get_score_color(scorecard['assessment_metadata']['overall_score'])}; }}
        .status {{ font-size: 24px; margin-top: 10px; }}
        .dimensions {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0; }}
        .dimension-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 20px; background-color: #f9f9f9; }}
        .dimension-score {{ font-size: 36px; font-weight: bold; }}
        .issue-summary {{ margin: 30px 0; }}
        .severity-critical {{ color: #dc3545; }}
        .severity-major {{ color: #fd7e14; }}
        .severity-minor {{ color: #ffc107; }}
        .severity-warning {{ color: #6c757d; }}
        .recommendations {{ background-color: #e7f3ff; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .recommendation {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007bff; background-color: white; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Healthcare Data Quality Assessment Report</h1>
            <div class="overall-score">{scorecard['assessment_metadata']['overall_score']:.1f}</div>
            <div class="status">{scorecard['assessment_metadata']['assessment_status']}</div>
            <p>Generated on: {scorecard['assessment_metadata']['timestamp']}</p>
        </div>
        
        <h2>Dimension Scores</h2>
        <div class="dimensions">
        """
        
        # Add dimension cards
        for dimension, metrics in scorecard['dimension_summary'].items():
            html += f"""
            <div class="dimension-card">
                <h3>{dimension}</h3>
                <div class="dimension-score" style="color: {self._get_score_color(metrics['score'])}">{metrics['score']:.1f}</div>
                <p>Status: {metrics['status']}</p>
                <p>Failed Records: {metrics['failed_records']} ({metrics['failure_rate']:.1f}%)</p>
                <p>Rules Passed: {metrics['rules_passed']}/{metrics['rules_evaluated']}</p>
                <p>Issues Found: {metrics['issue_count']}</p>
            </div>
            """
        
        html += """
        </div>
        
        <h2>Issue Analysis</h2>
        <div class="issue-summary">
        """
        
        # Add issue summary
        issue_analysis = scorecard['issue_analysis']
        html += f"""
        <p><strong>Total Issues Found:</strong> {issue_analysis['total_issues']}</p>
        <h3>Issues by Severity</h3>
        <ul>
            <li class="severity-critical">Critical: {issue_analysis['severity_breakdown']['critical']}</li>
            <li class="severity-major">Major: {issue_analysis['severity_breakdown']['major']}</li>
            <li class="severity-minor">Minor: {issue_analysis['severity_breakdown']['minor']}</li>
            <li class="severity-warning">Warning: {issue_analysis['severity_breakdown']['warning']}</li>
        </ul>
        """
        
        # Top failing rules table
        if issue_analysis['top_failing_rules']:
            html += """
            <h3>Top Failing Rules</h3>
            <table>
                <tr><th>Rule</th><th>Issue Count</th></tr>
            """
            for rule_info in issue_analysis['top_failing_rules'][:5]:
                html += f"<tr><td>{rule_info['rule']}</td><td>{rule_info['count']}</td></tr>"
            html += "</table>"
        
        html += "</div>"
        
        # Recommendations section
        html += """
        <h2>Recommendations</h2>
        <div class="recommendations">
        """
        
        for rec in scorecard['recommendations']:
            priority_color = {'Critical': '#dc3545', 'High': '#fd7e14', 'Medium': '#ffc107', 'Low': '#28a745'}.get(rec['priority'], '#6c757d')
            html += f"""
            <div class="recommendation">
                <strong style="color: {priority_color};">{rec['priority']} Priority - {rec['category']}</strong><br>
                {rec['recommendation']}
                <br><small>Impact: {rec['impact']}</small>
            </div>
            """
        
        html += """
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
    def _get_score_color(self, score: float) -> str:
        """Get color for score based on value"""
        if score >= 90:
            return "#28a745"  # Green
        elif score >= 80:
            return "#17a2b8"  # Blue
        elif score >= 70:
            return "#ffc107"  # Yellow
        elif score >= 60:
            return "#fd7e14"  # Orange
        else:
            return "#dc3545"  # Red
    
    def create_issue_details_report(self, assessment_results: Dict[QualityDimensionType, DimensionResult]) -> pd.DataFrame:
        """Create detailed issue report as DataFrame"""
        all_issues_data = []
        
        for dim_type, result in assessment_results.items():
            for issue in result.issues:
                issue_data = {
                    'Dimension': dim_type.value.title(),
                    'Severity': issue.severity,
                    'Rule': issue.rule_name,
                    'Description': issue.description,
                    'Field': issue.field_name,
                    'Record_ID': issue.record_id,
                    'Actual_Value': str(issue.actual_value) if issue.actual_value is not None else '',
                    'Expected_Value': str(issue.expected_value) if issue.expected_value is not None else '',
                    'Confidence': issue.confidence,
                    'Suggestion': issue.suggestion or ''
                }
                all_issues_data.append(issue_data)
        
        return pd.DataFrame(all_issues_data)


class DataQualityTrendAnalyzer:
    """
    Analyze trends in data quality over time across multiple assessments
    """
    
    def __init__(self):
        self.historical_assessments = []
    
    def add_assessment(self, scorecard: Dict[str, Any], assessment_name: str = None):
        """Add an assessment to trend analysis"""
        if assessment_name is None:
            assessment_name = f"Assessment_{len(self.historical_assessments) + 1}"
        
        assessment_record = {
            'name': assessment_name,
            'timestamp': scorecard['assessment_metadata']['timestamp'],
            'overall_score': scorecard['assessment_metadata']['overall_score'],
            'dimension_scores': {dim: metrics['score'] for dim, metrics in scorecard['dimension_summary'].items()},
            'total_issues': scorecard['issue_analysis']['total_issues'],
            'severity_breakdown': scorecard['issue_analysis']['severity_breakdown']
        }
        
        self.historical_assessments.append(assessment_record)
    
    def generate_trend_report(self) -> Dict[str, Any]:
        """Generate trend analysis report"""
        if len(self.historical_assessments) < 2:
            return {"message": "Need at least 2 assessments for trend analysis"}
        
        trend_data = {
            'assessment_count': len(self.historical_assessments),
            'time_span': {
                'start': self.historical_assessments[0]['timestamp'],
                'end': self.historical_assessments[-1]['timestamp']
            },
            'overall_score_trend': self._calculate_score_trend(),
            'dimension_trends': self._calculate_dimension_trends(),
            'issue_trends': self._calculate_issue_trends(),
            'quality_improvement': self._assess_quality_improvement()
        }
        
        return trend_data
    
    def _calculate_score_trend(self) -> Dict[str, Any]:
        """Calculate overall score trend"""
        scores = [assessment['overall_score'] for assessment in self.historical_assessments]
        
        return {
            'current_score': scores[-1],
            'previous_score': scores[-2],
            'change': scores[-1] - scores[-2],
            'trend': 'improving' if scores[-1] > scores[-2] else 'declining' if scores[-1] < scores[-2] else 'stable',
            'average_score': np.mean(scores),
            'best_score': max(scores),
            'worst_score': min(scores)
        }
    
    def _calculate_dimension_trends(self) -> Dict[str, Dict[str, Any]]:
        """Calculate trends for each dimension"""
        dimension_trends = {}
        
        # Get all unique dimensions
        all_dimensions = set()
        for assessment in self.historical_assessments:
            all_dimensions.update(assessment['dimension_scores'].keys())
        
        for dimension in all_dimensions:
            scores = []
            for assessment in self.historical_assessments:
                if dimension in assessment['dimension_scores']:
                    scores.append(assessment['dimension_scores'][dimension])
            
            if len(scores) >= 2:
                dimension_trends[dimension] = {
                    'current_score': scores[-1],
                    'previous_score': scores[-2],
                    'change': scores[-1] - scores[-2],
                    'trend': 'improving' if scores[-1] > scores[-2] else 'declining' if scores[-1] < scores[-2] else 'stable',
                    'average_score': np.mean(scores)
                }
        
        return dimension_trends
    
    def _calculate_issue_trends(self) -> Dict[str, Any]:
        """Calculate issue count trends"""
        total_issues = [assessment['total_issues'] for assessment in self.historical_assessments]
        
        return {
            'current_issues': total_issues[-1],
            'previous_issues': total_issues[-2],
            'change': total_issues[-1] - total_issues[-2],
            'trend': 'improving' if total_issues[-1] < total_issues[-2] else 'worsening' if total_issues[-1] > total_issues[-2] else 'stable',
            'average_issues': np.mean(total_issues)
        }
    
    def _assess_quality_improvement(self) -> str:
        """Assess overall quality improvement"""
        score_trend = self._calculate_score_trend()
        issue_trend = self._calculate_issue_trends()
        
        if score_trend['trend'] == 'improving' and issue_trend['trend'] == 'improving':
            return "Significant improvement"
        elif score_trend['trend'] == 'improving' or issue_trend['trend'] == 'improving':
            return "Moderate improvement"
        elif score_trend['trend'] == 'stable' and issue_trend['trend'] == 'stable':
            return "Stable"
        elif score_trend['trend'] == 'declining' or issue_trend['trend'] == 'worsening':
            return "Declining"
        else:
            return "Mixed results"
