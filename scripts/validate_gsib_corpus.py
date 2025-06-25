#!/usr/bin/env python3
"""
G-SIB Financial Corpus Validation Script

This script validates the synthetic G-SIB banking corpus for:
- Format consistency
- Numerical accuracy
- Regulatory compliance
- Content quality
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Container for validation results."""
    passed: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]

class GSIBCorpusValidator:
    """Validator for G-SIB financial corpus."""
    
    def __init__(self):
        self.regulatory_terms = {
            'basel_iii': ['CET1', 'Tier 1', 'Basel III', 'Risk-Weighted Assets', 'RWA'],
            'liquidity': ['LCR', 'NSFR', 'Liquidity Coverage', 'Net Stable Funding'],
            'capital': ['Capital Ratio', 'Capital Buffer', 'TLAC', 'MREL'],
            'risk_management': ['Stress Test', 'VaR', 'Expected Loss', 'Operational Risk'],
            'gsib_specific': ['G-SIB', 'Systemic', 'Resolution', 'FRTB', 'SREP']
        }
        
        self.numerical_patterns = {
            'percentage': r'\d+\.?\d*%',
            'ratio': r'\d+\.?\d*x?',
            'currency': r'\$\d+\.?\d*[BMK]?',
            'basis_points': r'\d+\s*(?:basis points|bp)',
        }
        
        self.regulatory_thresholds = {
            'cet1_minimum': 4.5,
            'lcr_minimum': 100.0,
            'nsfr_minimum': 100.0,
            'leverage_ratio_minimum': 3.0,
            'gsib_buffer_max': 3.5
        }

    def validate_corpus(self, corpus_path: str) -> ValidationResult:
        """Validate the entire corpus."""
        logger.info(f"Starting validation of corpus: {corpus_path}")
        
        errors = []
        warnings = []
        metrics = {
            'total_samples': 0,
            'format_valid': 0,
            'numerical_accurate': 0,
            'regulatory_compliant': 0,
            'coverage_score': 0.0
        }
        
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                samples = [json.loads(line.strip()) for line in f if line.strip()]
            
            metrics['total_samples'] = len(samples)
            logger.info(f"Loaded {len(samples)} samples for validation")
            
            # Validate each sample
            for i, sample in enumerate(samples):
                sample_result = self._validate_sample(sample, i)
                
                if sample_result.passed:
                    metrics['format_valid'] += 1
                else:
                    errors.extend([f"Sample {i}: {error}" for error in sample_result.errors])
                
                warnings.extend([f"Sample {i}: {warning}" for warning in sample_result.warnings])
            
            # Calculate coverage metrics
            coverage_result = self._calculate_coverage(samples)
            metrics.update(coverage_result)
            
            # Validate numerical accuracy across samples
            numerical_result = self._validate_numerical_accuracy(samples)
            metrics['numerical_accurate'] = numerical_result['accurate_count']
            errors.extend(numerical_result['errors'])
            
            # Check regulatory compliance
            compliance_result = self._validate_regulatory_compliance(samples)
            metrics['regulatory_compliant'] = compliance_result['compliant_count']
            warnings.extend(compliance_result['warnings'])
            
            # Overall validation result
            passed = (
                len(errors) == 0 and
                metrics['format_valid'] == metrics['total_samples'] and
                metrics['coverage_score'] >= 0.8
            )
            
            logger.info(f"Validation completed. Passed: {passed}")
            return ValidationResult(passed, errors, warnings, metrics)
            
        except Exception as e:
            logger.error(f"Validation failed with exception: {e}")
            return ValidationResult(False, [f"Validation exception: {str(e)}"], [], metrics)

    def _validate_sample(self, sample: Dict, index: int) -> ValidationResult:
        """Validate a single sample."""
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ['instruction', 'input', 'output']
        for field in required_fields:
            if field not in sample:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(sample[field], str) or not sample[field].strip():
                errors.append(f"Empty or invalid {field}")
        
        if errors:
            return ValidationResult(False, errors, warnings, {})
        
        # Validate content quality
        instruction = sample['instruction']
        input_text = sample['input']
        output_text = sample['output']
        
        # Check instruction clarity
        if len(instruction) < 20:
            warnings.append("Instruction may be too brief")
        
        # Check input realism
        if not self._contains_financial_data(input_text):
            warnings.append("Input may lack realistic financial data")
        
        # Check output completeness
        if len(output_text) < 50:
            warnings.append("Output may be too brief")
        
        # Check for regulatory terminology
        if not self._contains_regulatory_terms(instruction + " " + output_text):
            warnings.append("May lack sufficient regulatory terminology")
        
        return ValidationResult(True, errors, warnings, {})

    def _contains_financial_data(self, text: str) -> bool:
        """Check if text contains realistic financial data."""
        patterns = [
            r'\$\d+\.?\d*[BMK]',  # Currency amounts
            r'\d+\.?\d*%',        # Percentages
            r'\d+\.?\d*x',        # Ratios
            r'\d+\.?\d*\s*(?:basis points|bp)',  # Basis points
        ]
        
        return any(re.search(pattern, text) for pattern in patterns)

    def _contains_regulatory_terms(self, text: str) -> bool:
        """Check if text contains regulatory terminology."""
        text_upper = text.upper()
        
        for category, terms in self.regulatory_terms.items():
            if any(term.upper() in text_upper for term in terms):
                return True
        
        return False

    def _calculate_coverage(self, samples: List[Dict]) -> Dict[str, float]:
        """Calculate coverage metrics across regulatory domains."""
        coverage = {}
        
        for category, terms in self.regulatory_terms.items():
            covered_samples = 0
            for sample in samples:
                text = (sample['instruction'] + " " + sample['output']).upper()
                if any(term.upper() in text for term in terms):
                    covered_samples += 1
            
            coverage[f'{category}_coverage'] = covered_samples / len(samples) if samples else 0
        
        # Overall coverage score
        coverage['coverage_score'] = sum(coverage.values()) / len(coverage) if coverage else 0
        
        return coverage

    def _validate_numerical_accuracy(self, samples: List[Dict]) -> Dict[str, Any]:
        """Validate numerical calculations in outputs."""
        errors = []
        accurate_count = 0
        
        for i, sample in enumerate(samples):
            output = sample['output']
            
            # Extract calculations (simple pattern matching)
            calc_patterns = [
                r'(\d+\.?\d*)\s*/\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)',  # Division
                r'(\d+\.?\d*)\s*\*\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)',  # Multiplication
                r'(\d+\.?\d*)\s*\+\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)',  # Addition
                r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)',  # Subtraction
            ]
            
            calculations_found = False
            calculations_accurate = True
            
            for pattern in calc_patterns:
                matches = re.finditer(pattern, output)
                for match in matches:
                    calculations_found = True
                    try:
                        if '/' in match.group(0):
                            a, b, result = float(match.group(1)), float(match.group(2)), float(match.group(3))
                            expected = a / b if b != 0 else float('inf')
                        elif '*' in match.group(0):
                            a, b, result = float(match.group(1)), float(match.group(2)), float(match.group(3))
                            expected = a * b
                        elif '+' in match.group(0):
                            a, b, result = float(match.group(1)), float(match.group(2)), float(match.group(3))
                            expected = a + b
                        elif '-' in match.group(0):
                            a, b, result = float(match.group(1)), float(match.group(2)), float(match.group(3))
                            expected = a - b
                        
                        # Check accuracy with tolerance
                        tolerance = 0.01  # 1% tolerance
                        if abs(result - expected) / max(abs(expected), 1e-10) > tolerance:
                            calculations_accurate = False
                            errors.append(f"Sample {i}: Calculation error - {match.group(0)}")
                    
                    except (ValueError, ZeroDivisionError) as e:
                        calculations_accurate = False
                        errors.append(f"Sample {i}: Calculation parsing error - {str(e)}")
            
            if calculations_found and calculations_accurate:
                accurate_count += 1
            elif not calculations_found:
                # If no calculations found, assume it's accurate (qualitative content)
                accurate_count += 1
        
        return {
            'accurate_count': accurate_count,
            'errors': errors
        }

    def _validate_regulatory_compliance(self, samples: List[Dict]) -> Dict[str, Any]:
        """Validate regulatory compliance of content."""
        warnings = []
        compliant_count = 0
        
        for i, sample in enumerate(samples):
            output = sample['output']
            is_compliant = True
            
            # Check for regulatory threshold mentions
            for threshold_name, threshold_value in self.regulatory_thresholds.items():
                if threshold_name.replace('_', ' ').upper() in output.upper():
                    # Extract mentioned values and check against thresholds
                    percentages = re.findall(r'(\d+\.?\d*)%', output)
                    for pct_str in percentages:
                        try:
                            pct_value = float(pct_str)
                            if 'minimum' in threshold_name and pct_value < threshold_value:
                                warnings.append(f"Sample {i}: Value {pct_value}% below regulatory minimum {threshold_value}%")
                            elif 'maximum' in threshold_name and pct_value > threshold_value:
                                warnings.append(f"Sample {i}: Value {pct_value}% above regulatory maximum {threshold_value}%")
                        except ValueError:
                            continue
            
            # Check for Basel III compliance language
            basel_terms = ['Basel III', 'minimum requirement', 'regulatory', 'compliance']
            if any(term.lower() in output.lower() for term in basel_terms):
                compliant_count += 1
            else:
                is_compliant = False
        
        return {
            'compliant_count': compliant_count,
            'warnings': warnings
        }

    def generate_report(self, result: ValidationResult, output_path: str = None) -> str:
        """Generate a validation report."""
        report = []
        report.append("# G-SIB Corpus Validation Report")
        report.append(f"**Validation Status**: {'PASSED' if result.passed else 'FAILED'}")
        report.append("")
        
        # Metrics Summary
        report.append("## Metrics Summary")
        for key, value in result.metrics.items():
            if isinstance(value, float):
                report.append(f"- **{key.replace('_', ' ').title()}**: {value:.2%}")
            else:
                report.append(f"- **{key.replace('_', ' ').title()}**: {value}")
        report.append("")
        
        # Errors
        if result.errors:
            report.append("## Errors")
            for error in result.errors:
                report.append(f"- ERROR: {error}")
            report.append("")
        
        # Warnings
        if result.warnings:
            report.append("## Warnings")
            for warning in result.warnings[:10]:  # Limit to first 10
                report.append(f"- WARNING: {warning}")
            if len(result.warnings) > 10:
                report.append(f"- ... and {len(result.warnings) - 10} more warnings")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        if result.metrics.get('coverage_score', 0) < 0.8:
            report.append("- Increase coverage of regulatory domains")
        if result.metrics.get('numerical_accurate', 0) < result.metrics.get('total_samples', 1):
            report.append("- Review numerical calculations for accuracy")
        if result.errors:
            report.append("- Address format and content errors before training")
        if not result.errors and not result.warnings:
            report.append("- Corpus is ready for training!")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")
        
        return report_text

def main():
    """Main validation function."""
    if len(sys.argv) < 2:
        print("Usage: python validate_gsib_corpus.py <corpus_path> [output_report_path]")
        sys.exit(1)
    
    corpus_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(corpus_path).exists():
        logger.error(f"Corpus file not found: {corpus_path}")
        sys.exit(1)
    
    validator = GSIBCorpusValidator()
    result = validator.validate_corpus(corpus_path)
    
    # Generate and display report
    report = validator.generate_report(result, output_path)
    print("\n" + report)
    
    # Exit with appropriate code
    sys.exit(0 if result.passed else 1)

if __name__ == "__main__":
    main()