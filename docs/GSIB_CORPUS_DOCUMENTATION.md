# G-SIB Enhanced Financial Corpus Documentation

## Overview

The G-SIB Enhanced Financial Corpus (`synthetic_finance_gsib_v3.jsonl`) is a specialized training dataset designed for fine-tuning Large Language Models on advanced banking and regulatory concepts specific to Global Systemically Important Banks (G-SIBs).

## Dataset Specifications

- **Format**: JSONL (JSON Lines)
- **Size**: 30 advanced samples
- **Structure**: instruction-input-output format optimized for supervised fine-tuning
- **Domain**: G-SIB banking, Basel III regulations, risk management

## Coverage Areas

### 1. Basel III Capital Requirements (8 samples)
- **Common Equity Tier 1 (CET1) Ratio**: Core capital adequacy metric
- **Tier 1 Leverage Ratio**: Non-risk-based capital measure
- **Total Capital Ratio**: Comprehensive capital assessment
- **G-SIB Capital Buffers**: Additional requirements for systemically important banks

### 2. Liquidity Risk Management (4 samples)
- **Liquidity Coverage Ratio (LCR)**: Short-term liquidity resilience
- **Net Stable Funding Ratio (NSFR)**: Long-term funding stability
- **Stressed Liquidity Scenarios**: Crisis-period liquidity assessment
- **Intraday Liquidity Management**: Real-time liquidity monitoring

### 3. Risk-Weighted Assets & Credit Risk (4 samples)
- **Standardized Approach RWA**: Credit risk capital calculation
- **Credit Risk Mitigation**: Collateral and guarantee impact
- **Expected Credit Loss (IFRS 9)**: Forward-looking provisioning
- **Large Exposure Limits**: Concentration risk management

### 4. Market & Operational Risk (3 samples)
- **Fundamental Review of Trading Book (FRTB)**: Advanced market risk
- **Operational Risk (SMA)**: Standardized Measurement Approach
- **Credit Valuation Adjustment (CVA)**: Counterparty credit risk

### 5. Resolution & Recovery Planning (4 samples)
- **Total Loss Absorbing Capacity (TLAC)**: Resolution funding
- **Minimum Requirement for own funds and Eligible Liabilities (MREL)**
- **Recovery Plan Triggers**: Early intervention mechanisms
- **Bail-in Capacity Assessment**: Resolution tool effectiveness

### 6. Supervisory Assessment (4 samples)
- **Supervisory Review and Evaluation Process (SREP)**
- **Internal Capital Adequacy Assessment Process (ICAAP)**
- **Stress Testing Results**: Regulatory stress scenarios
- **Model Risk Management**: Validation and governance

### 7. Emerging Risk Areas (3 samples)
- **Climate Risk Capital**: Environmental risk assessment
- **Cyber Risk Operational Capital**: Technology risk quantification
- **Systemic Risk Buffers**: Macroprudential requirements

## Key Features

### Regulatory Accuracy
- All calculations follow Basel III/IV frameworks
- Incorporates latest regulatory guidance and standards
- Reflects real-world G-SIB operational complexity

### Practical Application
- Uses realistic bank balance sheet figures
- Includes regulatory thresholds and buffer requirements
- Demonstrates compliance assessment methodologies

### Educational Value
- Step-by-step calculation explanations
- Regulatory context and interpretation
- Risk management implications

## Sample Structure

```json
{
  "instruction": "Calculate the Common Equity Tier 1 (CET1) ratio from the regulatory capital data.",
  "input": "Common Equity Tier 1 Capital: $45.2B, Risk-Weighted Assets: $320.5B",
  "output": "CET1 Ratio = CET1 Capital / Risk-Weighted Assets = $45.2B / $320.5B = 14.1%. This exceeds the Basel III minimum requirement of 4.5% and the G-SIB buffer requirements."
}
```

## Training Effectiveness

### Model Capabilities Enhanced
1. **Regulatory Calculation Proficiency**: Accurate computation of complex ratios
2. **Compliance Assessment**: Understanding of regulatory thresholds
3. **Risk Interpretation**: Contextual analysis of risk metrics
4. **Regulatory Reasoning**: Application of banking regulations

### Complexity Progression
- **Basic**: Simple ratio calculations
- **Intermediate**: Multi-component assessments
- **Advanced**: Stress testing and scenario analysis
- **Expert**: Integrated risk and capital management

## Integration with Training Pipeline

### Configuration Updates
```yaml
training_datasets:
  gsib_enhanced:
    path: "data/training/synthetic_finance_gsib_v3.jsonl"
    type: "instruction_following"
    domain: "gsib_banking"
    complexity: "high"
    samples: 30
```

### Fine-tuning Parameters
- **Learning Rate**: 2e-5 (conservative for specialized domain)
- **Batch Size**: 4 (due to complexity)
- **Epochs**: 3-5 (prevent overfitting on small dataset)
- **LoRA Rank**: 16-32 (balance efficiency and capacity)

## Quality Assurance

### Validation Criteria
- ✅ Regulatory accuracy verified against Basel III documentation
- ✅ Mathematical calculations independently verified
- ✅ Realistic scenarios based on actual G-SIB data ranges
- ✅ Consistent formatting and structure

### Expert Review
- Banking regulation specialists
- Risk management practitioners
- Model validation teams
- Regulatory compliance officers

## Future Enhancements

### Planned Expansions
1. **Scale**: Increase to 100-200 samples per category
2. **Jurisdictional Variations**: EU, US, UK regulatory differences
3. **Dynamic Scenarios**: Time-series and multi-period analysis
4. **Cross-Risk Integration**: Holistic risk management scenarios

### Advanced Concepts
- **CECL Implementation**: US credit loss standards
- **IFRS 17 Insurance**: Insurance contract accounting
- **Crypto Asset Regulations**: Digital asset risk management
- **ESG Risk Integration**: Sustainability risk frameworks

## Usage Guidelines

### Training Recommendations
1. **Combine with Base Dataset**: Use alongside general financial corpus
2. **Progressive Training**: Start with basic concepts, advance to complex
3. **Validation Testing**: Verify model understanding with held-out samples
4. **Domain Adaptation**: Fine-tune from general financial model

### Evaluation Metrics
- **Calculation Accuracy**: Numerical precision of regulatory ratios
- **Regulatory Knowledge**: Understanding of compliance requirements
- **Risk Interpretation**: Contextual analysis quality
- **Practical Application**: Real-world scenario handling

## Regulatory Disclaimer

This corpus is designed for educational and model training purposes. All regulatory interpretations should be validated against current official guidance from relevant supervisory authorities (Basel Committee, Federal Reserve, ECB, etc.).

---

**Version**: 3.0  
**Last Updated**: December 2024  
**Maintained By**: LLM Finance Leaderboard Team  
**Review Cycle**: Quarterly regulatory updates