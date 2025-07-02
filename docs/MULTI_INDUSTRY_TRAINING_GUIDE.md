# Multi-Industry Training Guide

## Overview

The LLM Leaderboard system has been expanded to support fine-tuning across multiple industries beyond just finance. Users can now select from Finance, Education, Retail, and Healthcare industries, or combine multiple industries for comprehensive training.

## Supported Industries

### 🏦 Finance
- **Focus**: Banking, investment, regulatory compliance, and financial analysis
- **Dataset**: G-SIB banking corpus with Basel III compliance scenarios
- **Sample Count**: 30 high-complexity samples
- **Key Metrics**: Financial accuracy, regulatory compliance, calculation accuracy

### 🎓 Education
- **Focus**: Educational analytics, student performance, and institutional metrics
- **Dataset**: Educational analytics corpus covering K-12 and higher education
- **Sample Count**: 20 medium-complexity samples
- **Key Metrics**: Educational accuracy, statistical analysis, policy compliance

### 🛒 Retail & E-commerce
- **Focus**: Retail analytics, customer metrics, and business performance
- **Dataset**: Retail business analytics covering sales, inventory, and customer analysis
- **Sample Count**: 20 medium-complexity samples
- **Key Metrics**: Retail accuracy, business metrics, customer analysis

### 🏥 Healthcare
- **Focus**: Healthcare analytics, patient outcomes, and clinical metrics
- **Dataset**: Healthcare analytics covering patient care, quality metrics, and outcomes
- **Sample Count**: 20 medium-complexity samples
- **Key Metrics**: Clinical accuracy, patient safety, quality metrics

### 🌐 Multi-Industry
- **Focus**: Combined training across all industry domains
- **Dataset**: Integrated corpus with samples from all industries
- **Sample Count**: 80 mixed-complexity samples
- **Key Metrics**: Cross-domain accuracy, general analytics, domain adaptation

## Dataset Files

```
data/training/
├── synthetic_finance_gsib_v3.jsonl      # Finance industry corpus
├── synthetic_education_v1.jsonl         # Education industry corpus
├── synthetic_retail_v1.jsonl           # Retail industry corpus
├── synthetic_healthcare_v1.jsonl       # Healthcare industry corpus
└── combined_multi_industry_corpus.jsonl # Multi-industry combined corpus
```

## Training Configuration

The system automatically selects the appropriate dataset based on industry selection:

- **Single Industry**: Uses industry-specific dataset
- **Multiple Industries**: Uses combined multi-industry corpus
- **All Industries**: Uses comprehensive multi-industry corpus

### Industry-Specific Evaluation Metrics

Each industry has tailored evaluation metrics:

#### Finance
- Calculation accuracy (30% weight, 1% tolerance)
- Regulatory knowledge (25% weight)
- Explanation quality (25% weight)
- Practical application (20% weight)

#### Education
- Statistical accuracy (35% weight, 2% tolerance)
- Policy knowledge (25% weight)
- Interpretation quality (25% weight)
- Practical application (15% weight)

#### Retail
- Business calculation accuracy (35% weight, 1% tolerance)
- Market knowledge (25% weight)
- Insight quality (25% weight)
- Practical application (15% weight)

#### Healthcare
- Clinical calculation accuracy (40% weight, 0.5% tolerance)
- Medical knowledge (30% weight)
- Interpretation quality (20% weight)
- Practical application (10% weight)

## Using the Training Dashboard

### 1. Industry Selection

1. Navigate to the Training Dashboard
2. In the "Industry Selection" section, choose one or more industries:
   - Check individual industry boxes for specific training
   - Select "Multi-Industry" for comprehensive training
   - Multiple selections automatically use combined corpus

### 2. Model Configuration

1. Select your base model from available options
2. Configure training parameters
3. Enable fine-tuning option

### 3. Job Configuration

1. Provide an optional job name
2. Set priority level (Low, Normal, High)
3. Enable auto-evaluation and result saving

### 4. Dataset Information

The dashboard displays:
- Total sample count for selected industries
- Number of industries selected
- Estimated training time
- Detailed dataset breakdown

### 5. Job Monitoring

Monitor your training jobs with:
- Real-time progress tracking
- Industry-specific information display
- GPU utilization and performance metrics
- Priority-based queue management

## Sample Training Data Format

Each industry dataset follows the instruction-following format:

```json
{
  "instruction": "Calculate the student-teacher ratio for a school district.",
  "input": "Total Students: 12,450, Total Teachers: 485",
  "output": "Student-Teacher Ratio = Total Students / Total Teachers = 12,450 / 485 = 25.7:1. This exceeds the recommended ratio of 20:1 for elementary schools and suggests the need for additional teaching staff.",
  "industry": "education"
}
```

## Training Estimates

### Time Estimates
- **Education**: ~2 hours (20 samples)
- **Retail**: ~2 hours (20 samples)
- **Healthcare**: ~2 hours (20 samples)
- **Finance**: ~3 hours (30 samples)
- **Multi-Industry**: ~8 hours (80 samples)

### Resource Requirements
- **GPU Memory**: 16GB minimum (for 7B models)
- **System Memory**: 32GB recommended
- **Storage**: 50GB for model artifacts

## Best Practices

### Single Industry Training
- Use for domain-specific applications
- Faster training and evaluation
- Higher specialization in chosen domain

### Multi-Industry Training
- Use for general-purpose models
- Better cross-domain generalization
- Longer training time but broader capabilities

### Model Selection
- **7B Models**: Good balance of performance and resource usage
- **13B Models**: Better performance, higher resource requirements
- **3B Models**: Faster training, suitable for experimentation

## Troubleshooting

### Common Issues

1. **Dataset Not Found**
   - Ensure all dataset files exist in `data/training/`
   - Run the test script: `python scripts/test_multi_industry_training.py`

2. **Memory Issues**
   - Reduce batch size in training configuration
   - Use 4-bit quantization for larger models
   - Consider smaller model variants

3. **Training Failures**
   - Check GPU availability and memory
   - Verify dataset format and integrity
   - Review error messages in job monitoring

### Validation

Run the test script to validate your setup:

```bash
python scripts/test_multi_industry_training.py
```

This will verify:
- All dataset files exist and are valid
- Training configuration is properly set up
- Data models support industry information

## Future Enhancements

Planned improvements include:
- Additional industry domains (Legal, Manufacturing, etc.)
- Dynamic dataset mixing ratios
- Industry-specific model architectures
- Advanced evaluation metrics per domain
- Automated hyperparameter tuning per industry

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review system logs in the training dashboard
3. Run the validation test script
4. Check GPU and system resource availability