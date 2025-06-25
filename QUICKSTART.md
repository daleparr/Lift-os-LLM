# LLM Finance Leaderboard - Quick Start Guide

Welcome to the LLM Finance Leaderboard! This guide will help you get up and running quickly.

## 🚀 Quick Setup (5 minutes)

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd llm-finance-leaderboard
python setup.py
```

### 2. Configure API Keys
Edit the `.env` file with your API keys:
```bash
# Required for vector storage
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here

# Optional for model access
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here  # ONE token for 7,000+ models
```

### 3. Launch Dashboard
```bash
streamlit run streamlit_app/main.py
```

Visit `http://localhost:8501` to see the dashboard!

## 🎯 What You Get

### Interactive Dashboard
- **🏆 Leaderboard**: Real-time model rankings with quality and efficiency metrics
- **📊 Model Comparison**: Side-by-side performance analysis
- **📋 Task Analysis**: Breakdown by complexity tiers (Low/Medium/High)
- **⚙️ Data Management**: Monitor data collection and vector store status
- **🚀 Benchmark Runner**: Execute evaluations with custom configurations

### Three Complexity Tiers
- **Low (30%)**: EPS extraction, ratio identification from single documents
- **Medium (40%)**: Multi-document analysis, sentiment classification
- **High (30%)**: Causal reasoning, forecasting, portfolio recommendations

### Supported Models
- **Stock Models**: Mistral 7B, Llama 2 13B, Falcon 40B, Qwen 7B, Phi-3 Mini
- **Finance-Tuned**: FinMA 7B, FinGPT variants, custom LoRA fine-tunes

## 🧪 Demo Mode

Run a demonstration benchmark:
```bash
python scripts/run_benchmark.py --verbose
```

This will:
- Simulate evaluations on sample models
- Generate mock performance data
- Save results to the database
- Display results in the dashboard

## 📊 Real Data Collection

### SEC Filings
```python
from src.data.collectors.sec_filings import SECFilingCollector

collector = SECFilingCollector()
filings = collector.collect_filings(
    tickers=["JPM", "BAC", "C"],
    filing_types=["10-Q", "10-K"],
    limit_per_company=10
)
```

### Market Data
```python
from src.data.collectors.market_data import MarketDataCollector

collector = MarketDataCollector()
data = collector.collect_daily_data(
    tickers=["JPM", "BAC", "C"],
    start_date="2023-01-01",
    end_date="2024-01-01"
)
```

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Layer    │    │ LangGraph Pipeline│    │ Evaluation      │
│                 │    │                  │    │                 │
│ • SEC Filings   │───▶│ Retriever Node  ▶│───▶│ Quality Metrics │
│ • Transcripts   │    │ Parser Node    ▶ │    │ Efficiency      │
│ • Market Data   │    │ Analysis Node  ▶ │    │ Success Rate    │
│ • News          │    │ Draft Node     ▶ │    │                 │
│                 │    │ Critic Node    ▶ │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Pinecone Vector │    │   AgentState     │    │   Leaderboard   │
│ Store           │    │   Management     │    │   Dashboard     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔧 Configuration

### Model Configuration
Edit `src/config/models_config.yaml` to add new models:
```yaml
custom_models:
  my_finance_model:
    name: "my-org/finance-llama-7b"
    display_name: "Custom Finance Llama 7B"
    provider: "huggingface"
    parameters: "7B"
    cost_per_1k_tokens: 0.0002
```

### Task Configuration
Edit `configs/benchmark_config.yaml` to customize evaluation:
```yaml
evaluation:
  seeds: [42, 123, 456]
  temperature: 0.1
  timeout_minutes: 30

scoring:
  weights:
    quality: 0.9
    efficiency: 0.1
```

## 📈 Scoring System

### Composite Score Formula
```
QualityScore = 0.3×Low_F1 + 0.4×Med_ROUGE + 0.3×Hard_Human
CostScore = normalize(latency × $/kTok)
FinalScore = 0.9×QualityScore + 0.1×(1 - CostScore)
```

### Quality Metrics by Tier
- **Low**: Exact match F1, token overlap
- **Medium**: ROUGE-1/2, FactScore
- **High**: Human ratings, MAPE vs consensus

## 🐳 Docker Deployment

### Development
```bash
docker-compose up -d
```

### Production
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 📚 Key Files

| File | Purpose |
|------|---------|
| `streamlit_app/main.py` | Main dashboard application |
| `src/config/settings.py` | Configuration management |
| `src/data/collectors/` | Data collection modules |
| `src/agents/` | Multi-agent pipeline |
| `src/evaluation/` | Scoring and metrics |
| `configs/benchmark_config.yaml` | Evaluation settings |
| `scripts/run_benchmark.py` | Benchmark execution |

## 🔍 Troubleshooting

### Common Issues

**"Pinecone connection failed"**
- Check your API key and environment in `.env`
- Ensure you have a Pinecone account and index

**"Module not found"**
- Run `pip install -r requirements.txt`
- Check Python version (3.9+ required)

**"Database locked"**
- Stop any running processes
- Delete `data/leaderboard.db` and restart

**"Streamlit won't start"**
- Check port 8501 isn't in use
- Try `streamlit run streamlit_app/main.py --server.port 8502`

### Getting Help

1. Check the logs in `logs/leaderboard.log`
2. Run with `--verbose` flag for detailed output
3. Review the architecture document: `LLM_Finance_Leaderboard_Architecture.md`

## 🎯 Next Steps

1. **Add Real Data**: Configure API keys and collect actual financial data
2. **Custom Models**: Add your fine-tuned models to the registry
3. **New Tasks**: Extend the task suite with domain-specific evaluations
4. **Production Deploy**: Use Docker for scalable deployment
5. **Monitoring**: Set up W&B logging for experiment tracking

## 📄 License

MIT License - see LICENSE file for details.

---

**Happy benchmarking! 🚀**

For detailed architecture information, see `LLM_Finance_Leaderboard_Architecture.md`