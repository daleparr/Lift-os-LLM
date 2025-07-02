# 🤖 Lift-os-LLM - AI-Native Content Analysis Microservice

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production-ready microservice for LLM-powered content analysis, optimization, and AI surfacing**

A comprehensive API service that leverages Large Language Models to analyze content, optimize for AI search engines, and provide intelligent recommendations for maximum AI visibility.

## 🚀 **Quick Start**

### **Option 1: Docker Deployment (Recommended)**
```bash
git clone https://github.com/daleparr/Lift-os-LLM.git
cd Lift-os-LLM
cp .env.example .env  # Configure your API keys
docker-compose up -d
```

### **Option 2: Local Development**
```bash
git clone https://github.com/daleparr/Lift-os-LLM.git
cd Lift-os-LLM
pip install -r requirements.txt
cp .env.example .env  # Configure your API keys
python -m uvicorn src.main:app --reload
```

### **Option 3: One-Command Setup**
```bash
git clone https://github.com/daleparr/Lift-os-LLM.git
cd Lift-os-LLM
python scripts/setup.py  # Automated setup with dependency installation
```

## 🎯 **Key Features**

### 🧠 **LLM-Powered Analysis**
- **Multi-Model Support**: OpenAI, Anthropic, HuggingFace (7,000+ models)
- **Content Intelligence**: Deep semantic analysis and understanding
- **AI Surfacing Scores**: 6-dimension scoring for AI search optimization
- **Cross-Domain Analysis**: Finance, Education, Retail, Healthcare expertise

### 🔧 **Content Optimization**
- **AI Search Optimization**: Perplexity, SGE, Amazon AI compatibility
- **Structured Data Generation**: Schema.org, JSON-LD, Open Graph
- **Meta Tag Enhancement**: SEO and AI-native search optimization
- **Knowledge Graph Analysis**: Entity relationships and semantic context

### 🚀 **Production-Ready API**
- **FastAPI Framework**: High-performance async API with auto-documentation
- **JWT Authentication**: Secure token-based authentication
- **Rate Limiting**: Configurable request throttling
- **Comprehensive Monitoring**: Health checks, metrics, and observability

### 🤖 **Advanced AI Capabilities**
- **Vector Embeddings**: Semantic similarity and content matching
- **Batch Processing**: High-throughput content analysis
- **Model Fine-tuning**: Custom model training and optimization
- **Multi-Agent Workflows**: Sophisticated content processing pipelines

## 📊 **API Endpoints**

### **Core Analysis**
- `POST /api/v1/analyze` - Comprehensive content analysis with AI scoring
- `POST /api/v1/optimize` - Generate optimized content for AI visibility
- `POST /api/v1/embeddings` - Vector embeddings and semantic analysis
- `POST /api/v1/knowledge-graph` - Entity extraction and relationship mapping

### **Batch Operations**
- `POST /api/v1/batch/analyze` - Bulk content analysis with queue management
- `GET /api/v1/batch/status/{job_id}` - Check batch job status and results

### **Model Management**
- `POST /api/v1/models/evaluate` - Compare model performance
- `POST /api/v1/models/finetune` - Submit fine-tuning jobs
- `GET /api/v1/models/available` - List available models and capabilities

### **Monitoring**
- `GET /health` - Service health check
- `GET /ready` - Readiness probe
- `GET /metrics` - Prometheus metrics
- `GET /docs` - Interactive API documentation

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    Lift-os-LLM API Gateway                 │
├─────────────────────────────────────────────────────────────┤
│  Authentication │ Rate Limiting │ Validation │ Monitoring  │
├─────────────────────────────────────────────────────────────┤
│                   Analysis Services                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────┐ │
│  │ Content     │ │ LLM         │ │ Vector      │ │ Model  │ │
│  │ Analysis    │ │ Orchestrator│ │ Embeddings  │ │ Manager│ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────┘ │
│  ┌─────────────┐ ┌─────────────┐                           │
│  │ Knowledge   │ │ Optimization│                           │
│  │ Graph       │ │ Engine      │                           │
│  └─────────────┘ └─────────────┘                           │
├─────────────────────────────────────────────────────────────┤
│     Redis Cache    │    PostgreSQL  │    Pinecone Vector   │
└─────────────────────────────────────────────────────────────┘
```

## 📁 **Project Structure**

```
lift-os-llm/
├── 📂 src/
│   ├── 📂 api/                 # FastAPI routes and endpoints
│   │   ├── v1/
│   │   │   ├── analysis.py
│   │   │   ├── optimization.py
│   │   │   ├── models.py
│   │   │   └── batch.py
│   │   └── middleware/
│   ├── 📂 services/            # Core business logic
│   │   ├── content_analysis.py
│   │   ├── llm_orchestrator.py
│   │   ├── vector_embeddings.py
│   │   ├── knowledge_graph.py
│   │   └── optimization_engine.py
│   ├── 📂 models/              # Data models and schemas
│   │   ├── requests.py
│   │   ├── responses.py
│   │   └── entities.py
│   ├── 📂 core/                # Core utilities and config
│   │   ├── config.py
│   │   ├── security.py
│   │   ├── database.py
│   │   └── logging.py
│   └── main.py                 # FastAPI application entry point
├── 📂 tests/
│   ├── 📂 unit/                # Unit tests
│   ├── 📂 integration/         # Integration tests
│   └── 📂 fixtures/            # Test data and mocks
├── 📂 scripts/                 # Utility and deployment scripts
│   ├── setup.py                # Automated setup script
│   ├── deploy.sh               # Deployment automation
│   └── migrate.py              # Database migration
├── 📂 .github/workflows/       # CI/CD pipeline
├── 📂 config/                  # Configuration files
├── 🐳 Dockerfile               # Multi-stage container build
├── 🐳 docker-compose.yml       # Production stack
├── 🐳 docker-compose.dev.yml   # Development stack
├── 📋 requirements.txt         # Python dependencies
├── 📋 pyproject.toml           # Project configuration
├── 📝 README.md                # This file
├── 🤝 CONTRIBUTING.md          # Contribution guidelines
├── ⚖️ LICENSE                  # MIT License
└── 🔒 .env.example             # Environment template
```

## 🔧 **Configuration**

### **Required Environment Variables**
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Authentication
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# LLM Providers (choose one or more)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HUGGINGFACE_API_TOKEN=hf_...

# Vector Database
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-west1-gcp

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/liftos_llm
REDIS_URL=redis://localhost:6379

# Monitoring
LOG_LEVEL=INFO
ENABLE_METRICS=true
```

## 🚀 **Usage Examples**

### **Content Analysis**
```python
import requests

# Analyze content for AI surfacing
response = requests.post(
    "http://localhost:8000/api/v1/analyze",
    headers={"Authorization": "Bearer YOUR_JWT_TOKEN"},
    json={
        "content": {
            "title": "Premium Wireless Headphones",
            "description": "High-quality audio experience...",
            "url": "https://example.com/product/123"
        },
        "options": {
            "include_embeddings": True,
            "include_knowledge_graph": True,
            "analysis_depth": "comprehensive"
        }
    }
)

result = response.json()
print(f"AI Surfacing Score: {result['ai_surfacing_score']['overall']}")
```

### **Content Optimization**
```python
# Generate optimized content
response = requests.post(
    "http://localhost:8000/api/v1/optimize",
    headers={"Authorization": "Bearer YOUR_JWT_TOKEN"},
    json={
        "original_content": {
            "html": "<html>...</html>",
            "url": "https://example.com/product/123"
        },
        "optimization_targets": [
            "schema_markup",
            "meta_tags",
            "semantic_clarity",
            "ai_search_compatibility"
        ]
    }
)

optimized = response.json()
print(f"Score Improvement: +{optimized['improvements']['score_increase']} points")
```

### **Batch Processing**
```python
# Submit batch analysis job
response = requests.post(
    "http://localhost:8000/api/v1/batch/analyze",
    headers={"Authorization": "Bearer YOUR_JWT_TOKEN"},
    json={
        "urls": [
            "https://example.com/product/1",
            "https://example.com/product/2",
            "https://example.com/product/3"
        ],
        "options": {
            "analysis_type": "comprehensive",
            "priority": "normal"
        }
    }
)

job_id = response.json()["job_id"]
print(f"Batch job submitted: {job_id}")
```

## 🔒 **Security Features**

### **Authentication & Authorization**
- JWT token-based authentication
- Role-based access control (RBAC)
- API key management for service-to-service communication
- Rate limiting per user/IP address

### **Security Hardening**
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CORS configuration
- Security headers (HSTS, CSP, etc.)

### **Data Protection**
- Encryption in transit (TLS 1.3)
- Secure secrets management
- No persistent storage of sensitive content
- GDPR compliance ready

## 📊 **Performance & Scaling**

### **Performance Metrics**
- **Response Time**: < 2 seconds (95th percentile)
- **Throughput**: 1000+ requests per minute
- **Concurrent Processing**: Configurable worker pools
- **Cache Hit Rate**: 80%+ for repeated analyses

### **Scaling Options**
- **Horizontal Scaling**: Load balancer ready
- **Auto-scaling**: Kubernetes HPA support
- **Resource Optimization**: Intelligent caching and batching
- **Multi-region Deployment**: Global distribution support

## 🧪 **Testing**

### **Run Tests**
```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# All tests with coverage
pytest --cov=src --cov-report=html

# Load testing
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

### **Test Coverage**
- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint testing
- **Load Tests**: Performance and stress testing
- **Security Tests**: Vulnerability scanning

## 🐳 **Docker Deployment**

### **Development**
```bash
# Start development stack
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose logs -f api

# Access API
curl http://localhost:8000/health
```

### **Production**
```bash
# Start production stack
docker-compose up -d

# Scale API instances
docker-compose up --scale api=3

# Monitor resources
docker stats
```

## 📚 **Documentation**

### **API Documentation**
- **Interactive Docs**: http://localhost:8000/docs (Swagger UI)
- **ReDoc**: http://localhost:8000/redoc (Alternative documentation)
- **OpenAPI Spec**: http://localhost:8000/openapi.json

### **Additional Resources**
- **[Contributing Guide](CONTRIBUTING.md)**: Development and contribution guidelines
- **[API Reference](docs/api-reference.md)**: Detailed endpoint documentation
- **[Deployment Guide](docs/deployment.md)**: Production deployment instructions
- **[Architecture Guide](docs/architecture.md)**: System design and components

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### **Development Setup**
```bash
# Clone repository
git clone https://github.com/daleparr/Lift-os-LLM.git
cd Lift-os-LLM

# Install dependencies
pip install -r requirements-dev.txt
pip install -e .

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **FastAPI** for the high-performance web framework
- **HuggingFace** for the transformers library and model hub
- **OpenAI & Anthropic** for LLM API access
- **Pinecone** for vector database capabilities
- **Previous Lift OS Surfacing** for architectural patterns

---

**⭐ Star this repository if you find it useful!**

**🔗 [API Documentation](http://localhost:8000/docs) | [Contributing](CONTRIBUTING.md) | [Deployment Guide](docs/deployment.md)**
