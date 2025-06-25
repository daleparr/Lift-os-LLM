# Streamlit Cloud Deployment Guide

## 🚀 **Quick Fix for Streamlit Cloud**

The LLM Finance Leaderboard has been updated to support Streamlit Cloud deployment with lightweight dependencies.

## 📋 **Deployment Steps**

### **1. Update Streamlit Cloud Configuration**

In your Streamlit Cloud app settings:

**Main file path**: `streamlit_app/main_cloud.py`
**Requirements file**: `requirements-streamlit.txt`

### **2. Alternative: Use GitHub Branch**

Create a `streamlit-cloud` branch with cloud-specific configuration:

```bash
git checkout -b streamlit-cloud
git push origin streamlit-cloud
```

Then deploy from the `streamlit-cloud` branch in Streamlit Cloud.

## 🔧 **What Was Fixed**

### **Dependency Issues Resolved**
- ❌ **Removed**: `nvidia-ml-py3>=12.535.0` (not available for Python 3.13)
- ❌ **Removed**: `torch`, `transformers` (too large for Streamlit Cloud)
- ❌ **Removed**: `pinecone-client`, `chromadb` (require API keys/setup)
- ❌ **Removed**: Heavy ML dependencies that cause conflicts

### **Cloud-Compatible Dependencies**
- ✅ **Kept**: `streamlit`, `pandas`, `numpy`, `plotly`
- ✅ **Kept**: `openai`, `anthropic` (for API demos)
- ✅ **Kept**: Basic utilities and visualization libraries

## 📊 **Demo Features Available**

The cloud version includes:

### **📈 Model Leaderboard**
- Sample performance data and visualizations
- Interactive charts and metrics
- Model comparison tables

### **🎯 Training Dashboard**
- Mock training job status
- Configuration interface
- Progress visualization

### **🤖 Agent Analysis**
- Document upload simulation
- Sample analysis workflows
- Multi-agent pipeline demonstration

### **📚 Documentation**
- Complete links to GitHub documentation
- Installation instructions
- Feature overview

## 🔗 **File Structure for Cloud**

```
streamlit_app/
├── main.py              ← Full version (local deployment)
├── main_cloud.py        ← Cloud version (Streamlit Cloud)
└── components/          ← Shared components

requirements.txt         ← Full dependencies (local)
requirements-streamlit.txt ← Cloud dependencies
```

## 🚀 **Deployment Options**

### **Option 1: Update Existing App**
1. Go to Streamlit Cloud app settings
2. Change main file to: `streamlit_app/main_cloud.py`
3. Change requirements file to: `requirements-streamlit.txt`
4. Redeploy

### **Option 2: New App Deployment**
1. Create new Streamlit Cloud app
2. Point to: https://github.com/daleparr/llm_leaderboard
3. Set main file: `streamlit_app/main_cloud.py`
4. Set requirements: `requirements-streamlit.txt`

## 💡 **Demo vs Full Version**

| Feature | Cloud Demo | Full Local Version |
|---------|------------|-------------------|
| **UI Dashboard** | ✅ Full functionality | ✅ Full functionality |
| **Sample Data** | ✅ Mock leaderboard | ✅ Real evaluations |
| **Visualizations** | ✅ Interactive charts | ✅ Live data charts |
| **Model Training** | ❌ Demo only | ✅ Real GPU training |
| **API Integration** | ❌ Demo only | ✅ Live API calls |
| **Agent Workflows** | ❌ Demo only | ✅ Real document analysis |

## 🔧 **For Full Functionality**

To use the complete system with real model evaluation and training:

```bash
# Clone repository
git clone https://github.com/daleparr/llm_leaderboard.git
cd llm_leaderboard

# Local setup
python setup.py
# Edit .env with your API keys

# Run full version
streamlit run streamlit_app/main.py
```

## 📞 **Support**

- **Repository**: https://github.com/daleparr/llm_leaderboard
- **Issues**: Report deployment issues on GitHub
- **Documentation**: Complete guides in `docs/` directory

The cloud demo provides a great overview of the system's capabilities while the full local deployment offers complete functionality for production use.