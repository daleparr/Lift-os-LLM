FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY streamlit_app/ ./streamlit_app/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY pyproject.toml .

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/results logs

# Expose ports
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "streamlit_app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]