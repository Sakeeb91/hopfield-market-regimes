FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Create directories
RUN mkdir -p data results models

# Set the entry point
ENTRYPOINT ["python", "run_market_regime_classifier.py"]

# Default command
CMD ["--help"] 