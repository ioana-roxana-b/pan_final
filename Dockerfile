FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Minimal system dependencies for spaCy model and NLTK data
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy and NLTK data
RUN python -m spacy download en_core_web_sm && \
    python -m nltk.downloader punkt averaged_perceptron_tagger

# Copy code
COPY . .

# Set entry point
ENTRYPOINT ["python3", "main.py"]
