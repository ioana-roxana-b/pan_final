FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Remove WORKDIR
# WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm && \
    python -m nltk.downloader punkt averaged_perceptron_tagger

COPY . .

ENTRYPOINT ["python3", "main.py"]
