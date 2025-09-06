FROM python:3.11-slim-bullseye

WORKDIR /application

COPY requirements.txt .

RUN apt-get -y update && apt-get install -y \
    awscli \
    ffmpeg \
    libsm6 \
    libxext6 \
    unzip \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY application.py .

CMD ["python3", "application.py"]
