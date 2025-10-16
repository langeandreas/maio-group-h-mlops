# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3-slim

# Config that can be overridden from the tag workflow
ARG MODEL_VERSION=v0.1
ARG MODEL_PATH=models/model.pkl
ENV MODEL_VERSION=${MODEL_VERSION}
ENV MODEL_PATH=${MODEL_PATH}

EXPOSE 8000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + model artifacts
WORKDIR /maio-group-h
COPY app /maio-group-h/app
COPY models /maio-group-h/models

CMD ["python", "app/app.py"]

