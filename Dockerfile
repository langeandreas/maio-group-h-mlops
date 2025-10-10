# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3-slim

EXPOSE 8000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /maio-group-h
COPY app /maio-group-h/app
COPY models /maio-group-h/models

CMD ["python", "app/app.py"]

