FROM python:3.11-slim

# Системные зависимости (opencv + tesseract)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем requirements и ставим зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Если у тебя модели в models/, они уже попадут внутрь контейнера

ENV PYTHONUNBUFFERED=1

EXPOSE 5000

# Предполагаем, что Flask-приложение = app:app (файл app.py, объект app)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
