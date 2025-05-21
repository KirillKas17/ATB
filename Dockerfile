# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && \
    apt-get install -y build-essential wget git libffi-dev libssl-dev libatlas-base-dev liblapack-dev gfortran && \
    apt-get install -y libta-lib0 libta-lib0-dev && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Установка pywt через pip (scipy, numpy подтянутся автоматически)
RUN pip install pywt

# Создание рабочей директории
WORKDIR /app

# Копирование проекта
COPY . /app

# Установка зависимостей Python
RUN pip install --no-cache-dir -r config/requirements.txt

# Экспорт переменных окружения
ENV PYTHONUNBUFFERED=1

# Команда запуска по умолчанию
 