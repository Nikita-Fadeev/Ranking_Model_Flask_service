# Указываем базовый образ Python версии 3.9
FROM python:3.9

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /workspace

# Копируем файл requirements.txt в контейнер
COPY requirements.txt requirements.txt

RUN pip install --upgrade pip

# Устанавливаем зависимости, указанные в requirements.txt
RUN pip install -r requirements.txt

RUN python -m nltk.downloader punkt

# Копируем все файлы проекта в контейнер
COPY . .

ENV FLASK_ENV=development

ENV FLASK_APP=main.py

# Указываем команду для запуска приложения

CMD ["flask", "run", "--host=0.0.0.0"]

