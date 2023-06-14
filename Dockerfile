FROM python:3.8
ENV PYTHONUNBUFFERED=1
WORKDIR /code
COPY requirements.txt /code/
RUN pip --no-cache-dir install -r requirements.txt
COPY . /code/
