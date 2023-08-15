FROM python:3.9-bullseye
RUN apt update
RUN apt install -y python3-pip python3-dev 
RUN pip install --upgrade pip
WORKDIR /app
COPY . /app
RUN pip --no-cache-dir install -r requirements.txt
EXPOSE 3000
CMD ["python", "app.py"]
