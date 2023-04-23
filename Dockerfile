FROM huggingface/transformers-pytorch-gpu:4.28.1
WORKDIR /app
RUN set -xe \
    && apt-get update \
    && apt-get install python3-pip -y
RUN apt-get install python3-pip -y
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -r requirements.txt --no-cache-dir
COPY app.py /app/app.py
CMD ["python3", "app.py"]