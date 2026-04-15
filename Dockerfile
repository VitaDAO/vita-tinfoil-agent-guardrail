FROM python:3.13-slim-bookworm

# Build deps for llama-cpp-python
RUN apt-get update && apt-get install -y cmake build-essential curl && rm -rf /var/lib/apt/lists/*

# Install Python deps
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py ./

RUN mkdir -p /home/user/.cache && useradd -m -d /home/user user && chown -R user:user /app /home/user

ENV HOME=/home/user

USER user
EXPOSE 8000

CMD ["python", "/app/server.py"]
