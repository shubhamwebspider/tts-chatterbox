# ---------- STAGE 1: Build stage ----------
FROM python:3.12 AS build

WORKDIR /app

# Copy required files and install dependencies
COPY requirements.txt  requirements.txt
RUN pip install --user -r requirements.txt

ENV PATH=/root/.local/bin:$PATH

# ---------- STAGE 2: Production stage ----------
FROM python:3.12-slim AS production

WORKDIR /app

# Copy the installed dependencies
COPY --from=build /root/.local /root/.local

ENV PATH=/root/.local/bin:$PATH

COPY . .

# Expose app port
EXPOSE 8000

# Start the application
#CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
#CMD ["gunicorn","server:app","-k", "uvicorn.workers.UvicornWorker","--workers", "4","--bind", "0.0.0.0:8000", "--keep-alive", "10"]
CMD ["python", "server_run.py"]

# Docker build command
# docker build -t shubhamwebspider/voiceapi:latest .
# Docker run command
# docker run -d -p 8000:8000 shubhamwebspider/voiceapi:latest
