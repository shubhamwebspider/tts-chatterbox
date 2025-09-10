# server_run.py

import os
import multiprocessing

def main():
    # Compute number of workers: default = 2*CPU + 1
    cpu = multiprocessing.cpu_count()
    calculated_workers = cpu * 2 + 1
    max_workers = 3  # Adjust this as needed
    workers = min(calculated_workers, max_workers)

    cmd = [
        "gunicorn",
        "server:app",                     # your ASGI/FastAPI app
        "-k",
        "uvicorn.workers.UvicornWorker",
        "--workers", str(workers),
        "--bind", "0.0.0.0:8000",
        "--keep-alive", "20",
    ]

    # Replace current process with Gunicorn
    os.execvp(cmd[0], cmd)

if __name__ == "__main__":
    main()
