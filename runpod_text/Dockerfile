# Dockerfile
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application, RunPod wrapper code, and start script
COPY main.py ./
COPY runpod_wrapper.py ./
COPY start.sh ./

# Ensure the start script is executable
RUN chmod +x /app/start.sh

# Expose the port FastAPI will run on
EXPOSE 8000

CMD ["/bin/sh","-c","./start.sh"]