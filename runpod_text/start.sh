# start.sh
#!/bin/sh

# Run the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Run the RunPod wrapper script
python runpod_wrapper.py
