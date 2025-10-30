# Dockerfile

# 1. Start from a lightweight official Python image
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the file listing our dependencies
COPY requirements.txt .

# 4. Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all our application files into the container
# This includes app.py, the .pkl file, and the templates/ folder
COPY . .

# 6. Expose the port that gunicorn will run on
EXPOSE 8000

# 7. Define the command to run when the container starts
# We use gunicorn to run 4 worker processes, binding to all IPs on port 8000
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8000", "app:app"]