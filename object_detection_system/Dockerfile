# Use the official Python image from the Docker Hub
FROM python:3.11.7

# Update and install necessary system packages
RUN apt update && apt install -y ffmpeg 

# Install required Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Set the entrypoint to run the Flask application
CMD ["app.py"]
ENTRYPOINT [ "python" ]