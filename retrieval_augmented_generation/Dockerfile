# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any necessary dependencies specified in requirements.txt
RUN pip install -r requirements.txt

# Download the NLTK 'punkt' data
RUN python -m nltk.downloader punkt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME rag_venv

# Run app.py when the container launches
CMD ["python", "app.py"]