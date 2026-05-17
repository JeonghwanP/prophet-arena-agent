FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent files
COPY my_agent.py .
COPY server.py .

# Expose port
EXPOSE 8000

# Start the server
CMD ["python", "server.py"]
