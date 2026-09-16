FROM python:3.10-slim

WORKDIR /app

# Install dependencies required for tf2onnx
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and create directories
COPY src/ /app/src/
RUN mkdir -p /app/data /app/models

# Command to run by default: generate data, train model, and simulate edge inference!
CMD ["sh", "-c", "python src/data_gen.py && python src/train.py && echo '\n--- Tests complete, starting edge simulation! ---' && python src/simulate_edge.py"]
