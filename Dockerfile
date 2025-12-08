# 1. Base Image (Lightweight Python)
FROM python:3.9-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy dependencies first (to cache layers)
COPY requirements.txt .

# 4. Install dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the source code and model
COPY src/ src/
COPY experiments/ experiments/

# 6. Expose the port
EXPOSE 8000

# 7. Command to run the API
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]