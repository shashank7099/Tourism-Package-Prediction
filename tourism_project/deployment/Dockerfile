
# Use the official lightweight Python image [cite: 1]
FROM python:3.9-slim

# 1. Create a non-root user for security [cite: 1]
RUN useradd -m -u 1000 user

# 2. Set environment variables [cite: 1]
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    REPO_ID="shashankksaxena/tourism-model" \
    MODEL_FILENAME="model.pkl"

# 3. Set the working directory [cite: 1]
WORKDIR $HOME/app

# 4. Install system dependencies [cite: 1]
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements and install [cite: 2]
# Switch to user BEFORE installing to ensure correct permissions in .local
COPY --chown=user requirements.txt .
USER user
RUN pip install --no-cache-dir --user -r requirements.txt

# 6. Copy the rest of the application files [cite: 3]
COPY --chown=user . .

# 7. Expose port 7860 for Hugging Face Spaces [cite: 3]
EXPOSE 7860

# 8. Run the application
# We use 'python -m streamlit' to bypass path/permission issues with the binary
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
