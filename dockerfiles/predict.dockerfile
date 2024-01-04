# Base image
FROM python:3.10-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/
COPY models/ models/

WORKDIR /
RUN pip install . --no-cache-dir #(1)
# --no-cache-dir to prevent caching of downloaded Python packages. This:
# Minimizes the image size: by avoiding the inclusion of cached package files.
# Ensures consistent builds: and not influenced by cached artifacts from previous builds.
# Fetches latest dependencies: avoid relying on outdated versions of dependencies.

ENTRYPOINT ["python", "-u", "src/predict_model.py"]
# the "u" makes sure that any output from our script (e.g. print statements) gets redirected to our terminal.
# If not included you would need to use docker logs to inspect your run.