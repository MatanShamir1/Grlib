FROM python:3.11-slim

# Switch to root
USER root

# Set workdir
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 git build-essential curl unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy and unzip the dataset
COPY dataset.zip .
RUN unzip dataset.zip && rm dataset.zip
RUN mv dataset_new /preloaded_data

# Change ownership so jovyan can access
RUN chown -R ${NB_UID}:${NB_GID} /preloaded_data

# Switch back to default user
USER ${NB_UID}

CMD ["bash"]
