FROM python:3.11-slim

# Switch to root
USER root

# Set workdir
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 git build-essential curl unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy and unzip the dataset
COPY trained_agents.zip gr_cache.zip ./
RUN unzip trained_agents.zip && \
    unzip gr_cache.zip && \
    rm trained_agents.zip gr_cache.zip && \
    mv trained_agents /trained_agents && \
    mv gr_cache /gr_cache && \
    chown -R ${NB_UID}:${NB_GID} /trained_agents /gr_cache

# Switch back to default user
USER ${NB_UID}

CMD ["bash"]
