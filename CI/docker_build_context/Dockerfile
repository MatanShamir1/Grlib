FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install unzip
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

# Copy and unzip the dataset
COPY dataset.zip .
RUN unzip dataset.zip && rm dataset.zip
RUN mv dataset_new dataset

# Just start with bash by default
CMD [ "bash" ]
