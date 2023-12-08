#!/bin/bash

# Define the Docker image
docker_image="deepstream_triton6.1:001"

# Check if any container based on the image is running
if docker ps -q --filter ancestor="$docker_image" | grep -q .; then
    echo "Container based on $docker_image is already running. Stopping it..."
    docker stop $(docker ps -q --filter ancestor="$docker_image")
else
    echo "No container based on $docker_image is running."
fi
