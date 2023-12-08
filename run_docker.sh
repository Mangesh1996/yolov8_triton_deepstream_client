#!/bin/bash

num=1
docker_image="deepstream_triton_yolov8_6.1:001"

# Check if any container based on the image is running
if docker ps -q --filter ancestor="$docker_image" | grep -q .; then
    echo "Container based on $docker_image is already running. Stopping it..."
    docker stop $(docker ps -q --filter ancestor="$docker_image")
else
    echo "No container based on $docker_image is running."
fi


#Check if the Docker image exists
if docker image inspect "$docker_image" --format="Docker image found."; then
    echo "Proceeding with execution..."
for ((i=0; i<num; i++)); 
    do
    docker run -it   --rm --gpus all --network host \
    -v $PWD:/opt/nvidia/deepstream/deepstream-6.1/sources/yolov8-triton-deepstream   -e XAUTHORITY=$XAUTHORITY -v $XAUTHORITY:$XAUTHORITY    $docker_image  python3 multiple_input_deepstream.py -f file:///opt/nvidia/deepstream/deepstream-6.1/samples/streams/sample_720p.mp4 --port $i
done
else
    echo "Docker image not found. Building the image..."
    docker build -t "$docker_image" .
     for ((i=0; i<num; i++)); do
     docker run -it  -d --rm --gpus all --network host \
    -v $PWD:/opt/nvidia/deepstream/deepstream-6.1/sources/yolov8-triton-deepstream   -e XAUTHORITY=$XAUTHORITY -v $XAUTHORITY:$XAUTHORITY   $docker_image  python3 multiple_input_deepstream.py -f file:///opt/nvidia/deepstream/deepstream-6.1/samples/streams/sample_720p.mp4 --port $i
    done
fi

