FROM nvcr.io/nvidia/deepstream:6.1.1-triton 
RUN pip3 install opencv-python

WORKDIR /opt/nvidia/deepstream/deepstream-6.1/sources/yolov8-triton-deepstream
COPY . /opt/nvidia/deepstream/deepstream-6.1/sources/yolov8-triton-deepstream
WORKDIR /opt/nvidia/deepstream/deepstream-6.1/sources/yolov8-triton-deepstream/configs
RUN pip3 install pyds-1.1.3-py3-none-linux_x86_64.whl
WORKDIR /opt/nvidia/deepstream/deepstream-6.1/sources/yolov8-triton-deepstream
ENTRYPOINT [""]