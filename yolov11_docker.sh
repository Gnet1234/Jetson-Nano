#!/bin/bash
xhost +

docker run -it \
--gpus all \
--net=host \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--env="NVIDIA_VISIBLE_DEVICES=all" \
--env="NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video" \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /home/jetson/temp:/ultralytics/ultralytics/temp \
--device=/dev/video0 \
-p 9090:9090 \
-p 8888:8888 \
yahboomtechnology/ultralytics:1.0.3 /bin/bash -c "pip install openpyxl && /bin/bash"
