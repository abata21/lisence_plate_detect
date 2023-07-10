# Nvidia cuda used
FROM nvidia/cuda:11.7.0-devel-ubuntu18.04
RUN apt-get update -y
RUN apt install python3.8 -y
RUN apt-get install -y python3-pip python3-dev build-essential
RUN apt-get install -y git

# opencv Error problem solution
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get -y install pkg-config
#RUN apt-get -y install libgtk2.0-dev
#RUN apt -y install make g++ pkg-config libgl1-mesa-dev libxcb*-dev libfontconfig1-dev libxkbcommon-x11-dev python libgtk-3-dev

# COPY folder
COPY test /yolo_ocr/test
COPY weights /yolo_ocr/weights
COPY yolov7 /yolo_ocr/yolov7
COPY Yolov7_StrongSORT_OSNet /yolo_ocr/Yolov7_StrongSORT_OSNet
COPY function.py /yolo_ocr/function.py
COPY lp_detection_tracking.py /yolo_ocr/lp_detection_tracking.py
COPY main.py /yolo_ocr/main.py
COPY ocr_log.txt /yolo_ocr/ocr_log.txt
COPY requirements.txt /yolo_ocr/requirements.txt

RUN pip3 install -U pip setuptools wheel
RUN pip3 install -r /yolo_ocr/requirements.txt

# RTX 3090 version
RUN pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# locale
RUN apt-get install -y locales
RUN localedef -f UTF-8 -i ko_KR ko_KR.UTF-8
ENV LC_ALL ko_KR.UTF-8
ENV PYTHONIOENCODING=utf-8

# setup module
RUN git clone https://github.com/KaiyangZhou/deep-person-reid
WORKDIR /deep-person-reid
RUN python3 setup.py develop

#RUN git clone https://github.com/XPixelGroup/BasicSR.git
#WORKDIR /BasicSR
#RUN python3 setup.py develop

# Keep Runing process (dedug)
ENTRYPOINT ["tail", "-f", "/dev/null"]

#EXPOSE ["80":"8080"]
#ENTRYPOINT ["python"]
#CMD ["main.py -c ./test/test6.mp4"]
