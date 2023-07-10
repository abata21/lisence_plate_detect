# YOLOv7 + EASY OCR
* yolov7 모델과 easy ocr 모델 결합
* GAN 모델
* Tracking 모델

# Use
* python main.py "-c" or "--camera" 카메라 경로 입력

***
## Requirements
com/requirements.txt 사용 및 easyocr 설치

* TorchReid 사용<Tracking 모델 사용>
> git clone https://github.com/KaiyangZhou/deep-person-reid
> 
> cd deep-person-reid Folder
> 
> python setup.py develop

* EasyOCR 사용
>pip install easyocr
 

~~* BaicSR 사용~~<GAN 모델 사용>
> git clone https://github.com/XPixelGroup/BasicSR.git
> 
> cd BasicSR Folder
> 
> python setup.py develop


~~* Real-ESRGAN 모델 사용~~<GAN 모델 사용>
>git clone https://github.com/xinntao/Real-ESRGAN
> 
> cd Real-ESRGAN Folder
> 
> python setup.py develop

***
## Open Source URL
* yolov7 모델 requirements
  >[yolov7-u7](https://github.com/WongKinYiu/yolov7/tree/u7)
* easy ocr 모델 requirements
  >[easyocr](https://github.com/JaidedAI/EasyOCR)
* Yolov7_StrongSORT_OSNet requirements
  >[yolov7_StrongSORT](https://github.com/mikel-brostrom/Yolov7_StrongSORT_OSNet)
* Real-esrgan 모델 requirements
  >[SRGAN-PyTOrch](https://github.com/xinntao/Real-ESRGAN)
* BasicSR 모듈 <Real-ESRGAN 모델에서 사용>
  >[BasicSR](https://github.com/XPixelGroup/BasicSR)
* TorchReid <tracking 모델에서 사용>
  >[Torchreid](https://github.com/KaiyangZhou/deep-person-reid)

***
## warning
* easy ocr 사용 시 opencv 버전에 따라 안되는 것이 있음. 최신버전 사용
* opencv-python-4.7.0.68 구동확인