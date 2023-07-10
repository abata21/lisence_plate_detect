# -*- coding:utf-8 -*-
import sys
from pathlib import Path
import torch
from function import threshold_blur, find_contour, classes_division, check_class_bright_control, find_lp_chars, \
    distortion_correction

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / "yolov7") not in sys.path:
    sys.path.append(str(ROOT / "yolov7"))
if str(ROOT / "yolov7" / "seg") not in sys.path:
    sys.path.append(str(ROOT / "yolov7" / "seg"))
if str(ROOT / "yolov7" / "seg" / "weights") not in sys.path:
    sys.path.append(str(ROOT / "yolov7" / "seg" / "weights"))
if str(ROOT / "yolov7" / "seg" / "models") not in sys.path:
    sys.path.append(str(ROOT / "yolov7" / "seg" / "models"))
if str(ROOT / "yolov7" / "seg" / "utils") not in sys.path:
    sys.path.append(str(ROOT / "yolov7" / "seg" / "utils"))

# Strong SORT
if str(ROOT / "Yolov7_StrongSORT_OSNet") not in sys.path:
    sys.path.append(str(ROOT / "Yolov7_StrongSORT_OSNet"))
if str(ROOT / "Yolov7_StrongSORT_OSNet" / "strong_sort") not in sys.path:
    sys.path.append(str(ROOT / "Yolov7_StrongSORT_OSNet" / "strong_sort"))
if str(ROOT / "Yolov7_StrongSORT_OSNet" / "strong_sort" / "configs") not in sys.path:
    sys.path.append(str(ROOT / "Yolov7_StrongSORT_OSNet" / "strong_sort" / "configs"))

from yolov7.seg.models.common import DetectMultiBackend
from yolov7.seg.utils.general import (LOGGER, Profile, check_img_size, non_max_suppression, scale_coords)
from yolov7.seg.utils.plots import Annotator, colors, plot_one_box
from yolov7.seg.utils.torch_utils import select_device
from yolov7.seg.utils.augmentations import letterbox
from yolov7.seg.utils.segment.general import process_mask, scale_masks
from yolov7.seg.utils.segment.plots import plot_masks
from Yolov7_StrongSORT_OSNet.strong_sort.utils.parser import get_config
from Yolov7_StrongSORT_OSNet.strong_sort.strong_sort import StrongSORT
import numpy as np
import easyocr
import cv2


class lp_detection_tracking():
    def __init__(self):
        # Switch model
        self.use_detection_model = True
        self.use_ocr_model = True
        self.use_tracking_model = True
        self.use_utils_function = False

        if self.use_detection_model:
            # detection model
            self.yolo_weights = ROOT / 'yolov7/seg/weights/0414_best.pt'
            self.source = ROOT / 'yolov7/seg/data/images'
            self.data = ROOT / 'yolov7/seg/data/custom.yaml'
            self.imgsz = (640, 640)  # inference size (height, width)
            self.conf_thres = 0.8  # confidence threshold
            self.iou_thres = 0.45  # NMS IOU threshold
            self.max_det = 1000  # maximum detections per image
            self.device = '0'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            self.view_img = True  # show results
            self.save_txt = True  # save results to *.txt
            self.save_conf = False  # save confidences in --save-txt labels
            self.save_crop = False  # save cropped prediction boxes
            self.nosave = False  # do not save images/videos
            self.classes = 0, 1, 2, 3, 4, 5, 6, 7  # filter by class: --class 0, or --class 0 2 3
            self.agnostic_nms = False  # class-agnostic NMS
            self.augment = False  # augmented inference
            self.visualize = False  # visualize features
            self.update = False  # update all models
            self.project = ROOT / 'runs/predict-seg'  # save results to project/name
            self.name = 'exp'  # save results to project/name
            self.exist_ok = False  # existing project/name ok, do not increment
            self.line_thickness = 3  # bounding box thickness (pixels)
            self.hide_labels = False  # hide labels
            self.hide_conf = False  # hide confidences
            self.half = False  # use FP16 half-precision inference
            self.dnn = False  # use OpenCV DNN for ONNX inference
            self.prev_truck_location = 0
            self.cur_truck_location = 0
            self.exit_flag = {"True": 0, "Stop": 0, "False": 0}

        if self.use_tracking_model:
            # tracking model
            self.strong_sort_weights = ROOT / 'weights/osnet_x0_25_msmt17.pt'
            self.config_strongsort = ROOT / 'Yolov7_StrongSORT_OSNet/strong_sort/configs/strong_sort.yaml'
            self.tracking_method = 'strongsort'

        if self.use_ocr_model:
            # ocr model
            self.reader = easyocr.Reader(['ko'], gpu=True, model_storage_directory= ROOT / 'weights/easy_ocr_model')
            self.plate_allow_list = '0123456789서울부산대구인천광주대전울산경기강원충북충남전북전남경북경남제주세종' \
                                    '가나다라마' \
                                    '거너더러머버서어저' \
                                    '고노도로모보소오조' \
                                    '구누두루무부수우주' \
                                    '바사아자허하호배 '

    def model_init(self):
        if self.use_detection_model:
            self.detection_model_init()
        if self.use_tracking_model:
            self.tracking_model_init()

    def detection_model_init(self):
        # yolo
        self.device = select_device(self.device)
        self.yolo_model = DetectMultiBackend(self.yolo_weights, device=self.device, dnn=self.dnn, data=self.data,
                                             fp16=self.half)
        # self.model = attempt_load(self.yolo_weights, map_location=self.device)
        self.stride, self.names, self.pt = self.yolo_model.stride, self.yolo_model.names, self.yolo_model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        self.bs = 1
        self.yolo_model.warmup(imgsz=(1 if self.pt else self.bs, 3, *self.imgsz))
        self.seen, self.windwos, self.dt = 0, [], (Profile(), Profile(), Profile())

    def tracking_model_init(self):
        nr_sources = 1
        # strong sort
        self.cfg = get_config()
        self.cfg.merge_from_file(self.config_strongsort)
        self.strong_sort_list = []
        for i in range(nr_sources):
            self.strong_sort_list.append(
                StrongSORT(
                    self.strong_sort_weights,
                    self.device,
                    self.half,
                    max_dist=self.cfg.STRONGSORT.MAX_DIST,
                    max_iou_distance=self.cfg.STRONGSORT.MAX_IOU_DISTANCE,
                    max_age=self.cfg.STRONGSORT.MAX_AGE,
                    n_init=self.cfg.STRONGSORT.N_INIT,
                    nn_budget=self.cfg.STRONGSORT.NN_BUDGET,
                    mc_lambda=self.cfg.STRONGSORT.MC_LAMBDA,
                    ema_alpha=self.cfg.STRONGSORT.EMA_ALPHA,
                )
            )
        self.outputs = [None] * nr_sources
        self.trajectory = {}
        self.curr_frames, self.prev_frames = [None] * nr_sources, [None] * nr_sources

    def xyxy2xywh(self, x):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    # def inference_img(self, img_to_infer, plate_detection_tracking_flag=False):
    #     img = letterbox(img_to_infer, self.imgsz, stride=self.stride, auto=self.pt)[0]
    #     img = img.transpose((2, 0, 1))[::-1]
    #     im = np.ascontiguousarray(img)
    #     s = ""
    #     ocr_words = ""
    #     # id_list = []
    #     truck_id_list = []
    #     plate_id_list = []
    #     plate_detection_flag = False
    #     cp = None
    #     exit_vehicle = False
    #     if plate_detection_tracking_flag == False:
    #         self.use_tracking_model = False
    #     elif plate_detection_tracking_flag == True:
    #         self.use_tracking_model = True
    #
    #     with self.dt[0]:
    #         im = torch.from_numpy(im).to(self.device)
    #         im = im.half() if self.half else im.float()  # uint8 to fp16/32
    #         im /= 255  # 0 - 255 to 0.0 - 1.0
    #         if len(im.shape) == 3:
    #             im = im[None]  # expand for batch dim
    #
    #     with self.dt[1]:
    #         pred, out = self.yolo_model(im, augment=self.augment)
    #         proto = out[1]
    #
    #     with self.dt[2]:
    #         pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
    #                                    max_det=self.max_det, nm=32)
    #
    #     for i, det in enumerate(pred):
    #         self.seen += 1
    #         im0 = img_to_infer.copy()
    #
    #         if self.use_tracking_model:
    #             self.curr_frames[i] = im0
    #
    #         self.annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
    #
    #         s += '%gx%g' % im.shape[2:]
    #
    #         if self.use_tracking_model:
    #             if self.cfg.STRONGSORT.ECC:
    #                 self.strong_sort_list[i].tracker.camera_update(self.prev_frames[i], self.curr_frames[i])
    #
    #         if det is not None and len(det):
    #             all_bbox = det[:, :4]
    #             all_cls = det[:, 5]
    #
    #             masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
    #             det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
    #
    #             # segmentation
    #             mcolors = [colors(int(cls), True) for cls in det[:, 5]]
    #             # mcolors = [colors(int(cls), True) if cls != 7 else masks for cls in det[:, 5]]
    #             im_masks = plot_masks(im[i], masks, mcolors)
    #
    #             self.annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)
    #
    #             # masking 된 부분 추출, pass Truck object
    #             for index, value in enumerate(all_cls):
    #                 if not int(value) == 7:
    #                     plate_detection_flag = True
    #                     cls_tensor = all_bbox[index].unsqueeze(0)
    #                     plate_mask = process_mask(proto[i], det[:, 6:], cls_tensor, im0.shape[0:2], upsample=True)
    #
    #             # Print results
    #             for c in det[:, -1].unique():
    #                 n = (det[:, -1] == c).sum()
    #                 s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
    #
    #             xywhs = self.xyxy2xywh(det[:, 0:4])
    #             confs = det[:, 4]
    #             clss = det[:, 5]
    #
    #             if self.use_tracking_model:
    #                 self.outputs[i] = self.strong_sort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
    #
    #             if self.use_detection_model and self.use_tracking_model:
    #                 if len(self.outputs[i]) > 0:
    #                     for j, (output, conf) in enumerate(zip(self.outputs[i], confs)):
    #                         bboxes = output[0:4]
    #                         id = output[4]
    #                         cls = output[5]
    #
    #                         x = int(bboxes[0])
    #                         y = int(bboxes[1])
    #                         w = int(bboxes[2]) - x
    #                         h = int(bboxes[3]) - y
    #                         c = int(cls)
    #                         id = int(id)
    #
    #                         # 프레임 전체 이미지의 100분의 1보다 작으면 출차 True
    #                         if w * h < im0.shape[0] * im0.shape[1] * 0.01:
    #                             exit_vehicle = True
    #
    #                         if self.use_ocr_model and not self.use_utils_function:
    #                             crop_plate = im0[y + (h // 20):y + h - (h // 20), x + (w // 25):x + w - (w // 25)]
    #
    #                         if c != 7:
    #                             self.cur_truck_location = x
    #                             truck_value = self.cur_truck_location - self.prev_truck_location
    #                             if truck_value < 0:
    #                                 self.exit_flag["False"] += 1
    #                                 self.prev_truck_location = self.cur_truck_location
    #                             elif truck_value == 0:
    #                                 self.exit_flag["Stop"] += 1
    #                                 self.prev_truck_location = self.cur_truck_location
    #                             else:
    #                                 self.exit_flag["True"] += 1
    #                                 self.prev_truck_location = self.cur_truck_location
    #
    #                             if self.use_utils_function:
    #                                 crop_plate = distortion_correction(plate_mask, im0)
    #                                 cph, cpw, cc = crop_plate.shape
    #
    #                                 if c == 0 or c == 1 or c == 5 or c == 6:  # 초록,주황 번호판(하얀 문자)
    #                                     bright_cp = check_class_bright_control(crop_plate)
    #                                     cp = ~bright_cp
    #                                 else:
    #                                     cp = check_class_bright_control(crop_plate)
    #
    #                                 img_blur_thresh = threshold_blur(cp)
    #                                 contours_dict = find_contour(img_blur_thresh)
    #                                 matched_result = find_lp_chars(contours_dict, cpw, cph)
    #                                 if matched_result:
    #                                     crop_plate = classes_division(matched_result, cp, cpw, cph, c)
    #                                 else:
    #                                     crop_plate = cp
    #
    #                             if self.use_ocr_model:
    #                                 ocr_result = self.reader.readtext(crop_plate, batch_size=5,
    #                                                                   allowlist=self.plate_allow_list)
    #                                 # ocr_test_img = crop_plate.copy()
    #
    #                                 for (bbox, text, prob) in ocr_result:
    #                                     ocr_words += text
    #                                     # (tl, tr, br, bl) = bbox
    #                                     # tl = (int(tl[0]), int(tl[1]))
    #                                     # tr = (int(tr[0]), int(tr[1]))
    #                                     # br = (int(br[0]), int(br[1]))
    #                                     # bl = (int(bl[0]), int(bl[1]))
    #                                     # cv2.rectangle(ocr_test_img, tl, br, color=(0, 255, 0), thickness=1)
    #                             plate_id_list.append([id, ocr_words])
    #                         else:
    #                             truck_id_list.append(id)
    #
    #                         label = None if self.hide_labels else (
    #                             f'{id} {self.names[c]}' if self.hide_conf else f'{id} {conf:.2f}')
    #                         # plot_one_box(bboxes, im0, label=label, color=colors(c, True), line_thickness=2)
    #                         color = colors(c, True)
    #                         self.annotator.box_label(bboxes, label, ocr_words, color=color)
    #                         # id_list.append([id, ocr_words])
    #
    #                 self.prev_frames[i] = self.curr_frames[i]
    #
    #             elif self.use_detection_model and not self.use_tracking_model:
    #                 for *xyxy, conf, cls in reversed(det[:, :6]):
    #                     x = int(xyxy[0])
    #                     y = int(xyxy[1])
    #                     w = int(xyxy[2]) - x
    #                     h = int(xyxy[3]) - y
    #                     c = int(cls)
    #                     label = None if self.hide_labels else (
    #                         self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
    #                     self.annotator.box_label(xyxy, label, ocr_words, color=colors(c, True))
    #         else:
    #             if self.use_tracking_model:
    #                 self.strong_sort_list[i].increment_ages()
    #             else:
    #                 plate_detection_flag = False
    #         im0 = self.annotator.result()
    #
    #     return im0, plate_detection_flag, plate_id_list, truck_id_list, self.exit_flag if self.use_tracking_model else None, cp, exit_vehicle

    def inference_img2(self, img_to_infer, plate_detection_tracking_flag=False):
        img = letterbox(img_to_infer, self.imgsz, stride=self.stride, auto=self.pt)[0]
        img = img.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(img)
        s = ""
        ocr_words = ""
        # id_list = []
        truck_id_list = []
        plate_id_list = []
        plate_detection_flag = False
        cp = None
        exit_vehicle = False
        if plate_detection_tracking_flag == False:
            self.use_tracking_model = False
        elif plate_detection_tracking_flag == True:
            self.use_tracking_model = True

        with self.dt[0]:
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        with self.dt[1]:
            pred, out = self.yolo_model(im, augment=self.augment)
            proto = out[1]

        with self.dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                       max_det=self.max_det, nm=32)

        for i, det in enumerate(pred):
            self.seen += 1
            im0 = img_to_infer.copy()

            if self.use_tracking_model:
                self.curr_frames[i] = im0

            self.annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))

            s += '%gx%g' % im.shape[2:]

            if self.use_tracking_model:
                if self.cfg.STRONGSORT.ECC:
                    self.strong_sort_list[i].tracker.camera_update(self.prev_frames[i], self.curr_frames[i])

            if det is not None and len(det):
                all_bbox = det[:, :4]
                all_cls = det[:, 5]

                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # segmentation
                mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                # mcolors = [colors(int(cls), True) if cls != 7 else masks for cls in det[:, 5]]
                im_masks = plot_masks(im[i], masks, mcolors)

                self.annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)

                # masking 된 부분 추출, pass Truck object
                for index, value in enumerate(all_cls):
                    if not int(value) == 7:
                        plate_detection_flag = True
                        cls_tensor = all_bbox[index].unsqueeze(0)
                        plate_mask = process_mask(proto[i], det[:, 6:], cls_tensor, im0.shape[0:2], upsample=True)

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "

                xywhs = self.xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                if self.use_tracking_model:
                    self.outputs[i] = self.strong_sort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                if self.use_detection_model and self.use_tracking_model:
                    if len(self.outputs[i]) > 0:
                        for j, (output, conf) in enumerate(zip(self.outputs[i], confs)):
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]

                            x = int(bboxes[0])
                            y = int(bboxes[1])
                            w = int(bboxes[2]) - x
                            h = int(bboxes[3]) - y
                            c = int(cls)
                            id = int(id)

                            if self.use_ocr_model and not self.use_utils_function:
                                crop_plate = im0[y + (h // 20):y + h - (h // 20), x + (w // 25):x + w - (w // 25)]

                            if c != 7:
                                # 프레임 전체 이미지의 100분의 1보다 작으면 출차 True
                                if w * h < im0.shape[0] * im0.shape[1] * 0.01:
                                    exit_vehicle = True

                                self.cur_truck_location = x
                                truck_value = self.cur_truck_location - self.prev_truck_location
                                if truck_value < 0:
                                    self.exit_flag["False"] += 1
                                    self.prev_truck_location = self.cur_truck_location
                                elif truck_value == 0:
                                    self.exit_flag["Stop"] += 1
                                    self.prev_truck_location = self.cur_truck_location
                                else:
                                    self.exit_flag["True"] += 1
                                    self.prev_truck_location = self.cur_truck_location

                                if self.use_utils_function:
                                    crop_plate = distortion_correction(plate_mask, im0)
                                    cph, cpw, cc = crop_plate.shape

                                    if c == 0 or c == 1 or c == 5 or c == 6:  # 초록,주황 번호판(하얀 문자)
                                        bright_cp = check_class_bright_control(crop_plate)
                                        cp = ~bright_cp
                                    else:
                                        cp = check_class_bright_control(crop_plate)

                                    img_blur_thresh = threshold_blur(cp)
                                    contours_dict = find_contour(img_blur_thresh)
                                    matched_result = find_lp_chars(contours_dict, cpw, cph)
                                    if matched_result:
                                        crop_plate = classes_division(matched_result, cp, cpw, cph, c)
                                    else:
                                        crop_plate = cp

                                if self.use_ocr_model:
                                    ocr_result = self.reader.readtext(crop_plate, batch_size=5,
                                                                      allowlist=self.plate_allow_list)
                                    for (bbox, text, prob) in ocr_result:
                                        ocr_words += text
                                plate_id_list.append([id, ocr_words])
                            else:
                                truck_id_list.append(id)

                            label = None if self.hide_labels else (
                                f'{id} {self.names[c]}' if self.hide_conf else f'{id} {conf:.2f}')
                            # plot_one_box(bboxes, im0, label=label, color=colors(c, True), line_thickness=2)
                            color = colors(c, True)
                            self.annotator.box_label(bboxes, label, ocr_words, color=color)
                            if not ocr_words == '':
                                ocr_words = ''
                            # id_list.append([id, ocr_words])

                    self.prev_frames[i] = self.curr_frames[i]

                elif self.use_detection_model and not self.use_tracking_model:
                    for *xyxy, conf, cls in reversed(det[:, :6]):
                        x = int(xyxy[0])
                        y = int(xyxy[1])
                        w = int(xyxy[2]) - x
                        h = int(xyxy[3]) - y
                        c = int(cls)
                        label = None if self.hide_labels else (
                            self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                        self.annotator.box_label(xyxy, label, ocr_words, color=colors(c, True))
            else:
                if self.use_tracking_model:
                    self.strong_sort_list[i].increment_ages()
                else:
                     plate_detection_flag = False
            im0 = self.annotator.result()

        # return im0,  plate_detection_flag, plate_id_list, truck_id_list, self.exit_flag if self.use_tracking_model else None, cp, exit_vehicle
        return im0, plate_detection_flag, plate_id_list, truck_id_list, self.exit_flag, cp, exit_vehicle
