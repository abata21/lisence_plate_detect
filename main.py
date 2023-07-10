# -*- coding:utf-8 -*-
import datetime
import os
import cv2
import time
from PIL import Image
import lp_detection_tracking
import numpy as np
import string
import argparse
import logging
logging.basicConfig(level=logging.INFO)

def video_show(original_img, infer_img, fps_str, window_name):
    # resize video
    resize_original_img = cv2.resize(original_img, (640, 480))
    resize_infer_img = cv2.resize(infer_img, (640, 480))

    # Check FPS
    cv2.putText(resize_original_img, fps_str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    # Convert to one Window
    numpy_horizontal_img = np.hstack((resize_original_img, resize_infer_img))
    cv2.imshow(window_name, numpy_horizontal_img)
    cv2.waitKey(1)


def fps_calculator(cur_time, prev_time):
    sec = cur_time - prev_time
    prev_time = cur_time

    fps = 1 / (sec)
    fps_str = "FPS : %0.1f" % fps
    return fps_str, prev_time


# def input_rtsp(camera_path, log_save_s, lp_conf, lp_img_save):
#     prev_time = 0
#     detection_yolo = lp_detection_tracking.lp_detection_tracking()
#     detection_yolo.model_init()
#     rtsp = cv2.VideoCapture(camera_path)
#
#     """
#     plate_detection_flag : yolo 모델에서 번호판이 탐지되었을 경우 알려주는 플래그
#     plate_detection_tracking_flag : 설정된 프레임 후 모든 처리과정을 실행시키기 위한 플래그
#     detection_frame_count :작은 크기의 번호판이 아닌 온전한 번호판을 탐지하기 위한 프레임 무시 번호, 설정된 프레임 동안 번호판만 탐지하기 위함
#     del_id_list : 삭제할 ID 리스트
#     detect_plate_id_dict : 탐지된 번호판 딕셔너리 {"id" : [frame count, 24frame chek]}
#     plate_ocr_4word_dict : 번호판 마지막 4자리 점수 계산 딕셔너리 {ocr 4word : value}
#     plate_ocr_all_number_dict : 번호판 전체 번호 점수 계산 딕셔너리 {ocr word : value}
#     plate_ocr_5_char_dict : 뒤에서 다섯번째 '한글'을 나타내기 위한 딕셔너리 {ocr 5char : value}
#     """
#     plate_detection_tracking_flag = False
#     detection_frame_count = 0
#     del_id_list = []
#     detect_plate_id_dict = {}
#     plate_ocr_4word_dict = {}
#     plate_ocr_all_number_dict = {}
#     plate_ocr_5_char_dict = {}
#     ocr_list = []
#     count_24 = 0
#     frame_count = 0
#
#     while True:
#         rtsp_ret, rtsp_img = rtsp.read()
#         rtsp_cur_time = time.time()
#
#         if rtsp_ret:
#             """
#             딕셔너리를 확인하고 설정된 프레임을 만족한 키 값을 삭제하기 위함
#             딕셔너리를 탐지하고 있는동안에는 바로 삭제가 불가능, 리스트를 만들어 아이디 값을 저장하고 저장된 아이디를 이용 딕셔너리 삭제
#             """
#             # if detect_plate_id_dict is not None:
#             #     for key, value in detect_plate_id_dict.items():
#             #         if value[1]:
#             #             del_id_list.append(key)
#             #
#             # if len(del_id_list) > 0:
#             #     for del_id_key in del_id_list:
#             #         if del_id_key in detect_plate_id_dict:
#             #             del detect_plate_id_dict[del_id_key]
#
#             if detection_frame_count == 30:
#                 plate_detection_tracking_flag = True
#
#             # plate_id_list = [id, ocr_words]
#             rtsp_infer_img, plate_detection_flag, plate_id_list, truck_id_list, exit_flag, crop_plate, exit_vehicle=\
#                 detection_yolo.inference_img(rtsp_img, plate_detection_tracking_flag=plate_detection_tracking_flag)
#
#             if exit_vehicle:  # 출차 차량의 경우
#                 plate_detection_tracking_flag = False
#                 detection_frame_count = 0
#
#             if not plate_detection_flag:
#                 detection_frame_count = 0
#
#             if plate_detection_flag and not plate_detection_tracking_flag:
#                 detection_frame_count += 1
#
#             logging.info("Detection Frame Count : %i,"
#                          " count_24 : %i,"
#                          " plate_detection_flag : %s,"
#                          " plate_detection_tracking_flag : %s,"
#                          " exit_vehicle : %s",
#                          detection_frame_count,
#                          count_24,
#                          plate_detection_flag,
#                          plate_detection_tracking_flag,
#                          exit_vehicle
#                          )
#             small_plate_detect = False
#             if plate_detection_tracking_flag:
#                 for index, value in enumerate(plate_id_list):
#                     plate_id = value[0]
#                     ocr_word = value[1]
#
#                     all_plate_number = ocr_word.translate(str.maketrans('', '', string.punctuation))
#                     all_plate_number = all_plate_number.replace(" ", "")
#                     last_4_word = all_plate_number[-4:]
#
#                     # 문자열 의 마지막 4자리와 맨앞 첫 글자가 같을 경우 초기화
#                     # OCR 결과가 좋지 않으면 밑의 소스도 무용지물
#                     # if ocr_list and last_4_word == ocr_list[2] and all_plate_number[0] == ocr_list[1][0]:
#                     if ocr_list and last_4_word == ocr_list[2]:
#                         # print("check")
#                         plate_detection_tracking_flag = False
#                         detection_frame_count = 0
#                         count_24 = 0
#                         continue
#                     else:
#                         if 5 < len(all_plate_number):
#                             # last 4 word score
#                             if last_4_word not in plate_ocr_4word_dict:
#                                 plate_ocr_4word_dict[last_4_word] = 1
#                             else:
#                                 plate_ocr_4word_dict[last_4_word] += 1
#
#                             # all number score, 5th korean check
#                             # if len(all_plate_number) >= 5:
#                             if all_plate_number not in plate_ocr_all_number_dict:
#                                 if all_plate_number[-5].isalpha():
#                                     plate_ocr_5_char_dict[all_plate_number[-5]] = 1
#                                 plate_ocr_all_number_dict[all_plate_number] = 1
#                             else:
#                                 if all_plate_number[-5].isalpha():
#                                     plate_ocr_5_char_dict[all_plate_number[-5]] += 1
#                                 plate_ocr_all_number_dict[all_plate_number] += 1
#
#                             if count_24 == 24:
#                                 truck_T_S_F = max(exit_flag, key=exit_flag.get)
#                                 exit_flag["True"] = 0
#                                 exit_flag["Stop"] = 0
#                                 exit_flag["False"] = 0
#                                 # detect_plate_id_dict[plate_id][1] = True
#                                 ocr_all_number = max(plate_ocr_all_number_dict, key=plate_ocr_all_number_dict.get)
#                                 ocr_all_number = mod_area_name(ocr_all_number)
#                                 ocr_last_4_word = max(plate_ocr_4word_dict, key=plate_ocr_4word_dict.get)
#                                 ocr_front_chars = ocr_all_number[:-4]  # 뒤의 문자 4자리를 제외한 앞의 문자열
#
#                                 if len(ocr_list) == 0:
#                                     ocr_list = [ocr_all_number, ocr_front_chars, ocr_last_4_word, plate_id]
#                                     if log_save_s:  # True 일 경우  로그 저장
#                                         log_save(ocr_last_4_word, ocr_all_number, plate_ocr_5_char_dict)
#
#                                     if lp_img_save:
#                                         save_detect_img(crop_plate, ocr_all_number)
#
#                                     print("OCR_last_4 : ", ocr_last_4_word)
#                                     print("OCR_all_number : ", ocr_all_number)
#
#                                 else:  # 리스트 가 비어 있지 않은 경우
#                                     score = lp_decision(ocr_front_chars, ocr_list, ocr_last_4_word, plate_id,
#                                                         ocr_all_number, lp_conf)
#                                     # print("score", score)
#
#                                     # lp_conf 값이 지정 값을 넘으면 동일 번호판 으로 인식
#                                     if score >= lp_conf:
#                                         pass
#                                     else:
#                                         ocr_list = [ocr_all_number, ocr_front_chars, ocr_last_4_word, plate_id]
#                                         if log_save_s:  # True 일 경우  로그 저장
#                                             log_save(ocr_last_4_word, ocr_all_number, plate_ocr_5_char_dict)
#                                         if lp_img_save:
#                                             save_detect_img(crop_plate, ocr_all_number)
#
#                                         print("OCR_last_4 : ", ocr_last_4_word)
#                                         print("OCR_all_number : ", ocr_all_number)
#
#                                 plate_ocr_4word_dict.clear()
#                                 plate_ocr_all_number_dict.clear()
#                                 plate_ocr_5_char_dict.clear()
#                                 plate_detection_tracking_flag = False
#                                 detection_frame_count = 0
#                                 count_24 = 0
#                             else:
#                                 count_24 += 1
#                                 # detect_plate_id_dict[plate_id][0] += 1
#
#             fps_str, prev_time = fps_calculator(rtsp_cur_time, prev_time)
#             window_name = str("210_Video")
#             video_show(rtsp_img, rtsp_infer_img, fps_str, window_name)
#             frame_count += 1
#         else:
#             cv2.destroyAllWindows()
#             break

def input_rtsp2(camera_path, log_save_s, lp_conf, lp_img_save):
    prev_time = 0
    detection_yolo = lp_detection_tracking.lp_detection_tracking()
    detection_yolo.model_init()
    rtsp = cv2.VideoCapture(camera_path)

    """
    plate_detection_flag : yolo 모델에서 번호판이 탐지되었을 경우 알려주는 플래그
    plate_detection_tracking_flag : 설정된 프레임 후 모든 처리과정을 실행시키기 위한 플래그
    detection_frame_count :작은 크기의 번호판이 아닌 온전한 번호판을 탐지하기 위한 프레임 무시 번호, 설정된 프레임 동안 번호판만 탐지하기 위함
    del_id_list : 삭제할 ID 리스트
    detect_plate_id_dict : 탐지된 번호판 딕셔너리 {"id" : [frame count, 24frame chek]}
    plate_ocr_4word_dict : 번호판 마지막 4자리 점수 계산 딕셔너리 {ocr 4word : value}
    plate_ocr_all_number_dict : 번호판 전체 번호 점수 계산 딕셔너리 {ocr word : value}
    plate_ocr_5_char_dict : 뒤에서 다섯번째 '한글'을 나타내기 위한 딕셔너리 {ocr 5char : value}
    """
    plate_detection_tracking_flag = False
    detection_frame_count = 0
    plate_ocr_4word_dict = {}
    plate_ocr_all_number_dict = {}
    plate_ocr_5_char_dict = {}
    ocr_list = []
    count_24 = 0
    truck_count = 0

    while True:
        rtsp_ret, rtsp_img = rtsp.read()
        rtsp_cur_time = time.time()

        if rtsp_ret:
            if detection_frame_count == 30:
                plate_detection_tracking_flag = True

            rtsp_infer_img, plate_detection_flag, plate_id_list, truck_id_list, exit_flag, crop_plate, small_plate_detect = \
                detection_yolo.inference_img2(rtsp_img, plate_detection_tracking_flag=plate_detection_tracking_flag)

            if plate_detection_flag and not small_plate_detect and not plate_detection_tracking_flag:
                detection_frame_count += 1
            else:
                detection_frame_count = 0

            # print("detection_frame_count :", detection_frame_count, ", "
            #       "small_plate_detect :", small_plate_detect, ", "
            #       "plate_detection_tracking_flag :", plate_detection_tracking_flag, ","
            #       "count_24 :", count_24)

            if plate_detection_tracking_flag:
                if len(plate_id_list) > 0:
                    for index, value in enumerate(plate_id_list):
                        plate_id = value[0]
                        ocr_word = value[1]

                        all_plate_number = ocr_word.translate(str.maketrans('', '', string.punctuation))
                        all_plate_number = all_plate_number.replace(" ", "")
                        last_4_word = all_plate_number[-4:]

                        if len(all_plate_number) > 5:
                            if ocr_list and last_4_word == ocr_list[2] and all_plate_number[0] == ocr_list[1][0]:
                                plate_detection_tracking_flag = False
                                detection_frame_count = 0
                                count_24 = 0
                            else:
                                if last_4_word not in plate_ocr_4word_dict:
                                    plate_ocr_4word_dict[last_4_word] = 1
                                else:
                                    plate_ocr_4word_dict[last_4_word] += 1

                                if all_plate_number not in plate_ocr_all_number_dict:
                                    if all_plate_number[-5].isalpha():
                                        plate_ocr_5_char_dict[all_plate_number[-5]] = 1
                                    plate_ocr_all_number_dict[all_plate_number] = 1
                                else:
                                    if all_plate_number[-5].isalpha():
                                        plate_ocr_5_char_dict[all_plate_number[-5]] += 1
                                    plate_ocr_all_number_dict[all_plate_number] += 1

                                if count_24 == 24:
                                    # 추후 트럭의 움직임 유무 확인 시
                                    truck_T_S_F = max(exit_flag, key=exit_flag.get)
                                    exit_flag["True"] = 0
                                    exit_flag["Stop"] = 0
                                    exit_flag["False"] = 0

                                    ocr_all_number = max(plate_ocr_all_number_dict, key=plate_ocr_all_number_dict.get)
                                    ocr_all_number = mod_area_name(ocr_all_number)
                                    ocr_last_4_word = max(plate_ocr_4word_dict, key=plate_ocr_4word_dict.get)
                                    ocr_front_chars = ocr_all_number[:-4]  # 뒤의 문자 4자리를 제외한 앞의 문자열

                                    if len(ocr_list) == 0:
                                        ocr_list = [ocr_all_number, ocr_front_chars, ocr_last_4_word, plate_id]
                                        if log_save_s:
                                            log_save(ocr_last_4_word, ocr_all_number, plate_ocr_5_char_dict)
                                        if lp_img_save:
                                            save_detect_img(crop_plate, ocr_all_number)

                                        truck_count += 1
                                        print("OCR_last_4 : ", ocr_last_4_word)
                                        print("OCR_all_number : ", ocr_all_number)
                                        print("in_truck_count : ", truck_count)
                                    else:
                                        score = lp_decision(ocr_front_chars, ocr_list, ocr_last_4_word, plate_id,
                                                            ocr_all_number, lp_conf)
                                        # lp_conf 값이 지정 값을 넘으면 동일 번호판 으로 인식
                                        if score >= lp_conf:
                                            pass
                                        else:
                                            ocr_list = [ocr_all_number, ocr_front_chars, ocr_last_4_word, plate_id]
                                            if log_save_s:  # True 일 경우  로그 저장
                                                log_save(ocr_last_4_word, ocr_all_number, plate_ocr_5_char_dict)
                                            if lp_img_save:
                                                save_detect_img(crop_plate, ocr_all_number)

                                            truck_count += 1
                                            print("OCR_last_4 : ", ocr_last_4_word)
                                            print("OCR_all_number : ", ocr_all_number)
                                            print("in_truck_count : ", truck_count)
                                    plate_ocr_4word_dict.clear()
                                    plate_ocr_all_number_dict.clear()
                                    plate_ocr_5_char_dict.clear()
                                    plate_detection_tracking_flag = False
                                    detection_frame_count = 0
                                    count_24 = 0
                                else:
                                    count_24 += 1
                        else:
                            plate_detection_tracking_flag = False
                            detection_frame_count = 0
                            count_24 = 0
                else:
                    plate_detection_tracking_flag = False
                    detection_frame_count = 0
                    count_24 = 0

            fps_str, prev_time = fps_calculator(rtsp_cur_time, prev_time)
            window_name = str("210_Video")
            video_show(rtsp_img, rtsp_infer_img, fps_str, window_name)
        else:
            cv2.destroyAllWindows()
            break

def lp_decision(ocr_front_chars, ocr_list, ocr_last_4_word, plate_id, ocr_all_number, lp_conf):
    score = 100  # 초기 점수를 100으로 설정
    # ocr 문자열 이 모두 동일 하면 같은 번호판 으로 판단
    if ocr_all_number == ocr_list[0]:
        score = 100  # ocr_all_number와 ocr_list[0]이 동일한 경우, 점수를 100으로 설정
        return score
    else:
        if ocr_last_4_word != ocr_list[2]:  # 뒤의 4자리가 다른 경우 차감
            for i in range(4):
                if ocr_list[2][i] != ocr_last_4_word[i]:
                    score -= 10

        # 앞의 문자 열이 다른 경우
        if ocr_list[1] != ocr_front_chars:
            # (1) 이전 문자열 길이 가 현재 문자열 보다 같거나 긺
            if len(ocr_list[1]) <= len(ocr_front_chars):
                for i in range(len(ocr_list[1])):
                    o_a_m_slice = ocr_front_chars[:len(ocr_list[1])]
                    if ocr_list[1][i] != o_a_m_slice[i]:
                        score -= 30 // len(o_a_m_slice)
                        if score < lp_conf:
                            return score

            # (2) 현재 문자열 이 이전 문자열 보다 긺
            elif len(ocr_list[1]) > len(ocr_front_chars):
                for i in range(len(ocr_front_chars)):
                    o_l_slice = ocr_list[1][:len(ocr_front_chars)]
                    if o_l_slice[i] != ocr_front_chars[i]:
                        score -= 30 // len(o_l_slice)
                    if score < lp_conf:
                        return score

        return score

        # id 값이 다른 경우
        # score -= 10 if plate_id != ocr_list[3] else 0

    # return score


def mod_area_name(ocr_all_number):
    # 수정할 대상 문자 및 대체할 문자를 딕셔너리로 정의
    error_correction_dict_1 = {'누': '부', '무': '부', '천': '전', '수': '부', '주': '충'}
    error_correction_dict_2 = {'수': '주', '가': '기', '님': '남', '묵': '북'}

    # ocr_all_number의 첫 번째 문자가 수정 대상 문자인 경우, 해당 문자를 대체 문자로 변경
    if ocr_all_number[0] in error_correction_dict_1.keys():
        corrected_char = error_correction_dict_1[ocr_all_number[0]]
        ocr_all_number = corrected_char + ocr_all_number[1:]
        return ocr_all_number

    # ocr_all_number의 두 번째 문자가 수정 대상 문자인 경우, 해당 문자를 대체 문자로 변경
    elif ocr_all_number[1] in error_correction_dict_2.keys():
        corrected_char = error_correction_dict_2[ocr_all_number[1]]
        ocr_all_number = ocr_all_number[0] + corrected_char + ocr_all_number[2:]
        return ocr_all_number

    else:
        return ocr_all_number


def log_save(ocr_last_4_word, ocr_all_number, plate_ocr_5_char_dict):
    with open('ocr_log.txt', 'a') as log_f:
        time_stream = time.strftime('%Y-%m-%d %H:%M:%S')
        log_f.write("[" + time_stream + "]" + " OCR_LAST_4 : " + str(
            ocr_last_4_word) +
                    "\n")
        if len(plate_ocr_5_char_dict) > 0:
            ocr_all_number = ocr_all_number[0:-5] + str(
                max(plate_ocr_5_char_dict,
                    key=plate_ocr_5_char_dict.get)) + ocr_all_number[-4:]
        log_f.write(
            "[" + time_stream + "]" + " OCR_ALL_NUMBER : " + str(
                ocr_all_number) +
            "\n")


def save_detect_img(crop_plate, ocr_all_number):
    # 현재 디렉토리 경로
    root_dir = os.getcwd()

    # detect_plate 디렉토리 경로
    detect_plate_dir = os.path.join(root_dir, 'detect_plate')

    # detect_plate 디렉토리가 없다면 생성
    if not os.path.exists(detect_plate_dir):
        os.makedirs(detect_plate_dir)

    # 오늘 날짜(yyyy-mm-dd)로 폴더 이름 생성
    today = datetime.date.today().strftime('%Y-%m-%d')
    today_dir = os.path.join(detect_plate_dir, today)

    # 오늘 날짜로 된 폴더가 없다면 생성
    if not os.path.exists(today_dir):
        os.mkdir(today_dir)

    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    crop_plate = Image.fromarray(crop_plate.astype(np.uint8))
    crop_plate.save(os.path.join(today_dir, f'{time_str + "_" + ocr_all_number}.jpg'), quality=1)


def check_path(camera_path, log_save_s, lp_conf, lp_img_save):
    if camera_path:
        # input_rtsp(camera_path, log_save_s, lp_conf, lp_img_save)
        input_rtsp2(camera_path, log_save_s, lp_conf, lp_img_save)
    else:
        print("Enter the camera path!")


def input_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--camera', default=None, help='set your camera IP')
    parser.add_argument('--log_save', default=True, help='You can choose log save.')
    parser.add_argument('--conf', type=float, default=60, help='Same license plate accuracy out of 100.')
    parser.add_argument('--img_save', default=False, help='Save the crop image of the detected license plate displayed '
                                                          'in the log')
    args = parser.parse_args()
    return args


def record_video(camera_path):
    rtsp = cv2.VideoCapture(camera_path)
    while True:
        rtsp_ret, rtsp_img = rtsp.read()

        if rtsp_ret:
            resize_original_img = cv2.resize(rtsp_img, (640, 480))
            cv2.imshow("record", resize_original_img)
            cv2.waitKey(1)
        else:
            cv2.destroyAllWindows()
            break

def main():
    args = input_parser()
    camera_path = 'C:/python/dataset/test.mp4'#args.camera
    log_save_s = args.log_save
    lp_conf = args.conf
    lp_img_save = args.img_save
    check_path(camera_path, log_save_s, lp_conf, lp_img_save)

    ### 사용안함 <오류수정중>
    # multi_rtsp = Multi_processing_rtsp_cls()
    # multi_rtsp.multi_processing_rtsp()


if __name__ == '__main__':
    main()
