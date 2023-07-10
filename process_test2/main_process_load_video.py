import string
import cv2
import time
import lp_detection_tracking

def fps_calculator(cur_time, prev_time):
    sec = cur_time - prev_time
    prev_time = cur_time

    fps = 1 / (sec)
    fps_str = "FPS : %0.1f" % fps
    return fps_str, prev_time


def input_rtsp(video, frame_q, detect_q,fps_q,object_stop_child_pipe):
    prev_time = 0
    detection_yolo = lp_detection_tracking.lp_detection_tracking()
    detection_yolo.model_init()
    rtsp = cv2.VideoCapture(video)

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
    del_id_list = []
    detect_plate_id_dict = {}
    plate_ocr_4word_dict = {}
    plate_ocr_all_number_dict = {}
    plate_ocr_5_char_dict = {}


    while True:
        rtsp_ret, rtsp_img = rtsp.read()
        rtsp_cur_time = time.time()

        if rtsp_ret:
            """
            딕셔너리를 확인하고 설정된 프레임을 만족한 키 값을 삭제하기 위함
            딕셔너리를 탐지하고 있는동안에는 바로 삭제가 불가능, 리스트를 만들어 아이디 값을 저장하고 저장된 아이디를 이용 딕셔너리 삭제
            """
            if detect_plate_id_dict is not None:
                for key, value in detect_plate_id_dict.items():
                    if value[1] == True:
                        del_id_list.append(key)

            if len(del_id_list) > 0:
                for del_id_key in del_id_list:
                    if del_id_key in detect_plate_id_dict:
                        del detect_plate_id_dict[del_id_key]

            if detection_frame_count == 30:
                plate_detection_tracking_flag = True

            rtsp_infer_img, plate_detection_flag, plate_id_list = \
                detection_yolo.inference_img2(rtsp_img, plate_detection_tracking_flag=plate_detection_tracking_flag)

            if not plate_detection_flag:
                detection_frame_count = 0

            if plate_detection_flag and not plate_detection_tracking_flag:
                detection_frame_count += 1

            if plate_detection_tracking_flag:
                for index, value in enumerate(plate_id_list):
                    plate_id = value[0]
                    ocr_word = value[1]

                    plate_id = 'plate_id_' + str(plate_id)

                    if plate_id in del_id_list:
                        plate_detection_tracking_flag = False
                        detection_frame_count = 0
                        continue
                    else:
                        del_id_list.clear()

                        if len(ocr_word) > 5:
                            all_plate_number = ocr_word.translate(str.maketrans('', '', string.punctuation))
                            all_plate_number = all_plate_number.replace(" ", "")
                            last_4_word = all_plate_number[-4:]

                            # last 4 word score
                            if last_4_word not in plate_ocr_4word_dict:
                                plate_ocr_4word_dict[last_4_word] = 1
                            else:
                                plate_ocr_4word_dict[last_4_word] += 1

                            # all number score, 5th korean check
                            if all_plate_number not in plate_ocr_all_number_dict:
                                if all_plate_number[-5].isalpha():
                                    plate_ocr_5_char_dict[all_plate_number[-5]] = 1
                                plate_ocr_all_number_dict[all_plate_number] = 1
                            else:
                                if all_plate_number[-5].isalpha():
                                    plate_ocr_5_char_dict[all_plate_number[-5]] += 1
                                plate_ocr_all_number_dict[all_plate_number] += 1

                            if plate_id not in detect_plate_id_dict:
                                detect_plate_id_dict[plate_id] = [1, False]
                            else:
                                if detect_plate_id_dict[plate_id][0] == 24:
                                    detect_plate_id_dict[plate_id][1] = True

                                    ocr_last_4_word = max(plate_ocr_4word_dict, key=plate_ocr_4word_dict.get)
                                    with open('ocr_log.txt', 'a') as log_f:
                                        time_stream = time.strftime('%Y-%m-%d %H:%M:%S')
                                        log_f.write("[" + time_stream + "]" + " OCR_LAST_4 : " + str(ocr_last_4_word) +
                                                    "\n")
                                        print("OCR_last_4 : ", ocr_last_4_word)

                                        ocr_all_number = max(plate_ocr_all_number_dict, key=plate_ocr_all_number_dict.get)

                                        if len(plate_ocr_5_char_dict) > 0:
                                            ocr_all_number = ocr_all_number[0:-5] + str(max(plate_ocr_5_char_dict,
                                                                     key=plate_ocr_5_char_dict.get)) + ocr_all_number[
                                                                                                       -4:]
                                        print(f"{video[-8:]} : ", ocr_all_number)
                                        log_f.write(
                                            "[" + time_stream + "]" + " OCR_ALL_NUMBER : " + str(ocr_all_number) +
                                            "\n")
                                    plate_ocr_4word_dict.clear()
                                    plate_ocr_all_number_dict.clear()
                                    plate_ocr_5_char_dict.clear()
                                    plate_detection_tracking_flag = False
                                    detection_frame_count = 0
                                else:
                                    detect_plate_id_dict[plate_id][0] += 1

            # frame, detect, fps Queue로 보냄

            fps_str, prev_time = fps_calculator(rtsp_cur_time, prev_time)
            frame_q.put(rtsp_img)
            detect_q.put(rtsp_infer_img)
            fps_q.put(fps_str)

            # if frame_q.qsize() > 64:
            #     time.sleep(3)


        else:
            # 탐지 완료시 pipe로 보냄

            object_stop_child_pipe.send('end')

            # object_stop_child_pipe.close()
            # # print('send')
            # rtsp.release()
            # frame_q.close()
            # detect_q.close()
            # fps_q.close()
            # break

