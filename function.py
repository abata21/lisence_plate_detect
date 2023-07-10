# -*- coding:utf-8 -*-
import itertools
import random
from copy import copy
from random import randint

import cv2
import numpy as np
from PIL import Image


def check_class_bright_control(crop_plate):
    cp_gray = cv2.cvtColor(crop_plate, cv2.COLOR_BGR2GRAY)  # 회색조로 변경하고 밝기 측정
    # 밝기 조절
    brightness = int(np.mean(cp_gray))
    permit_bright = 130  # 밝기의 범위는 0 ~ 255임 평균인 130을 기준으로 정해둠

    r = permit_bright - brightness
    if r > 0:  # 양수
        cp = cv2.add(cp_gray, r)
    else:  # 음수 ,0
        cp = cv2.subtract(cp_gray, abs(r))

    #  대비
    alpha = 2.0
    control_cp = np.clip((1 + alpha) * cp - 128 * alpha, 0, 255).astype(np.uint8)

    return control_cp


def threshold_blur(cp):  # 가우시안 블러 & 쓰레딩하여 컨투어가 잘 되도록 함
    # 기존에는 가우시안 블러를 진행 한 이미지를 OCR 인식에 진행했으나
    # 가우시안 블러는 Rotate를 정확하게 하기위해 진행 할 뿐 최종 이미지로 사용 되지 않음

    img_blurred = cv2.GaussianBlur(cp, ksize=(5, 5), sigmaX=0)

    img_blur_thresh = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=19,
        C=9
    )
    return img_blur_thresh


def find_contour(img_blur_thresh):  # 컨투어 찾기

    contours, _ = cv2.findContours(
        img_blur_thresh,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    contours_dict = []  # 컨투어와 컨투어를 감싸는 사각형에 대한 정보를 저장
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        contours_dict.append({  # 리스트안에 딕셔너리 추가
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })

    return contours_dict


def find_lp_chars(contours_dict, cpw, cph):  # 번호판 문자 찾기 (기본)
    possible_contours = []
    cnt = 0
    for c_d in contours_dict:
        area = c_d['w'] * c_d['h']  # 넓이
        ratio = c_d['w'] / c_d['h']  # 비율
        plate_area = cpw * cph
        #  컨투어 박스 면적이 0보단 크고 번호판 전체 넓이의 7분의 1보단 작음 / 길이가 너비 보다 같거나 커야 됨
        if 0 < area < plate_area // 7 and 0.3 < ratio < 1:
            c_d['idx'] = cnt
            cnt += 1
            possible_contours.append(c_d)

    matched_result = []  # 컨투어 박스 안에 컨투어 박스가 있는건 제외하고 담기
    for p_c in range(len(possible_contours)):
        inside_box = False
        for j in range(len(possible_contours)):
            if p_c != j and (possible_contours[p_c]['x'] > possible_contours[j]['x']) and (
                    possible_contours[p_c]['y'] > possible_contours[j]['y']) and (
                    possible_contours[p_c]['w'] < possible_contours[j]['w']) and (
                    possible_contours[p_c]['h'] < possible_contours[j]['h']) and (
                    possible_contours[p_c]['y'] + possible_contours[p_c]['h'] <
                    possible_contours[j]['y'] + possible_contours[j]['h']
            ):
                inside_box = True
                break
        if not inside_box:
            matched_result.append(possible_contours[p_c])

    #  가장 긴 컨투어 박스의 4/5 크기 이상 컨투어 박스만 담기
    highest_char_h = sorted(matched_result, key=lambda x: x['h'])
    matched_chars = []
    for mc in matched_result:
        if mc['h'] > (highest_char_h[-1]['h'] * 4 // 5):
            matched_chars.append(mc)

    return matched_chars


def distortion_correction(mask, im0):
    mask_np = mask.cpu().numpy()[0]  # GPU 상의 Tensor를 CPU 상의 NumPy 배열로 변환
    points_list = cv2.findNonZero((mask_np * 255).astype(np.uint8)).tolist()  # 좌표 리스트 반환
    points_list = [coord[0] for coord in points_list]

    sum_point_list = []
    diff_point_list = []
    for points in points_list:
        sum_pont = sum(points)
        diff_point = points[1] - points[0]
        sum_point_list.append(sum_pont)
        diff_point_list.append(diff_point)

    max_sum_index = sum_point_list.index(max(sum_point_list))
    min_sum_index = sum_point_list.index(min(sum_point_list))
    max_diff_index = diff_point_list.index(max(diff_point_list))
    min_diff_index = diff_point_list.index(min(diff_point_list))

    # x+y 최소 값이 좌하단//좌상단
    topLeft = points_list[min_sum_index]

    # x+y 최대 값이 우상단 좌표 // 우하단
    bottomRight = points_list[max_sum_index]

    # y-x 최소 값이 우하단 좌표  //우상단
    topRight = points_list[min_diff_index]

    # y-x 최대 값이 좌상단 좌표 //좌하단
    bottomLeft = points_list[max_diff_index]

    # 변환 전 번호판 4개 좌표
    origin_plate_point = np.float32([topLeft, topRight, bottomRight, bottomLeft])

    # 변환 후 번호판의 폭과 높이 계산
    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])

    width = max([w1, w2])  # 두 좌우 거리 간의 최대 값이 번호판 의 폭
    height = max([h1, h2])  # 두 상하 거리 간의 최대 값이 번호판 의 높이

    # 변환 후 번호판 4개 좌표
    change_plate_point = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

    # 변환 행렬 계산
    mtrx = cv2.getPerspectiveTransform(origin_plate_point, change_plate_point)

    # 원근 변환 적용
    img_distorted = cv2.warpPerspective(im0, mtrx, (width, height))

    return img_distorted


def rotation_correction(matched_result, cp, cpw, cph):
    PLATE_HEIGHT_PADDING = 1.1
    sorted_x_chars = sorted(matched_result, key=lambda x: x['x'])

    rx_list = []  # 우상단 x좌표 값을 담을 리스트
    for x_c in sorted_x_chars:
        rx_list.append(x_c['x'] + x_c['w'])  # 우상단 좌표 x값을 리스트에 담기
    index_right = rx_list.index(max(rx_list))  # 가장 우측의 x값을 가진 인덱스

    plate_cx = (sorted_x_chars[0]['cx'] + sorted_x_chars[index_right]['cx']) // 2  # 번호판 중앙 x 좌표
    plate_cy = (sorted_x_chars[0]['cy'] + sorted_x_chars[index_right]['cy']) // 2  # 번호판 중심 y 좌표

    plate_width = (sorted_x_chars[index_right]['x'] + sorted_x_chars[index_right]['w'] - sorted_x_chars[0][
        'x'])  # 번호판 너비 계산

    sum_height = 0  # 모든 컨투어 박스의 높이를 더함
    for s in sorted_x_chars:
        sum_height += s['h']
    plate_height = int(sum_height / len(sorted_x_chars)) * PLATE_HEIGHT_PADDING  # 평균 높이 계산 하고 오차 범위 줌

    if len(matched_result) > 1:
        # 번호판의 기울어진 각도 구하기
        triangle_height = sorted_x_chars[index_right]['cy'] - sorted_x_chars[0]['cy']  # 삼각함수 사용
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_x_chars[0]['cx'], sorted_x_chars[0]['cy']]) -
            np.array([sorted_x_chars[index_right]['cx'], sorted_x_chars[index_right]['cy']])
        )

        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle,
                                                  scale=1.0)
        img_rotated = cv2.warpAffine(cp, M=rotation_matrix,
                                     dsize=(cpw, cph))

        return img_rotated, plate_cy, plate_height
    else:
        return cp, plate_cy, plate_height


def classes_division(matched_result, img_distorted, cpw, cph, c):  # 번호판 종류 구분
    img_rotated, plate_cy, plate_height = rotation_correction(matched_result, img_distorted, cpw, cph)

    if c == 3 or c == 5 or c == 6:  # (2줄 번호판 종류2번)
        top_crop_img, under_crop_img = crop_composite(img_rotated, plate_cy, plate_height, c)
        attach_width_crop_img = cv2.hconcat([top_crop_img, under_crop_img])

        return attach_width_crop_img

    elif c == 2:  # 노랑 1줄
        irh, irw = img_rotated.shape
        img_cropped = img_rotated[0: irh, int(irw * 0.17): int(irw * 0.95)]
        attach_stuck_ko_img = find_stuck_ko(img_cropped, c)[1]
        vertical_composite_img = vertical_orientation_composite(matched_result, img_rotated, img_cropped)
        attach_img = cv2.hconcat([vertical_composite_img, attach_stuck_ko_img])

        return attach_img

    elif c == 4:  # 하양 1줄
        attach_stuck_ko_img = find_stuck_ko(img_rotated, c)[1]

        return attach_stuck_ko_img

    else:  # 주황, 초록 new
        top_crop_img, under_crop_img = crop_composite(img_rotated, plate_cy, plate_height, c)
        attach_stuck_ko_img = find_stuck_ko(under_crop_img, c)[0]
        attach_width_crop_img = cv2.hconcat([top_crop_img, attach_stuck_ko_img])

        return attach_width_crop_img


def crop_composite(img_rotated, under_plate_cy, under_plate_h, c):  # 2줄 번호판 윗줄 아랫 줄 crop , 합성
    irh, irw = img_rotated.shape
    plate_cx = irw // 2

    if c == 3 or c == 6:  # yellow 2line 과 green top_word는 윗줄 + 아랫줄 작업만 진행.
        top_crop_img = img_rotated[0: int(irh * 6 / 17), int(irw * 0.2): int(irw * 0.8)]  # 윗문자 crop
        under_crop_img = img_rotated[int(irh * 6/ 17): int(irh * 0.95), int(irw * 0.02): int(irw * 0.98)]

        top_crop_h, top_crop_w = top_crop_img.shape
        under_crop_h, under_crop_w = under_crop_img.shape  # 아래 crop한 부분의 세로 가로 값
        under_crop_mod_img = cv2.resize(under_crop_img, (int(under_crop_w * 2), under_crop_h))  # 가로:세로 = 1 : 2

        under_crop_mod_h, under_crop_mod_w = under_crop_mod_img.shape
        top_crop_img = cv2.resize(top_crop_img,
                                  (top_crop_w * under_crop_mod_h // top_crop_h, under_crop_mod_h))  # 윗줄을 아래줄 사이즈로 맞추기

        return top_crop_img, under_crop_mod_img

    else:
        top_crop_left = img_rotated[0: irh // 2, int(irw * 0.2): int(irw * 0.44)]
        top_crop_right = img_rotated[0: irh // 2, int(irw * 0.6): int(irw * 0.8)]
        under_crop_img = img_rotated[irh // 2: int(irh), int(irw * 0.02): int(irw * 0.98)]
        under_crop_h, under_crop_w = under_crop_img.shape

        composite_crop_img = cv2.hconcat([top_crop_left, top_crop_right])  # 윗글자 왼쪽 오른쪽 합성
        composite_h, composite_w = composite_crop_img.shape

        composite_crop_img = cv2.resize(composite_crop_img, (composite_w * under_crop_h // composite_h, under_crop_h))

        return composite_crop_img, under_crop_img


def vertical_orientation_composite(matched_chars, img_rotated, img_cropped):  # 세로 쓰기 되어 있는 한글 crop, 합성
    sorted_x_chars = sorted(matched_chars, key=lambda x: x['x'])
    h = sorted_x_chars[0]['h']  # 맨 앞 컨투어 의 h 값
    y = sorted_x_chars[0]['y']  # 맨 앞 컨투어 의 y 값

    irh, irw = img_rotated.shape

    # 로테이트 된 이미지 가져옴
    w = irw * 0.12
    vertical_crop_img = cv2.getRectSubPix(  # 윗글자 오른쪽 글자 crop
        img_rotated,
        patchSize=(int(w), int(h * 1.2)),
        center=(int((irw // 30) + (w // 2)), int((y + h) - (h // 2))))

    vertical_crop_h, vertical_crop_w = vertical_crop_img.shape
    first_ko = vertical_crop_img[0: int(h * 1.1 // 2), 0: int(vertical_crop_w)]  # 세로쓰기 한글 윗 부분

    first_ko_h, first_ko_w = first_ko.shape
    last_ko = vertical_crop_img[int(first_ko_h): int(vertical_crop_h), 0: int(first_ko_w)]  # 세로쓰기 한글 아랫 부분

    ich, icw = img_cropped.shape
    last_ko_h, last_ko_w = last_ko.shape

    if first_ko_h == 0:
        # 분모가 0인 경우에 대한 예외 처리
        return img_cropped

    else:
        front_img = cv2.resize(first_ko, (int(first_ko_w * ich / first_ko_h), ich))
        behind_img = cv2.resize(last_ko, (int(last_ko_w * ich / last_ko_h), ich))
        vertical_composite_img = cv2.hconcat([front_img, behind_img])

        return vertical_composite_img



def find_stuck_ko(crop_img_rotated, c):  # 번호판 사이 한글이 끼여 있는 경우
    cirh, cirw = crop_img_rotated.shape
    one_chars_w = cirw * 0.12

    if c == 0 or c == 1:
        stuck_ko_img = crop_img_rotated[0:cirh, int(cirw * 0.28): int(cirw * 0.42)]
        stuck_ko_img = cv2.resize(stuck_ko_img, (int(one_chars_w * 2.5), cirh))

        front_crop = cv2.getRectSubPix(  # 앞 문자 crop
            crop_img_rotated,
            patchSize=(int(cirw * 0.25), int(cirh)),
            center=(int(cirw * 0.15), int(cirh // 2))
        )

        back_crop = cv2.getRectSubPix(  # 뒷 문자 crop
            crop_img_rotated,
            patchSize=(int(cirw * 0.55), int(cirh)),
            center=(int(cirw * 0.7), int(cirh // 2))
        )
    elif c == 2:
        stuck_ko_img = crop_img_rotated[0:cirh, int(cirw * 0.28): int(cirw * 0.44)]
        stuck_ko_img = cv2.resize(stuck_ko_img, (int(one_chars_w * 4), cirh))
        front_crop = crop_img_rotated[0:cirh, 0: int(cirw * 0.28)]
        back_crop = crop_img_rotated[0:cirh, int(cirw * 0.44): cirw]

    else:  # 하양 1줄
        stuck_ko_img = crop_img_rotated[0:cirh, int(cirw * 0.32): int(cirw * 0.42)]
        stuck_ko_img = cv2.resize(stuck_ko_img, (int(one_chars_w * 2.5), cirh))

        front_crop = crop_img_rotated[0:cirh, 0: int(cirw * 0.32)]
        back_crop = crop_img_rotated[0:cirh, int(cirw * 0.42): cirw]

    front_crop_o = cv2.resize(front_crop, (int(front_crop.shape[1] * 1.5), cirh))
    back_crop_o = cv2.resize(back_crop, (int(back_crop.shape[1] * 1.5), cirh))

    front_crop_mod = cv2.resize(front_crop, (int(front_crop.shape[1] * 2), cirh))
    back_crop_mod = cv2.resize(back_crop, (int(back_crop.shape[1] * 2), cirh))

    attach_stuck_ko_img = cv2.hconcat([front_crop_o, stuck_ko_img, back_crop_o])
    attach_stuck_ko_mod_img = cv2.hconcat([front_crop_mod, stuck_ko_img, back_crop_mod])

    return attach_stuck_ko_img, attach_stuck_ko_mod_img

