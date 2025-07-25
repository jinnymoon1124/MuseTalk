import sys
from face_detection import FaceAlignment,LandmarksType
from os import listdir, path
import subprocess
import numpy as np
import cv2
import pickle
import os
import json
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
import torch
from tqdm import tqdm

"""
얼굴 감지 및 랜드마크 추출 모듈

이 모듈은 립싱크 시스템의 핵심 전처리 단계입니다:
1. 비디오의 각 프레임에서 얼굴을 찾습니다
2. 얼굴의 주요 지점들(눈, 코, 입, 턱 등)을 정확히 찾습니다
3. 립싱크를 적용할 얼굴 영역을 정확히 계산합니다

이 과정이 정확해야 AI가 입술 움직임을 자연스럽게 생성할 수 있습니다.
"""

# ===== AI 모델 초기화 =====

# GPU 사용 가능 여부 확인 (GPU가 있으면 더 빠르게 처리)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. DWPose 모델 초기화 - 사람의 전체 몸과 얼굴 랜드마크를 찾는 AI 모델
# 이 모델은 얼굴의 68개 주요 지점을 정확히 찾아줍니다
config_file = './musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'
model = init_model(config_file, checkpoint_file, device=device)

# 2. Face Detection 모델 초기화 - 얼굴 영역을 찾는 AI 모델
# 이 모델은 이미지에서 얼굴이 어디에 있는지 사각형으로 표시해줍니다
device = "cuda" if torch.cuda.is_available() else "cpu"
fa = FaceAlignment(LandmarksType._2D, flip_input=False,device=device)

# 얼굴을 찾지 못했을 때 사용할 기본값 (0,0,0,0)
coord_placeholder = (0.0,0.0,0.0,0.0)

def resize_landmark(landmark, w, h, new_w, new_h):
    """
    랜드마크 좌표를 이미지 크기에 맞게 조정하는 함수
    
    랜드마크: 얼굴의 주요 지점들 (눈, 코, 입 등)의 좌표
    이미지 크기가 바뀌면 랜드마크 좌표도 비례적으로 조정해야 합니다
    
    Args:
        landmark: 원본 랜드마크 좌표
        w, h: 원본 이미지의 너비와 높이
        new_w, new_h: 새로운 이미지의 너비와 높이
    
    Returns:
        조정된 랜드마크 좌표
    """
    w_ratio = new_w / w  # 너비 비율
    h_ratio = new_h / h  # 높이 비율
    landmark_norm = landmark / [w, h]  # 좌표를 0~1 사이로 정규화
    landmark_resized = landmark_norm * [new_w, new_h]  # 새로운 크기에 맞게 조정
    return landmark_resized

def read_imgs(img_list):
    """
    이미지 파일들을 읽어서 메모리에 로드하는 함수
    
    비디오는 여러 장의 이미지(프레임)로 구성되어 있습니다.
    이 함수는 각 프레임을 순서대로 읽어서 처리할 수 있도록 준비합니다.
    
    Args:
        img_list: 이미지 파일 경로들의 리스트
    
    Returns:
        frames: 읽어들인 이미지들의 리스트
    """
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):  # tqdm은 진행률을 보여주는 도구
        frame = cv2.imread(img_path)  # OpenCV를 사용하여 이미지 읽기
        frames.append(frame)
    return frames

def get_bbox_range(img_list, upperbondrange=0):
    """
    얼굴 영역의 조정 범위를 계산하는 함수
    
    이 함수는 사용자가 얼굴 영역을 얼마나 조정할 수 있는지 알려줍니다.
    강사가 고개를 돌리거나 움직일 때 얼굴 영역을 적절히 조정하기 위해 사용됩니다.
    
    Args:
        img_list: 이미지 파일 경로들의 리스트
        upperbondrange: 얼굴 영역을 위아래로 조정할 픽셀 수
    
    Returns:
        조정 가능한 범위 정보를 담은 텍스트
    """
    frames = read_imgs(img_list)  # 이미지들을 읽어오기
    batch_size_fa = 1  # 한 번에 처리할 이미지 수 (메모리 효율성을 위해 1개씩)
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    
    average_range_minus = []  # 위쪽 조정 범위들의 평균
    average_range_plus = []   # 아래쪽 조정 범위들의 평균
    
    for fb in tqdm(batches):
        # 1. DWPose 모델을 사용하여 얼굴 랜드마크 추출
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        
        # 2. 얼굴 랜드마크 추출 (68개 지점 중 23~91번이 얼굴 관련)
        face_land_mark = keypoints[0][23:91]  # 얼굴의 68개 주요 지점
        face_land_mark = face_land_mark.astype(np.int32)
        
        # 3. 얼굴 감지 모델을 사용하여 얼굴 영역 찾기
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        # 4. 각 얼굴에 대해 조정 범위 계산
        for j, f in enumerate(bbox):
            if f is None:  # 얼굴을 찾지 못한 경우
                coords_list += [coord_placeholder]
                continue
            
            # 5. 얼굴의 중심점과 조정 범위 계산
            # 랜드마크 29번: 얼굴의 중심점 (코 근처)
            half_face_coord = face_land_mark[29]
            
            # 위쪽 조정 범위: 랜드마크 30번과 29번의 거리
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            # 아래쪽 조정 범위: 랜드마크 29번과 28번의 거리
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            
            # 6. 사용자가 지정한 조정값 적용
            if upperbondrange != 0:
                # 양수: 아래쪽으로 조정, 음수: 위쪽으로 조정
                half_face_coord[1] = upperbondrange + half_face_coord[1]

    # 7. 조정 가능한 범위 정보 반환
    text_range = f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}"
    return text_range

def get_landmark_and_bbox(img_list, upperbondrange=0):
    """
    얼굴 랜드마크와 바운딩박스를 추출하는 메인 함수
    
    이 함수는 립싱크 시스템의 핵심입니다:
    1. 각 프레임에서 얼굴을 찾습니다
    2. 얼굴의 68개 주요 지점을 정확히 찾습니다
    3. 립싱크를 적용할 정확한 얼굴 영역을 계산합니다
    4. 강사의 움직임에 따라 얼굴 영역을 적절히 조정합니다
    
    Args:
        img_list: 이미지 파일 경로들의 리스트
        upperbondrange: 얼굴 영역을 위아래로 조정할 픽셀 수
                       양수: 아래쪽으로 조정 (턱 움직임을 더 포함)
                       음수: 위쪽으로 조정 (이마 부분을 더 포함)
    
    Returns:
        coords_list: 각 프레임의 얼굴 영역 좌표들
        frames: 읽어들인 이미지들
    """
    frames = read_imgs(img_list)  # 이미지들을 읽어오기
    batch_size_fa = 1  # 한 번에 처리할 이미지 수
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    
    average_range_minus = []
    average_range_plus = []
    
    for fb in tqdm(batches):
        # 1. DWPose 모델을 사용하여 얼굴 랜드마크 추출
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        
        # 2. 얼굴 랜드마크 추출 (68개 지점)
        face_land_mark = keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)
        
        # 3. 얼굴 감지 모델을 사용하여 얼굴 영역 찾기
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        # 4. 각 얼굴에 대해 정확한 영역 계산
        for j, f in enumerate(bbox):
            if f is None:  # 얼굴을 찾지 못한 경우
                coords_list += [coord_placeholder]
                continue
            
            # 5. 얼굴의 중심점과 조정 범위 계산
            half_face_coord = face_land_mark[29]  # 얼굴 중심점
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]  # 위쪽 범위
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]   # 아래쪽 범위
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            
            # 6. 사용자가 지정한 조정값 적용
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange + half_face_coord[1]
            
            # 7. 얼굴 영역의 정확한 경계 계산
            half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
            min_upper_bond = 0
            upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)
            
            # 8. 최종 얼굴 영역 좌표 계산
            # (왼쪽, 위쪽, 오른쪽, 아래쪽) 순서
            f_landmark = (np.min(face_land_mark[:, 0]), int(upper_bond), np.max(face_land_mark[:, 0]), np.max(face_land_mark[:, 1]))
            x1, y1, x2, y2 = f_landmark
            
            # 9. 계산된 영역이 유효한지 확인
            if y2-y1 <= 0 or x2-x1 <= 0 or x1 < 0:  # 잘못된 영역인 경우
                coords_list += [f]  # 원본 얼굴 감지 결과 사용
                w, h = f[2]-f[0], f[3]-f[1]
                print("error bbox:", f)
            else:
                coords_list += [f_landmark]  # 계산된 정확한 영역 사용
    
    # 10. 조정 가능한 범위 정보 출력
    print("********************************************bbox_shift parameter adjustment**********************************************************")
    print(f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}")
    print("*************************************************************************************************************************************")
    return coords_list, frames

# ===== 테스트 코드 =====
if __name__ == "__main__":
    # 테스트용 이미지 리스트
    img_list = ["./results/lyria/00000.png", "./results/lyria/00001.png", "./results/lyria/00002.png", "./results/lyria/00003.png"]
    crop_coord_path = "./coord_face.pkl"
    
    # 얼굴 랜드마크와 바운딩박스 추출
    coords_list, full_frames = get_landmark_and_bbox(img_list)
    
    # 결과를 파일로 저장 (다음 실행 시 재사용 가능)
    with open(crop_coord_path, 'wb') as f:
        pickle.dump(coords_list, f)
    
    # 추출된 얼굴 영역 확인
    for bbox, frame in zip(coords_list, full_frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]  # 얼굴 영역만 잘라내기
        print('Cropped shape', crop_frame.shape)
    
    print(coords_list)
