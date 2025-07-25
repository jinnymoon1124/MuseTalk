import torch
import time
import os
import cv2
import numpy as np
from PIL import Image
from .model import BiSeNet  # 얼굴 영역을 분할하는 모델
import torchvision.transforms as transforms

class FaceParsing():
    def __init__(self, left_cheek_width=80, right_cheek_width=80):
        # 얼굴 분할 모델 초기화
        self.net = self.model_init()
        
        # 입력 이미지를 모델에 맞게 전처리하는 함수 정의
        self.preprocess = self.image_preprocess()
        
        # 커널(kernel): 이미지에서 특정 영역을 강조하거나 확장할 때 사용하는 필터
        # 여기서는 턱 부분을 강조하는 필터를 직접 정의함
        cone_height = 21  # 위쪽 뾰족한 부분 높이
        tail_height = 12  # 아래 직사각형 꼬리 부분 높이
        total_size = cone_height + tail_height  # 전체 커널 크기
        
        # 전체 커널을 0으로 초기화
        kernel = np.zeros((total_size, total_size), dtype=np.uint8)
        center_x = total_size // 2  # 커널의 중심 위치 계산

        # 위쪽 삼각형(뾰족한 부분) 만들기
        for row in range(cone_height):
            if row < cone_height//2:
                continue  # 윗부분은 비우고 아랫부분만 사용
            width = int(2 * (row - cone_height//2) + 1)  # 점점 넓어지는 삼각형 너비
            start = int(center_x - (width // 2))
            end = int(center_x + (width // 2) + 1)
            kernel[row, start:end] = 1  # 해당 위치에 1을 넣어 영역 표시

        # 아래 직사각형 부분 만들기
        if cone_height > 0:
            base_width = int(kernel[cone_height-1].sum())  # 삼각형 맨 아래 폭과 동일하게 설정
        else:
            base_width = 1
        for row in range(cone_height, total_size):
            start = max(0, int(center_x - (base_width//2)))
            end = min(total_size, int(center_x + (base_width//2) + 1))
            kernel[row, start:end] = 1

        self.kernel = kernel  # 만들어진 턱 강조 커널 저장

        # 뺨 영역을 다듬을 때 사용할 납작한 타원형 커널 (좌우로 넓고 위아래로 얇음)
        self.cheek_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (35, 3))

        # 뺨 영역 마스크 (턱 쪽 영역이 지워지지 않도록 보호하기 위해 사용)
        self.cheek_mask = self._create_cheek_mask(
            left_cheek_width=left_cheek_width,
            right_cheek_width=right_cheek_width
        )

    def _create_cheek_mask(self, left_cheek_width=80, right_cheek_width=80):
        """
        뺨 영역을 마스크로 만들어 특정 영역만 처리 가능하도록 함.
        예: 이미지의 양 끝부분(왼쪽, 오른쪽) 1/4쯤 되는 부분을 뺨으로 지정.
        """
        mask = np.zeros((512, 512), dtype=np.uint8)
        center = 512 // 2
        cv2.rectangle(mask, (0, 0), (center - left_cheek_width, 512), 255, -1)  # 왼쪽 뺨
        cv2.rectangle(mask, (center + right_cheek_width, 0), (512, 512), 255, -1)  # 오른쪽 뺨
        return mask

    def model_init(self, 
                   resnet_path='./models/face-parse-bisent/resnet18-5c106cde.pth', 
                   model_pth='./models/face-parse-bisent/79999_iter.pth'):
        """
        얼굴 분할 모델 로드 및 초기화
        BiSeNet: 얼굴을 여러 부분으로 나누는 신경망
        """
        net = BiSeNet(resnet_path)
        if torch.cuda.is_available():
            net.cuda()  # GPU 사용 가능하면 GPU로 모델 이동
            net.load_state_dict(torch.load(model_pth)) 
        else:
            net.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu')))
        net.eval()  # 추론 모드로 전환 (학습 X)
        return net

    def image_preprocess(self):
        """
        이미지를 모델 입력에 맞게 변환하는 전처리 과정 정의
        - 텐서로 변환하고
        - 평균과 표준편차로 정규화
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, image, size=(512, 512), mode="raw"):
        """
        이미지를 받아 얼굴 분할 실행
        mode:
        - "raw": 얼굴 전체 영역만 남김
        - "jaw": 턱 강조 (턱 영역 확장 후 뺨 제외)
        - "neck": 목 포함
        """
        if isinstance(image, str):
            image = Image.open(image)  # 경로로 들어오면 이미지 열기

        width, height = image.size

        with torch.no_grad():  # 모델 추론 시, 메모리 절약을 위해 Gradient 저장 X
            image = image.resize(size, Image.BILINEAR)  # 이미지 크기 맞추기
            img = self.preprocess(image)  # 전처리
            img = torch.unsqueeze(img, 0)  # 배치 차원 추가 (1개 이미지)
            if torch.cuda.is_available():
                img = img.cuda()

            out = self.net(img)[0]  # 모델 실행 → 결과 얻기
            parsing = out.squeeze(0).cpu().numpy().argmax(0)  # 가장 확률 높은 클래스 선택 (픽셀마다)

            # 모드별 처리
            if mode == "neck":
                # 얼굴(1), 귀(11,12), 목(13,14)만 남기고 나머지 제거
                parsing[np.isin(parsing, [1, 11, 12, 13, 14])] = 255
                parsing[np.where(parsing != 255)] = 0
            elif mode == "jaw":
                # 턱 중심 영역 확장
                face_region = np.isin(parsing, [1]) * 255
                face_region = face_region.astype(np.uint8)
                
                # 얼굴 외곽 확장
                original_dilated = cv2.dilate(face_region, self.kernel, iterations=1)
                eroded = cv2.erode(original_dilated, self.cheek_kernel, iterations=2)

                # 뺨 영역만 보존되도록 마스크 처리
                face_region = cv2.bitwise_and(eroded, self.cheek_mask)
                face_region = cv2.bitwise_or(face_region, cv2.bitwise_and(original_dilated, ~self.cheek_mask))

                # 코(10) 제외한 나머지를 포함
                parsing[(face_region == 255) & (~np.isin(parsing, [10]))] = 255
                parsing[np.isin(parsing, [11, 12, 13])] = 255  # 귀, 목 등 추가
                parsing[np.where(parsing != 255)] = 0  # 나머지는 0으로 설정
            else:
                # 기본: 얼굴(1), 귀(11,12), 목(13)만 남기기
                parsing[np.isin(parsing, [1, 11, 12, 13])] = 255
                parsing[np.where(parsing != 255)] = 0

        # 결과를 이미지로 변환하여 반환
        parsing = Image.fromarray(parsing.astype(np.uint8))
        return parsing

# 프로그램 직접 실행 시 동작
if __name__ == "__main__":
    fp = FaceParsing()  # 클래스 초기화
    segmap = fp('154_small.png')  # 얼굴 분할 실행
    segmap.save('res.png')  # 결과 이미지 저장
