from PIL import Image
import numpy as np
import cv2
import copy

from .occlusion_detection import OcclusionDetector


def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x+x1)//2, (y+y1)//2
    w, h = x1-x, y1-y
    s = int(max(w, h)//2*expand)
    crop_box = [x_c-s, y_c-s, x_c+s, y_c+s]
    return crop_box, s


def face_seg(image, mode="raw", fp=None):
    """
    对图像进行面部解析，生成面部区域的掩码。

    Args:
        image (PIL.Image): 输入图像。

    Returns:
        PIL.Image: 面部区域的掩码图像。
    """
    seg_image = fp(image, mode=mode)  # 使用 FaceParsing 模型解析面部
    if seg_image is None:
        print("error, no person_segment")  # 如果没有检测到面部，返回错误
        return None

    seg_image = seg_image.resize(image.size)  # 将掩码图像调整为输入图像的大小
    return seg_image


def get_image(image, face, face_box, upper_boundary_ratio=0.5, expand=1.5, mode="raw", fp=None, 
              enable_occlusion_detection=True, occlusion_sensitivity=0.3):
    """
    将裁剪的面部图像粘贴回原始图像，并进行一些处理。

    Args:
        image (numpy.ndarray): 원본 이미지（신체 부분）。
        face (numpy.ndarray): 잘린 얼굴 이미지。
        face_box (tuple): 얼굴 경계상자의 좌표 (x, y, x1, y1)。
        upper_boundary_ratio (float): 얼굴 영역의 보존 비율을 제어하는 데 사용。
        expand (float): 확장 인수, 자르기 상자를 확대하는 데 사용。
        mode: 융합 mask 구축 방식
        fp: FaceParsing 객체
        enable_occlusion_detection: 가림 감지 기능 활성화 여부
        occlusion_sensitivity: 가림 감지 민감도 (0.0-1.0)

    Returns:
        numpy.ndarray: 처리된 이미지。
    """
    # numpy 배열을 PIL 이미지로 변환
    body = Image.fromarray(image[:, :, ::-1])  # 신체 부분 이미지(전체 이미지)
    face = Image.fromarray(face[:, :, ::-1])  # 얼굴 이미지

    x, y, x1, y1 = face_box  # 얼굴 경계상자의 좌표 가져오기
    crop_box, s = get_crop_box(face_box, expand)  # 확장된 자르기 상자 계산
    x_s, y_s, x_e, y_e = crop_box  # 자르기 상자의 좌표
    face_position = (x, y)  # 얼굴이 원본 이미지에서의 위치

    # 신체 이미지에서 확장된 얼굴 영역을 잘라냄（턱에서 경계까지 거리가 있음）
    face_large = body.crop(crop_box)
        
    ori_shape = face_large.size  # 잘린 후 이미지의 원본 크기

    # 잘린 얼굴 영역에 대해 얼굴 파싱을 수행하여 마스크 생성
    mask_image = face_seg(face_large, mode=mode, fp=fp)
    
    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 얼굴 영역의 마스크를 잘라냄
    
    mask_image = Image.new('L', ori_shape, 0)  # 전체가 검은색인 마스크 이미지 생성
    mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 얼굴 마스크를 검은색 이미지에 붙여넣기
    
    # 가림 감지 및 적응적 마스크 적용
    if enable_occlusion_detection:
        try:
            # 가림 감지기 초기화
            occlusion_detector = OcclusionDetector(sensitivity_threshold=occlusion_sensitivity)
            
            # 원본 이미지를 numpy 배열로 변환 (BGR 형식)
            original_image_np = np.array(body)[:, :, ::-1]  # RGB to BGR
            mask_array_initial = np.array(mask_image)
            
            # 턱 영역 바운딩 박스 계산 (확장된 crop_box 기준)
            jaw_bbox = (x - x_s, y - y_s + int((y1 - y) * 0.6), 
                       x1 - x_s, y1 - y_s)  # 턱 영역만 포함하도록 조정
            
            # 가림 감지 수행
            is_occluded, adjusted_mask = occlusion_detector.detect_occlusion_in_jaw_area(
                original_image_np, mask_array_initial, jaw_bbox
            )
            
            if is_occluded:
                print("가림 감지됨 - 스마트 립싱크 마스크 적용")
                
                # 입 영역을 우선 보존하는 스마트 마스크 생성
                smart_mask = occlusion_detector.create_smart_lip_sync_mask(
                    mask_array_initial, 
                    original_image_np[jaw_bbox[1]:jaw_bbox[3], jaw_bbox[0]:jaw_bbox[2]], 
                    jaw_bbox
                )
                
                mask_image = Image.fromarray(smart_mask)
                
                # 디버그용 시각화 저장 (선택사항)
                debug_vis = occlusion_detector.visualize_occlusion_detection(
                    original_image_np, mask_array_initial, smart_mask, jaw_bbox,
                    save_path="./results/debug/smart_occlusion_debug.png"
                )
            else:
                print("가림 감지되지 않음 - 원본 마스크 사용")
            
        except Exception as e:
            print(f"가림 감지 중 오류 발생: {e}")
            # 오류 발생 시 원본 마스크 사용
            pass
    
    # 얼굴 영역의 상반부를 보존（말하기 영역을 제어하는 데 사용）
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)  # 상반부의 경계 계산
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))  # 상반부 마스크 붙여넣기
    
    
    # 마스크에 가우시안 블러 적용, 가장자리를 더 부드럽게 만듦
    # 입 주변 흐림 문제를 해결하기 위해 블러 커널 크기를 줄임
    blur_kernel_size = int(0.04 * ori_shape[0] // 2 * 2) + 1  # 블러 커널 크기를 0.08에서 0.04로 줄임
    # blur_kernel_size가 최소 3이고 홀수인지 확인
    blur_kernel_size = max(3, blur_kernel_size)
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)  # 가우시안 블러
    
    # 더 부드러운 가장자리를 위한 추가 페더링 효과 적용 (페더링 강도 감소)
    feather_size = max(3, blur_kernel_size // 6)  # 페더링 크기를 //4에서 //6으로 줄임
    # feather_size가 홀수인지 확인
    if feather_size % 2 == 0:
        feather_size += 1
    # 디레이션 이터레이션을 줄여서 과도한 확장 방지
    mask_array = cv2.dilate(mask_array, np.ones((feather_size, feather_size), np.uint8), iterations=1)
    # 두 번째 블러의 강도도 줄임
    feather_blur_size = max(3, feather_size // 2)
    if feather_blur_size % 2 == 0:
        feather_blur_size += 1
    mask_array = cv2.GaussianBlur(mask_array, (feather_blur_size, feather_blur_size), 0)
    
    mask_image = Image.fromarray(mask_array)  # 블러 처리된 마스크를 PIL 이미지로 다시 변환
    
    # 잘린 얼굴 이미지를 확장된 얼굴 영역에 다시 붙여넣기
    face_large.paste(face, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    
    body.paste(face_large, crop_box[:2], mask_image)
    
    body = np.array(body)  # PIL 이미지를 numpy 배열로 다시 변환

    return body[:, :, ::-1]  # 처리된 이미지 반환（BGR에서 RGB로 변환）


def get_image_blending(image, face, face_box, mask_array, crop_box):
    body = Image.fromarray(image[:,:,::-1])
    face = Image.fromarray(face[:,:,::-1])

    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box
    face_large = body.crop(crop_box)

    mask_image = Image.fromarray(mask_array)
    mask_image = mask_image.convert("L")
    face_large.paste(face, (x-x_s, y-y_s, x1-x_s, y1-y_s))
    body.paste(face_large, crop_box[:2], mask_image)
    body = np.array(body)
    return body[:,:,::-1]


def get_image_prepare_material(image, face_box, upper_boundary_ratio=0.5, expand=1.5, fp=None, mode="raw"):
    body = Image.fromarray(image[:,:,::-1])

    x, y, x1, y1 = face_box
    #print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size
    if ori_shape[0] == 0 or ori_shape[1] == 0:
        raise ValueError(f"Failed to crop a valid face region. The detected face bounding box might be invalid: {face_box}")

    mask_image = face_seg(face_large, mode=mode, fp=fp)
    mask_small = mask_image.crop((x-x_s, y-y_s, x1-x_s, y1-y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
    # 确保 blur_kernel_size 至少为 3 且为奇数
    blur_kernel_size = max(3, blur_kernel_size)
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    return mask_array, crop_box
