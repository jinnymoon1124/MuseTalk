import cv2
import numpy as np
from PIL import Image
import torch
from typing import Tuple, Optional

class OcclusionDetector:
    """
    얼굴 가림 현상을 감지하고 적응적 마스크를 생성하는 클래스
    마이크, 손, 기타 물체에 의한 얼굴 가림을 감지하여 자연스러운 립싱크 구현
    """
    
    def __init__(self, sensitivity_threshold: float = 0.3):
        """
        초기화
        
        Args:
            sensitivity_threshold: 가림 감지 민감도 임계값 (0.0-1.0)
        """
        self.sensitivity_threshold = sensitivity_threshold
        
    def detect_occlusion_in_jaw_area(self, original_image: np.ndarray, 
                                   face_parsing_mask: np.ndarray,
                                   jaw_bbox: Tuple[int, int, int, int]) -> Tuple[bool, np.ndarray]:
        """
        턱/입 영역에서 가림 현상을 감지
        
        Args:
            original_image: 원본 이미지 (BGR)
            face_parsing_mask: 얼굴 파싱 마스크 (0-255)
            jaw_bbox: 턱 영역 바운딩 박스 [x1, y1, x2, y2]
            
        Returns:
            (is_occluded, adjusted_mask): 가림 여부와 조정된 마스크
        """
        x1, y1, x2, y2 = jaw_bbox
        
        # 턱 영역 추출
        jaw_region = original_image[y1:y2, x1:x2]
        jaw_mask = face_parsing_mask[y1:y2, x1:x2]
        
        # 가림 감지를 위한 여러 방법 적용
        is_occluded = self._detect_occlusion_multi_method(jaw_region, jaw_mask)
        
        if is_occluded:
            # 가려진 부분을 제외한 적응적 마스크 생성
            adjusted_mask = self._create_adaptive_mask(face_parsing_mask, jaw_region, jaw_bbox)
        else:
            adjusted_mask = face_parsing_mask
            
        return is_occluded, adjusted_mask
    
    def _detect_occlusion_multi_method(self, jaw_region: np.ndarray, jaw_mask: np.ndarray) -> bool:
        """
        다중 방법을 사용한 가림 감지
        
        Args:
            jaw_region: 턱 영역 이미지
            jaw_mask: 턱 영역 마스크
            
        Returns:
            bool: 가림 여부
        """
        # 방법 1: 색상 일관성 검사 (마이크는 보통 검은색이나 금속색)
        occlusion_score_color = self._detect_by_color_consistency(jaw_region, jaw_mask)
        
        # 방법 2: 엣지 밀도 검사 (마이크는 강한 엣지를 가짐)
        occlusion_score_edge = self._detect_by_edge_density(jaw_region, jaw_mask)
        
        # 방법 3: 텍스처 분석 (피부와 다른 텍스처 감지)
        occlusion_score_texture = self._detect_by_texture_analysis(jaw_region, jaw_mask)
        
        # 종합 점수 계산
        combined_score = (occlusion_score_color + occlusion_score_edge + occlusion_score_texture) / 3.0
        
        return combined_score > self.sensitivity_threshold
    
    def _detect_by_color_consistency(self, jaw_region: np.ndarray, jaw_mask: np.ndarray) -> float:
        """
        색상 일관성을 통한 가림 감지
        피부색과 다른 색상 영역을 감지
        """
        if jaw_mask.sum() == 0:
            return 0.0
            
        # 배열 크기 검증 및 조정
        if jaw_region.shape[:2] != jaw_mask.shape[:2]:
            print(f"크기 불일치 감지: jaw_region {jaw_region.shape}, jaw_mask {jaw_mask.shape}")
            # 더 작은 크기에 맞춰 조정
            min_h = min(jaw_region.shape[0], jaw_mask.shape[0])
            min_w = min(jaw_region.shape[1], jaw_mask.shape[1])
            jaw_region = jaw_region[:min_h, :min_w]
            jaw_mask = jaw_mask[:min_h, :min_w]
            
        # 마스크 영역의 평균 색상 계산
        masked_region = jaw_region[jaw_mask > 0]
        if len(masked_region) == 0:
            return 0.0
            
        # HSV 색공간으로 변환하여 피부색 범위 확인
        hsv_region = cv2.cvtColor(jaw_region, cv2.COLOR_BGR2HSV)
        masked_hsv = hsv_region[jaw_mask > 0]
        
        # 피부색 범위 정의 (HSV)
        skin_hue_range = (0, 25)  # 피부색 색조 범위
        skin_sat_range = (30, 255)  # 채도 범위
        
        # 피부색이 아닌 픽셀 비율 계산
        non_skin_pixels = 0
        total_pixels = len(masked_hsv)
        
        for pixel in masked_hsv:
            h, s, v = pixel
            if not (skin_hue_range[0] <= h <= skin_hue_range[1] and 
                   skin_sat_range[0] <= s <= skin_sat_range[1]):
                non_skin_pixels += 1
                
        return non_skin_pixels / total_pixels if total_pixels > 0 else 0.0
    
    def _detect_by_edge_density(self, jaw_region: np.ndarray, jaw_mask: np.ndarray) -> float:
        """
        엣지 밀도를 통한 가림 감지
        마이크 등의 물체는 강한 엣지를 가짐
        """
        # 그레이스케일 변환
        gray_region = cv2.cvtColor(jaw_region, cv2.COLOR_BGR2GRAY)
        
        # Canny 엣지 검출
        edges = cv2.Canny(gray_region, 50, 150)
        
        # 마스크 영역 내의 엣지 밀도 계산
        masked_edges = edges[jaw_mask > 0]
        if len(masked_edges) == 0:
            return 0.0
            
        edge_density = np.sum(masked_edges > 0) / len(masked_edges)
        
        # 정상적인 얼굴 영역보다 높은 엣지 밀도는 가림을 의미
        normal_edge_threshold = 0.1
        return max(0.0, (edge_density - normal_edge_threshold) / (1.0 - normal_edge_threshold))
    
    def _detect_by_texture_analysis(self, jaw_region: np.ndarray, jaw_mask: np.ndarray) -> float:
        """
        텍스처 분석을 통한 가림 감지
        피부의 부드러운 텍스처와 다른 거친 텍스처 감지
        """
        # 그레이스케일 변환
        gray_region = cv2.cvtColor(jaw_region, cv2.COLOR_BGR2GRAY)
        
        # LBP (Local Binary Pattern)를 사용한 텍스처 분석
        # 간단한 변분(variance) 기반 텍스처 측정 사용
        kernel = np.ones((3, 3), np.float32) / 9
        blurred = cv2.filter2D(gray_region, -1, kernel)
        variance = cv2.absdiff(gray_region, blurred)
        
        # 마스크 영역의 평균 텍스처 변분 계산
        masked_variance = variance[jaw_mask > 0]
        if len(masked_variance) == 0:
            return 0.0
            
        avg_variance = np.mean(masked_variance)
        
        # 정상 피부 텍스처보다 높은 변분은 가림을 의미
        normal_texture_threshold = 20.0
        return min(1.0, avg_variance / (normal_texture_threshold * 2))
    
    def _create_adaptive_mask(self, original_mask: np.ndarray, 
                            jaw_region: np.ndarray, 
                            jaw_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        가려진 부분을 고려한 스마트 적응적 마스크 생성
        가려진 부분은 제외하되, 가려지지 않은 입 영역은 보존하여 립싱크 유지
        
        Args:
            original_mask: 원본 얼굴 파싱 마스크
            jaw_region: 턱 영역 이미지
            jaw_bbox: 턱 영역 바운딩 박스
            
        Returns:
            np.ndarray: 스마트하게 조정된 마스크
        """
        adjusted_mask = original_mask.copy()
        x1, y1, x2, y2 = jaw_bbox
        
        # 가려진 영역을 세밀하게 분석
        jaw_mask_region = original_mask[y1:y2, x1:x2]
        
        # 색상 기반으로 가려진 픽셀 식별
        hsv_jaw = cv2.cvtColor(jaw_region, cv2.COLOR_BGR2HSV)
        
        # 가림 정도를 점진적으로 계산하는 마스크 생성
        occlusion_confidence = np.zeros_like(jaw_mask_region, dtype=np.float32)
        
        # 각 픽셀에 대해 가림 정도를 0-1 사이의 값으로 계산
        for i in range(hsv_jaw.shape[0]):
            for j in range(hsv_jaw.shape[1]):
                if jaw_mask_region[i, j] > 0:  # 원래 마스크 영역인 경우만
                    h, s, v = hsv_jaw[i, j]
                    
                    # 피부색 유사도 계산 (0: 완전히 다름, 1: 완전히 같음)
                    skin_similarity = self._calculate_skin_similarity(h, s, v)
                    
                    # 밝기 기반 가림 판단 (너무 어두우면 가려진 것으로 판단)
                    brightness_factor = min(1.0, v / 100.0)  # V값이 100 이하면 어두운 것으로 판단
                    
                    # 종합 가림 신뢰도 계산 (값이 낮을수록 가려진 것)
                    occlusion_confidence[i, j] = skin_similarity * brightness_factor
        
        # 가려진 영역을 부분적으로 마스크에서 제거 (완전 제거가 아닌 점진적 제거)
        for i in range(jaw_mask_region.shape[0]):
            for j in range(jaw_mask_region.shape[1]):
                if jaw_mask_region[i, j] > 0:
                    confidence = occlusion_confidence[i, j]
                    
                    # 신뢰도가 낮은 영역은 마스크 강도를 줄임 (완전 제거 X)
                    if confidence < 0.3:  # 매우 가려진 영역
                        adjusted_mask[y1+i, x1+j] = int(jaw_mask_region[i, j] * 0.1)  # 90% 감소
                    elif confidence < 0.5:  # 부분적으로 가려진 영역
                        adjusted_mask[y1+i, x1+j] = int(jaw_mask_region[i, j] * 0.4)  # 60% 감소
                    elif confidence < 0.7:  # 약간 가려진 영역
                        adjusted_mask[y1+i, x1+j] = int(jaw_mask_region[i, j] * 0.7)  # 30% 감소
                    # else: 가려지지 않은 영역은 원본 마스크 유지
        
        # 마스크의 연속성을 위한 부드러운 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 커널 크기 줄임
        adjusted_mask = cv2.morphologyEx(adjusted_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 가장자리 부드럽게 처리
        adjusted_mask = cv2.GaussianBlur(adjusted_mask, (3, 3), 0)
        
        return adjusted_mask

    def _calculate_skin_similarity(self, h: int, s: int, v: int) -> float:
        """
        주어진 HSV 값이 피부색과 얼마나 유사한지 계산
        
        Args:
            h, s, v: HSV 색상 값
            
        Returns:
            float: 피부색 유사도 (0.0-1.0)
        """
        # 피부색 범위 정의 (더 넓은 범위 사용)
        ideal_skin_h = 12  # 이상적인 피부색 색조
        ideal_skin_s = 150  # 이상적인 피부색 채도
        ideal_skin_v = 180  # 이상적인 피부색 명도
        
        # 각 채널별 유사도 계산
        h_similarity = max(0, 1 - abs(h - ideal_skin_h) / 25.0)  # 색조 유사도
        s_similarity = max(0, 1 - abs(s - ideal_skin_s) / 125.0)  # 채도 유사도
        v_similarity = max(0, 1 - abs(v - ideal_skin_v) / 100.0)  # 명도 유사도
        
        # 가중 평균으로 종합 유사도 계산
        # 색조가 가장 중요하고, 명도도 중요함
        total_similarity = (h_similarity * 0.5 + s_similarity * 0.2 + v_similarity * 0.3)
        
        return total_similarity

    def create_smart_lip_sync_mask(self, original_mask: np.ndarray,
                                  jaw_region: np.ndarray,
                                  jaw_bbox: Tuple[int, int, int, int],
                                  lip_priority_area: Tuple[int, int, int, int] = None) -> np.ndarray:
        """
        립싱크를 위한 스마트 마스크 생성
        입 영역은 최대한 보존하되, 명백히 가려진 부분만 제외
        
        Args:
            original_mask: 원본 마스크
            jaw_region: 턱 영역 이미지
            jaw_bbox: 턱 영역 바운딩 박스
            lip_priority_area: 입 영역 우선 보존 구역 (선택사항)
            
        Returns:
            np.ndarray: 립싱크 최적화된 마스크
        """
        x1, y1, x2, y2 = jaw_bbox
        smart_mask = original_mask.copy()
        
        # 입 영역 우선 보존 구역 설정 (전체 턱 영역의 하단 60%)
        if lip_priority_area is None:
            lip_y_start = y1 + int((y2 - y1) * 0.4)  # 턱 영역의 하단 60%
            lip_priority_area = (x1, lip_y_start, x2, y2)
        
        lip_x1, lip_y1, lip_x2, lip_y2 = lip_priority_area
        
        # 일반 적응적 마스크 생성
        adapted_mask = self._create_adaptive_mask(original_mask, jaw_region, jaw_bbox)
        
        # 입 영역에서는 더 관대한 기준 적용
        jaw_mask_region = original_mask[y1:y2, x1:x2]
        hsv_jaw = cv2.cvtColor(jaw_region, cv2.COLOR_BGR2HSV)
        
        for i in range(max(0, lip_y1-y1), min(jaw_mask_region.shape[0], lip_y2-y1)):
            for j in range(max(0, lip_x1-x1), min(jaw_mask_region.shape[1], lip_x2-x1)):
                if jaw_mask_region[i, j] > 0:
                    h, s, v = hsv_jaw[i, j]
                    
                    # 입 영역에서는 더 관대한 기준 사용
                    # 매우 어둡거나 명백히 금속성이 아닌 이상 보존
                    if v > 30 and not (h > 100 and s > 200):  # 매우 관대한 기준
                        # 입 영역은 최소 50% 이상 마스크 강도 유지
                        current_value = adapted_mask[y1+i, x1+j]
                        min_value = int(original_mask[y1+i, x1+j] * 0.5)
                        smart_mask[y1+i, x1+j] = max(current_value, min_value)
        
        return smart_mask

    def visualize_occlusion_detection(self, original_image: np.ndarray,
                                    original_mask: np.ndarray,
                                    adjusted_mask: np.ndarray,
                                    jaw_bbox: Tuple[int, int, int, int],
                                    save_path: Optional[str] = None) -> np.ndarray:
        """
        가림 감지 결과를 시각화
        
        Args:
            original_image: 원본 이미지
            original_mask: 원본 마스크
            adjusted_mask: 조정된 마스크
            jaw_bbox: 턱 영역 바운딩 박스
            save_path: 저장 경로 (선택사항)
            
        Returns:
            np.ndarray: 시각화된 이미지
        """
        x1, y1, x2, y2 = jaw_bbox
        
        # 시각화 이미지 생성
        vis_image = original_image.copy()
        
        # 원본 마스크를 파란색으로 표시
        vis_image[original_mask > 0] = [255, 0, 0]  # 빨간색
        
        # 조정된 마스크를 초록색으로 표시
        vis_image[adjusted_mask > 0] = [0, 255, 0]  # 초록색
        
        # 턱 영역 바운딩 박스 표시
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 노란색
        
        # 가려진 영역 (원본에는 있지만 조정된 마스크에는 없는 영역)을 보라색으로 표시
        occluded_region = (original_mask > 0) & (adjusted_mask == 0)
        vis_image[occluded_region] = [255, 0, 255]  # 보라색
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
            
        return vis_image 