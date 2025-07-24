# MuseTalk 립싱크 품질 개선사항

## 문제점 해결

### 1. FPS 관련 오류 수정
- **문제**: 25fps에서 50fps로 변경 시 오디오-비디오 동기화 오류 발생
- **해결**: 
  - `audio_processor.py`의 경계 체크 로직 개선
  - 범위 초과 시 패딩 처리 추가
  - 기본 fps를 25fps로 복원

### 2. 움직이는 영상 립싱크 품질 향상

#### 오디오 처리 개선
- **오디오 패딩 증가**: 좌우 패딩을 2에서 3으로 증가하여 더 많은 오디오 컨텍스트 제공
- **배치 크기 감소**: 8에서 4로 줄여 더 정확한 처리

#### 블렌딩 품질 향상
- **향상된 가우시안 블러**: 블러 커널 크기를 0.05에서 0.08로 증가
- **추가 페더링 효과**: 더 부드러운 경계를 위한 추가 처리
- **개선된 마스크 처리**: 더 자연스러운 얼굴 합성

#### 프레임 연속성 개선
- **향상된 사이클링**: 더 많은 프레임(5개)을 사용한 순환 처리
- **템포럴 스무딩**: 연속된 프레임 간 평활화 옵션 추가

## 새로운 사용법

### 기본 사용법 (개선된 품질)
```bash
python -m scripts.inference --inference_config configs/inference/test.yaml
```

### 움직이는 영상에 최적화된 사용법
```bash
python -m scripts.inference \
    --inference_config configs/inference/test.yaml \
    --temporal_smoothing \
    --enhance_motion_consistency \
    --batch_size 4 \
    --audio_padding_length_left 3 \
    --audio_padding_length_right 3
```

### 고품질 처리 (느리지만 최고 품질)
```bash
python -m scripts.inference \
    --inference_config configs/inference/test.yaml \
    --temporal_smoothing \
    --batch_size 2 \
    --audio_padding_length_left 4 \
    --audio_padding_length_right 4 \
    --use_float16
```

## 파라미터 설명

### 새로운 파라미터들
- `--temporal_smoothing`: 연속 프레임 간 평활화 활성화
- `--enhance_motion_consistency`: 움직임 일관성 향상 (예약됨)
- `--audio_padding_length_left/right`: 오디오 컨텍스트 패딩 (기본값: 3)
- `--batch_size`: 배치 크기 (작을수록 품질 향상, 기본값: 4)

### 기존 파라미터 최적화
- `--fps`: 기본값 25fps로 복원 (안정성 향상)
- `--parsing_mode`: "jaw" 모드 권장 (움직이는 영상에 적합)
- `--left_cheek_width/right_cheek_width`: 90 (기본값 유지)

## 성능 vs 품질 트레이드오프

| 설정 | 처리 속도 | 품질 | 권장 사용 |
|------|----------|------|-----------|
| 기본 | 빠름 | 보통 | 일반적인 정적 이미지 |
| --temporal_smoothing | 중간 | 좋음 | 움직이는 영상 |
| --batch_size 2 + smoothing | 느림 | 최고 | 고품질 결과물 필요 시 |

## 문제 해결

### 여전히 끊김 현상이 있는 경우
1. `--temporal_smoothing` 옵션 사용
2. `--batch_size`를 2로 줄이기
3. `--audio_padding_length_left/right`를 4로 증가

### 메모리 부족 오류
1. `--batch_size`를 1로 줄이기
2. `--use_float16` 옵션 사용

### 립싱크 정확도가 떨어지는 경우
1. `bbox_shift` 파라미터 조정
2. `parsing_mode`를 "jaw"로 설정
3. 오디오 품질 확인 (16kHz 권장)

## 추가 팁

1. **영상 전처리**: 안정적인 얼굴 영역을 위해 영상을 전처리하는 것을 권장
2. **오디오 품질**: 고품질 오디오(16kHz, mono)를 사용하면 립싱크 품질이 향상됨
3. **하드웨어**: GPU 메모리가 충분한 경우 배치 크기를 늘려 처리 속도 향상 가능 