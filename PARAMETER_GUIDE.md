# MuseTalk 파라미터 완전 가이드

## 현재 설정 분석
- **bbox_shift**: 0 
- **extra_margin**: 0
- **parsing_mode**: jaw
- **left_cheek_width**: 20
- **right_cheek_width**: 20

## 각 파라미터 상세 설명

### 1. bbox_shift (경계 상자 이동)
**의미**: 얼굴 마스크의 상단 경계를 조정하여 오디오 특성이 입술 움직임에 미치는 영향을 제어

**작동 원리**:
- **양수 값**: 마스크가 아래쪽(landmark30 방향)으로 이동 → 입 열림 증가
- **음수 값**: 마스크가 위쪽(landmark28 방향)으로 이동 → 입 열림 감소
- **0**: 기본 위치(landmark29 근처)

**현재 설정 (0)의 효과**: 
- 균형잡힌 입술 움직임
- 대부분의 경우에 적합한 기본값

**권장 설정**:
- **일반적인 경우**: -2 ~ +3
- **입이 너무 많이 벌어지는 경우**: -3 ~ -1
- **입이 충분히 벌어지지 않는 경우**: +2 ~ +5

### 2. extra_margin (추가 여백)
**의미**: 얼굴 영역 아래쪽에 추가되는 여백으로, 턱과 목 부분의 움직임 범위를 결정

**작동 원리**:
- 얼굴 bounding box의 하단을 확장
- 턱의 움직임과 목 부분의 자연스러운 연결을 제어

**현재 설정 (0)의 문제점**:
- 턱 움직임이 제한될 수 있음
- 목과 턱의 경계가 부자연스러울 수 있음

**권장 설정**:
- **일반적인 경우**: 10-15
- **턱이 큰 사람**: 15-25
- **목이 긴 사람**: 20-30
- **정적 이미지**: 5-10

### 3. parsing_mode (파싱 모드)
**의미**: 얼굴 영역을 분할하는 방식을 결정

**옵션**:
- **jaw**: 턱과 입 주변 영역을 정교하게 분할 (권장)
- **raw**: 기본적인 얼굴 영역만 분할
- **neck**: 목 부분까지 포함

**현재 설정 (jaw)의 장점**:
- 입과 턱 주변의 정교한 제어
- 자연스러운 말하기 동작 구현
- 움직이는 영상에 최적화

**권장**: **jaw 모드 유지** (최적의 선택)

### 4. left_cheek_width & right_cheek_width (볼 너비)
**의미**: jaw 모드에서 좌우 볼 영역의 편집 범위를 결정

**작동 원리**:
- 볼 영역의 마스크 크기를 제어
- 얼굴의 자연스러운 형태 유지에 영향

**현재 설정 (20, 20)의 문제점**:
- 너무 작은 값으로 볼 영역이 과도하게 제한됨
- 부자연스러운 경계선 생성 가능
- 얼굴 형태 왜곡 위험

**권장 설정**:
- **일반적인 경우**: 80-100
- **넓은 얼굴**: 90-120
- **좁은 얼굴**: 60-80
- **정밀한 제어**: 70-90

## 🎯 최적 설정 권장값

### 자연스러운 결과를 위한 기본 설정
```
bbox_shift: 0
extra_margin: 15
parsing_mode: jaw
left_cheek_width: 90
right_cheek_width: 90
```

### 움직이는 영상용 고품질 설정
```
bbox_shift: -1
extra_margin: 20
parsing_mode: jaw
left_cheek_width: 85
right_cheek_width: 85
```

### 정적 이미지용 설정
```
bbox_shift: 1
extra_margin: 10
parsing_mode: jaw
left_cheek_width: 95
right_cheek_width: 95
```

## 🔧 문제별 해결 방법

### 입이 너무 많이 벌어지는 경우
```
bbox_shift: -3
extra_margin: 12
left_cheek_width: 80
right_cheek_width: 80
```

### 입이 충분히 벌어지지 않는 경우
```
bbox_shift: 3
extra_margin: 18
left_cheek_width: 100
right_cheek_width: 100
```

### 턱 움직임이 부자연스러운 경우
```
bbox_shift: 0
extra_margin: 25
left_cheek_width: 90
right_cheek_width: 90
```

### 볼 부분이 어색한 경우
```
bbox_shift: 0
extra_margin: 15
left_cheek_width: 70 (좁은 얼굴) / 110 (넓은 얼굴)
right_cheek_width: 70 (좁은 얼굴) / 110 (넓은 얼굴)
```

## 📊 파라미터 상호작용

### bbox_shift와 extra_margin의 관계
- bbox_shift가 양수일 때: extra_margin을 약간 증가 (더 많은 턱 공간 필요)
- bbox_shift가 음수일 때: extra_margin을 약간 감소 (덜 공격적인 턱 움직임)

### cheek_width와 parsing_mode의 관계
- jaw 모드에서만 cheek_width가 의미있음
- 작은 cheek_width는 더 정밀한 제어, 큰 값은 더 자연스러운 블렌딩

## 🎨 단계별 최적화 과정

### 1단계: 기본 설정으로 시작
```
bbox_shift: 0
extra_margin: 15
parsing_mode: jaw
left_cheek_width: 90
right_cheek_width: 90
```

### 2단계: 입술 움직임 조정
- 결과를 보고 bbox_shift를 -3~+3 범위에서 조정

### 3단계: 턱 움직임 조정
- extra_margin을 10-25 범위에서 조정

### 4단계: 볼 영역 미세조정
- cheek_width를 70-110 범위에서 조정

## ⚡ 빠른 테스트 방법

"1. Test Inpainting" 버튼을 사용하여 첫 번째 프레임만으로 빠르게 테스트:

1. 기본 설정으로 테스트
2. bbox_shift만 조정하여 테스트
3. extra_margin 조정
4. cheek_width 조정
5. 최종 전체 영상 생성

## 💡 추가 팁

1. **얼굴 크기에 따른 조정**: 큰 얼굴은 모든 값을 약간 증가
2. **영상 품질에 따른 조정**: 저화질 영상은 더 보수적인 설정 사용
3. **오디오 품질의 영향**: 고품질 오디오는 더 세밀한 설정 가능
4. **실시간 vs 배치**: 실시간 처리시 더 안정적인 설정 사용

현재 설정에서 **left/right_cheek_width를 90**으로, **extra_margin을 15**로 변경하시면 훨씬 자연스러운 결과를 얻으실 수 있습니다! 