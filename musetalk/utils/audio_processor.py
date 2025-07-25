import math
import os

import librosa
import numpy as np
import torch
from einops import rearrange
from transformers import AutoFeatureExtractor


class AudioProcessor:
    """
    오디오 처리 클래스 - 음성 파일을 분석하여 립싱크에 필요한 특징을 추출하는 역할
    
    이 클래스는 다음과 같은 작업을 수행합니다:
    1. 음성 파일을 읽어서 분석 가능한 형태로 변환
    2. 음성의 특징을 추출하여 AI 모델이 이해할 수 있는 숫자로 변환
    3. 비디오 프레임과 동기화할 수 있도록 시간별로 분할
    """
    
    def __init__(self, feature_extractor_path="openai/whisper-tiny/"):
        """
        오디오 프로세서 초기화
        
        Args:
            feature_extractor_path: OpenAI의 Whisper 모델 경로
                                   (음성을 분석하는 AI 모델의 위치)
        """
        # Whisper 모델의 특징 추출기를 로드
        # 이 도구는 음성 파일을 AI가 이해할 수 있는 숫자 형태로 변환해줍니다
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path)

    def get_audio_feature(self, wav_path, start_index=0, weight_dtype=None):
        """
        음성 파일에서 특징을 추출하는 함수
        
        동작 과정:
        1. 음성 파일을 읽어서 16kHz로 변환 (AI 모델이 이해하기 쉬운 형태)
        2. 30초씩 나누어서 처리 (긴 음성도 효율적으로 처리하기 위해)
        3. 각 구간을 AI가 이해할 수 있는 숫자 형태로 변환
        
        Args:
            wav_path: 음성 파일 경로
            start_index: 시작 위치 (현재는 사용하지 않음)
            weight_dtype: 데이터 타입 (현재는 사용하지 않음)
            
        Returns:
            features: 추출된 음성 특징들 (30초씩 나눈 구간별)
            len(librosa_output): 전체 음성 길이 (샘플 수)
        """
        # 파일이 존재하는지 확인
        if not os.path.exists(wav_path):
            return None
            
        # 음성 파일을 읽어서 16kHz로 변환
        # 16kHz는 사람의 음성을 충분히 표현할 수 있는 표준 주파수입니다
        librosa_output, sampling_rate = librosa.load(wav_path, sr=16000)
        assert sampling_rate == 16000  # 16kHz가 맞는지 확인
        
        # 긴 음성을 30초씩 나누어서 처리
        # AI 모델이 한 번에 처리할 수 있는 최적의 길이입니다
        segment_length = 30 * sampling_rate  # 30초 × 16000 = 480,000개 샘플
        segments = [librosa_output[i:i + segment_length] for i in range(0, len(librosa_output), segment_length)]

        features = []
        # 각 30초 구간을 순서대로 처리
        for segment in segments:
            # Whisper 특징 추출기를 사용하여 음성을 AI가 이해할 수 있는 형태로 변환
            audio_feature = self.feature_extractor(
                segment,
                return_tensors="pt",  # PyTorch 텐서 형태로 반환
                sampling_rate=sampling_rate
            ).input_features
            
            # 데이터 타입 변환이 필요한 경우 적용
            if weight_dtype is not None:
                audio_feature = audio_feature.to(dtype=weight_dtype)
            features.append(audio_feature)

        return features, len(librosa_output)

    def get_whisper_chunk(
        self,
        whisper_input_features,
        device,
        weight_dtype,
        whisper,
        librosa_length,
        fps=50,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
    ):
        """
        음성 특징을 비디오 프레임과 동기화할 수 있도록 분할하는 함수
        
        이 함수는 립싱크의 핵심입니다:
        - 비디오는 1초에 50장의 이미지(프레임)로 구성됩니다
        - 음성도 같은 시간에 맞춰서 50개 구간으로 나눕니다
        - 각 프레임마다 해당 시간의 음성 특징을 제공합니다
        
        동작 과정:
        1. Whisper 모델을 사용하여 음성의 숨겨진 특징을 추출
        2. 비디오 프레임 수에 맞춰서 음성을 분할
        3. 각 프레임에 해당하는 음성 특징을 준비
        4. 경계 부분에서 부드럽게 연결되도록 패딩 추가
        
        Args:
            whisper_input_features: 추출된 음성 특징들
            device: GPU 또는 CPU 사용 여부
            weight_dtype: 데이터 타입 (float16 또는 float32)
            whisper: Whisper AI 모델
            librosa_length: 전체 음성 길이
            fps: 비디오 프레임 레이트 (초당 프레임 수, 기본값 50)
            audio_padding_length_left: 왼쪽 패딩 길이 (이전 프레임과의 연결을 위해)
            audio_padding_length_right: 오른쪽 패딩 길이 (다음 프레임과의 연결을 위해)
            
        Returns:
            audio_prompts: 각 프레임에 해당하는 음성 특징들
        """
        # 각 프레임에 필요한 음성 특징의 길이 계산
        # 예: 왼쪽 2프레임 + 현재 1프레임 + 오른쪽 2프레임 = 총 5프레임 분량
        audio_feature_length_per_frame = 2 * (audio_padding_length_left + audio_padding_length_right + 1)
        whisper_feature = []
        
        # 여러 개의 30초 음성 특징을 순서대로 처리
        for input_feature in whisper_input_features:
            # GPU로 데이터 이동 및 데이터 타입 변환
            input_feature = input_feature.to(device).to(weight_dtype)
            
            # Whisper 모델의 인코더를 사용하여 음성의 숨겨진 특징을 추출
            # hidden_states=True는 중간 결과들도 함께 가져오라는 의미입니다
            audio_feats = whisper.encoder(input_feature, output_hidden_states=True).hidden_states
            
            # 여러 층의 특징들을 하나로 합치기
            audio_feats = torch.stack(audio_feats, dim=2)
            whisper_feature.append(audio_feats)

        # 모든 30초 구간의 특징들을 하나로 연결
        whisper_feature = torch.cat(whisper_feature, dim=1)
        
        # 마지막 구간의 패딩 부분을 제거 (정확한 길이 맞추기)
        sr = 16000  # 샘플링 레이트 (초당 16000개 샘플)
        audio_fps = 50  # 오디오 처리용 프레임 레이트
        fps = int(fps)  # 실제 비디오 프레임 레이트
        
        # 비디오 프레임과 오디오 프레임 간의 변환 비율 계산
        # 예: 비디오가 25fps면, 오디오 50fps 대비 0.5배
        whisper_idx_multiplier = audio_fps / fps
        
        # 전체 비디오 프레임 수 계산
        # 예: 10초 음성 × 50fps = 500프레임
        num_frames = math.floor((librosa_length / sr) * fps)
        
        # 실제 오디오 특징의 길이 계산
        actual_length = math.floor((librosa_length / sr) * audio_fps)
        
        # 정확한 길이만큼만 사용 (패딩 제거)
        whisper_feature = whisper_feature[:,:actual_length,...]

        # 패딩 양 계산 (경계에서 부드럽게 연결하기 위해)
        padding_nums = math.ceil(whisper_idx_multiplier)
        
        # 시작과 끝에 패딩 추가
        # 이렇게 하면 첫 프레임과 마지막 프레임에서도 충분한 정보를 얻을 수 있습니다
        whisper_feature = torch.cat([
            # 시작 부분에 0으로 채운 패딩 추가
            torch.zeros_like(whisper_feature[:, :padding_nums * audio_padding_length_left]),
            # 실제 음성 특징
            whisper_feature,
            # 끝 부분에 0으로 채운 패딩 추가 (3배로 더 많이 추가하여 안전성 확보)
            torch.zeros_like(whisper_feature[:, :padding_nums * 3 * audio_padding_length_right])
        ], 1)

        audio_prompts = []
        # 각 비디오 프레임에 대해 해당하는 음성 특징을 준비
        for frame_index in range(num_frames):
            try:
                # 현재 프레임에 해당하는 오디오 위치 계산
                audio_index = math.floor(frame_index * whisper_idx_multiplier)
                
                # 필요한 음성 특징의 끝 위치 계산
                end_index = audio_index + audio_feature_length_per_frame
                
                # 배열 범위를 벗어나지 않는지 확인
                if end_index > whisper_feature.shape[1]:
                    # 범위를 벗어나는 경우, 사용 가능한 프레임만 사용하고 나머지는 0으로 채움
                    available_frames = whisper_feature.shape[1] - audio_index
                    if available_frames > 0:
                        # 사용 가능한 프레임만 가져오기
                        audio_clip = whisper_feature[:, audio_index:]
                        
                        # 부족한 부분을 0으로 채우기
                        if available_frames < audio_feature_length_per_frame:
                            padding_needed = audio_feature_length_per_frame - available_frames
                            padding = torch.zeros_like(whisper_feature[:, :padding_needed])
                            audio_clip = torch.cat([audio_clip, padding], dim=1)
                    else:
                        # 사용 가능한 프레임이 없는 경우, 모두 0으로 채움
                        audio_clip = torch.zeros_like(whisper_feature[:, :audio_feature_length_per_frame])
                else:
                    # 정상적인 경우, 필요한 구간만큼 가져오기
                    audio_clip = whisper_feature[:, audio_index: end_index]
                
                # 길이가 정확한지 확인 (디버깅용)
                assert audio_clip.shape[1] == audio_feature_length_per_frame
                audio_prompts.append(audio_clip)
                
            except Exception as e:
                # 오류 발생 시 상세한 정보 출력
                print(f"Error occurred: {e}")
                print(f"whisper_feature.shape: {whisper_feature.shape}")
                print(f"audio_clip.shape: {audio_clip.shape}")
                print(f"num frames: {num_frames}, fps: {fps}, whisper_idx_multiplier: {whisper_idx_multiplier}")
                print(f"frame_index: {frame_index}, audio_index: {audio_index}-{end_index}")
                
                # 오류가 발생해도 프로그램이 중단되지 않도록 0으로 채운 데이터 사용
                audio_clip = torch.zeros_like(whisper_feature[:, :audio_feature_length_per_frame])
                audio_prompts.append(audio_clip)
                continue

        # 모든 프레임의 음성 특징을 하나의 큰 배열로 합치기
        audio_prompts = torch.cat(audio_prompts, dim=0)  # T, 10, 5, 384 형태
        
        # 배열 형태를 재구성하여 AI 모델이 처리하기 쉬운 형태로 변환
        # 'b c h w -> b (c h) w': 배치, 채널, 높이, 너비를 배치, (채널×높이), 너비로 변환
        audio_prompts = rearrange(audio_prompts, 'b c h w -> b (c h) w')
        
        return audio_prompts

if __name__ == "__main__":
    # 테스트 코드 - 이 파일을 직접 실행했을 때만 동작
    audio_processor = AudioProcessor()
    wav_path = "./2.wav"
    audio_feature, librosa_feature_length = audio_processor.get_audio_feature(wav_path)
    print("Audio Feature shape:", audio_feature.shape)
    print("librosa_feature_length:", librosa_feature_length)

