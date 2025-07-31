import math
import os

import librosa
import numpy as np
import torch
from einops import rearrange
from transformers import AutoFeatureExtractor
from typing import List, Dict


class AudioProcessor:
    """
    오디오 처리 클래스 - 음성 파일을 분석하여 립싱크에 필요한 특징을 추출하는 역할
    
    이 클래스는 다음과 같은 작업을 수행합니다:
    1. 음성 파일을 읽어서 분석 가능한 형태로 변환
    2. 음성의 특징을 추출하여 AI 모델이 이해할 수 있는 숫자로 변환
    3. 비디오 프레임과 동기화할 수 있도록 시간별로 분할
    """
    
    def __init__(self, feature_extractor_path="openai/whisper-tiny"):
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

    def split_audio_for_multi_gpu(self, wav_path: str, segment_duration: int = 30, 
                                 num_gpus: int = 1) -> List[Dict]:
        """
        멀티 GPU 병렬 처리를 위해 오디오를 세그먼트로 분할
        
        강사 강의 영상의 특성을 고려하여 설계:
        - 움직이는 강사
        - 고개를 많이 돌리는 동작
        - 핸드마이크로 인한 얼굴 가림
        
        Args:
            wav_path: 음성 파일 경로
            segment_duration: 각 세그먼트 길이 (초, 기본값 30초)
            num_gpus: 사용할 GPU 개수
            
        Returns:
            List[Dict]: 각 세그먼트 정보
            [
                {
                    'segment_id': 0,
                    'start_time': 0.0,
                    'end_time': 30.0,
                    'start_sample': 0,
                    'end_sample': 480000,
                    'audio_data': numpy_array,
                    'gpu_id': 0
                },
                ...
            ]
        """
        # 파일 존재 여부 확인
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Audio file not found: {wav_path}")
            
        print(f"🎵 [AudioProcessor] 오디오 파일 로딩 시작: {wav_path}")
        
        # 음성 파일 로드 (16kHz로 표준화)
        librosa_output, sampling_rate = librosa.load(wav_path, sr=16000)
        assert sampling_rate == 16000, "샘플링 레이트가 16kHz가 아닙니다"
        
        # 전체 오디오 길이 계산
        total_duration = len(librosa_output) / sampling_rate
        segment_length_samples = segment_duration * sampling_rate  # 30초 = 480,000 샘플
        
        print(f"📊 [AudioProcessor] 오디오 정보:")
        print(f"   - 전체 길이: {total_duration:.2f}초 ({len(librosa_output):,} 샘플)")
        print(f"   - 세그먼트 길이: {segment_duration}초 ({segment_length_samples:,} 샘플)")
        print(f"   - 예상 세그먼트 수: {int(np.ceil(total_duration / segment_duration))}개")
        
        segments = []
        segment_id = 0
        gpu_assignment = 0  # GPU 할당을 위한 카운터
        
        # 세그먼트별로 분할
        for start_sample in range(0, len(librosa_output), segment_length_samples):
            end_sample = min(start_sample + segment_length_samples, len(librosa_output))
            start_time = start_sample / sampling_rate
            end_time = end_sample / sampling_rate
            
            # 세그먼트 데이터 추출
            segment_data = librosa_output[start_sample:end_sample]
            
            # 세그먼트 정보 생성
            segment_info = {
                'segment_id': segment_id,
                'start_time': start_time,
                'end_time': end_time,
                'start_sample': start_sample,
                'end_sample': end_sample,
                'audio_data': segment_data,
                'gpu_id': gpu_assignment % num_gpus,  # 라운드 로빈 방식으로 GPU 할당
                'duration': end_time - start_time
            }
            
            segments.append(segment_info)
            segment_id += 1
            gpu_assignment += 1
            
            # 세그먼트별 상세 정보 출력 (처음 5개와 마지막 5개만)
            if segment_id <= 5 or segment_id > len(segments) - 5:
                print(f"   📦 세그먼트 {segment_id-1}: {start_time:.1f}-{end_time:.1f}초 → GPU {gpu_assignment-1}")
            elif segment_id == 6:
                print(f"   📦 ... (중간 세그먼트 생략)")
            
        print(f"✅ [AudioProcessor] 오디오 분할 완료:")
        print(f"   - 총 {len(segments)}개 세그먼트 생성 (총 {total_duration:.2f}초)")
        print(f"   - {num_gpus}개 GPU에 라운드 로빈 방식으로 분산 배치")
        
        # GPU별 할당된 세그먼트 수 계산
        gpu_counts = {}
        for segment in segments:
            gpu_id = segment['gpu_id']
            gpu_counts[gpu_id] = gpu_counts.get(gpu_id, 0) + 1
        
        print(f"   - GPU별 할당 현황:")
        for gpu_id in sorted(gpu_counts.keys()):
            print(f"     GPU {gpu_id}: {gpu_counts[gpu_id]}개 세그먼트")
        
        return segments

    def create_dynamic_segments(self, wav_path: str, segment_duration: int = 30) -> List[Dict]:
        """
        완전 동적 작업 분배를 위한 세그먼트 생성
        
        GPU 할당 없이 세그먼트만 생성하여 동적 큐에서 처리할 수 있도록 함
        강사 강의 영상의 특성을 고려:
        - 움직이는 강사와 고개 움직임
        - 핸드마이크로 인한 얼굴 가림 등
        
        Args:
            wav_path: 음성 파일 경로
            segment_duration: 각 세그먼트 길이 (초, 기본값 30초)
            
        Returns:
            List[Dict]: GPU 할당 없는 순수한 세그먼트 정보 리스트
        """
        # 파일 존재 여부 확인
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Audio file not found: {wav_path}")
            
        print(f"🎵 [AudioProcessor] 동적 세그먼트 생성 시작: {wav_path}")
        
        # 오디오 파일 로딩 (16kHz로 리샘플링)
        librosa_output, sampling_rate = librosa.load(wav_path, sr=16000)
        total_duration = len(librosa_output) / sampling_rate
        segment_length_samples = segment_duration * sampling_rate
        
        print(f"   - 오디오 길이: {total_duration:.2f}초")
        print(f"   - 샘플링 레이트: {sampling_rate}Hz")
        print(f"   - 세그먼트 길이: {segment_duration}초")
        
        segments = []
        segment_id = 0
        
        # 세그먼트별로 분할 (GPU 할당 없이)
        for start_sample in range(0, len(librosa_output), segment_length_samples):
            end_sample = min(start_sample + segment_length_samples, len(librosa_output))
            start_time = start_sample / sampling_rate
            end_time = end_sample / sampling_rate
            
            # 세그먼트 데이터 추출
            segment_data = librosa_output[start_sample:end_sample]
            
            # 세그먼트 정보 생성 (GPU 할당 없음)
            segment_info = {
                'segment_id': segment_id,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'start_sample': start_sample,
                'end_sample': end_sample,
                'audio_data': segment_data,
                'sample_rate': sampling_rate
            }
            
            segments.append(segment_info)
            segment_id += 1
            
            # 세그먼트 정보 출력 (처음 5개와 마지막 5개만)
            if segment_id <= 5 or segment_id > len(segments) - 5:
                print(f"   📦 세그먼트 {segment_id-1}: {start_time:.1f}-{end_time:.1f}초 ({end_time-start_time:.1f}초)")
            elif segment_id == 6:
                print(f"   📦 ... (중간 세그먼트 생략)")
            
        print(f"✅ [AudioProcessor] 동적 세그먼트 생성 완료:")
        print(f"   - 총 {len(segments)}개 세그먼트 생성 (총 {total_duration:.2f}초)")
        print(f"   - GPU 할당 없음 (동적 큐에서 처리)")
        
        return segments

    def get_audio_feature_from_segment(self, segment_data: np.ndarray, 
                                     weight_dtype=None) -> torch.Tensor:
        """
        세그먼트 데이터에서 오디오 특징 추출
        
        Args:
            segment_data: 오디오 세그먼트 데이터
            weight_dtype: 데이터 타입
            
        Returns:
            torch.Tensor: 추출된 오디오 특징
        """
        # Whisper 특징 추출기를 사용하여 음성을 AI가 이해할 수 있는 형태로 변환
        audio_feature = self.feature_extractor(
            segment_data,
            return_tensors="pt",  # PyTorch 텐서 형태로 반환
            sampling_rate=16000
        ).input_features
        
        # 데이터 타입 변환이 필요한 경우 적용
        if weight_dtype is not None:
            audio_feature = audio_feature.to(dtype=weight_dtype)
            
        return audio_feature

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

