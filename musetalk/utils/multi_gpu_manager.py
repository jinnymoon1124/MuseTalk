import torch
import torch.multiprocessing as mp
from typing import List, Dict, Tuple, Any
import queue
import threading
import time
import numpy as np
import cv2
import math
from .audio_processor import AudioProcessor
from musetalk.utils.utils import load_all_model, datagen
from transformers import WhisperModel
import copy
import logging
import os

# 멀티프로세싱 설정 - spawn 방식으로 복원 (fork는 CUDA 재초기화 불가)
try:
    # fork 방식은 CUDA 재초기화가 불가능하므로 spawn으로 복원
    # spawn에서 target 함수 대신 환경 변수 기반으로 GPU 할당
    mp.set_start_method('spawn', force=True)
    print("🔧 [MultiGPU] 멀티프로세싱 방법: spawn (CUDA 재초기화 필요)")
except RuntimeError as e:
    print(f"⚠️ [MultiGPU] 멀티프로세싱 설정 경고: {e}")
    print("   - 이미 설정된 start_method를 사용합니다")

def worker_env_wrapper(gpu_id: int, model_paths: Dict[str, str], 
                      use_float16: bool, task_queue: mp.Queue, 
                      result_queue: mp.Queue, main_gpu_id: int = None):
    """
    워커 프로세스 환경 변수 설정 래퍼 함수
    multiprocessing.Process가 시작되면 가장 먼저 실행되어 환경 변수를 설정합니다.
    """
    # 🔥 SUPER CRITICAL: 워커 프로세스임을 표시하는 특별한 파일 생성
    import os
    import tempfile
    
    # 환경 변수 설정
    os.environ['MUSETALK_WORKER_PROCESS'] = 'TRUE'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['WORKER_GPU_ID'] = str(gpu_id)
    os.environ['FORCE_WORKER_GPU'] = str(gpu_id)
    
    # 🔥 CRITICAL: 워커 프로세스임을 알리는 임시 파일 생성
    worker_flag_file = f"/tmp/musetalk_worker_{os.getpid()}.flag"
    with open(worker_flag_file, 'w') as f:
        f.write(f"GPU_{gpu_id}")
    
    print(f"🔥 [GPU {gpu_id} Wrapper] 환경 변수 및 플래그 파일 설정 완료:")
    print(f"   - MUSETALK_WORKER_PROCESS: {os.environ.get('MUSETALK_WORKER_PROCESS')}")
    print(f"   - CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"   - WORKER_GPU_ID: {os.environ.get('WORKER_GPU_ID')}")
    print(f"   - Worker Flag File: {worker_flag_file}")
    
    try:
        # 실제 워커 함수 호출
        return dynamic_worker_process(gpu_id, model_paths, use_float16, task_queue, result_queue, main_gpu_id)
    finally:
        # 정리: 플래그 파일 삭제
        try:
            os.remove(worker_flag_file)
        except:
            pass

def dynamic_worker_process(gpu_id: int, model_paths: Dict[str, str], 
                          use_float16: bool, task_queue: mp.Queue, 
                          result_queue: mp.Queue, main_gpu_id: int = None):
    """
    동적 작업 분배를 위한 워커 프로세스
    
    GPU가 작업을 완료하면 즉시 다음 작업을 가져와서 처리하는 방식
    
    Args:
        main_gpu_id: 메인 프로세스가 사용 중인 GPU ID (피해야 할 GPU)
    """
    try:
        # 🔥 CRITICAL: 워커 프로세스 시작 즉시 환경 변수 설정 (app.py 전역 코드보다 먼저 실행)
        import os
        import sys
        
        # 🔥 SUPER CRITICAL: 워커 프로세스임을 명시하는 특별한 환경 변수 설정
        os.environ['MUSETALK_WORKER_PROCESS'] = 'TRUE'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        os.environ['WORKER_GPU_ID'] = str(gpu_id)
        os.environ['FORCE_WORKER_GPU'] = str(gpu_id)
        
        print(f"🔧 [GPU {gpu_id} Dynamic Worker] 워커 프로세스 시작됨")
        print(f"🔥 [GPU {gpu_id} Dynamic Worker] dynamic_worker_process 함수 실행 중!")
        print(f"🔥 [GPU {gpu_id} Dynamic Worker] 워커 프로세스 내 환경 변수 강제 설정:")
        print(f"   - MUSETALK_WORKER_PROCESS: {os.environ.get('MUSETALK_WORKER_PROCESS')}")
        print(f"   - CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"   - WORKER_GPU_ID: {os.environ.get('WORKER_GPU_ID')}")
        print(f"   - FORCE_WORKER_GPU: {os.environ.get('FORCE_WORKER_GPU')}")
        print(f"⚠️ [GPU {gpu_id} Dynamic Worker] 주의: 이 로그가 보이면 래퍼 함수가 작동하고 있습니다!")
        
        # 메인 프로세스가 사용 중인 GPU와 겹치는 경우 경고
        if main_gpu_id is not None and gpu_id == main_gpu_id:
            print(f"⚠️ [GPU {gpu_id} Dynamic Worker] 메인 프로세스와 같은 GPU 사용 - 메모리 부족 가능성")
        
        # 🔥 CRITICAL: PyTorch 재임포트하여 CUDA 환경 초기화 (환경 변수 적용)
        import torch
        if torch.cuda.is_available():
            # 메모리 단편화 방지 설정 (워커 프로세스 시작 시점)
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,garbage_collection_threshold:0.6'
            
            torch.cuda.empty_cache()
            print(f"🔧 [GPU {gpu_id} Dynamic Worker] PyTorch CUDA 재초기화 완료")
            print(f"   - 메모리 단편화 방지 설정 적용: max_split_size_mb=256")
        
        # GPU 설정 - CUDA_VISIBLE_DEVICES로 인해 항상 cuda:0 사용
        device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES로 매핑됨
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()  # GPU 메모리 정리
        
        print(f"🔧 [GPU {gpu_id} Dynamic Worker] GPU 설정 완료")
        print(f"   - 논리 디바이스: {device} (실제 물리 GPU: {gpu_id})")
        print(f"   - CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        
        # GPU 메모리 상태 확인 및 초기 정리
        if torch.cuda.is_available():
            # 워커 시작 시 메모리 정리
            torch.cuda.empty_cache()
            
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
            print(f"   - 할당된 메모리: {memory_allocated:.2f}GB")
            print(f"   - 예약된 메모리: {memory_reserved:.2f}GB")
            
            # 메모리 단편화 방지 설정은 이미 워커 시작 시점에 적용됨
        
        # 모델 로드
        print(f"🔧 [GPU {gpu_id} Dynamic Worker] 모델 로딩 시작...")
        start_time = time.time()
        worker_instance = GPUWorker(device, model_paths, use_float16)
        load_time = time.time() - start_time
        print(f"✅ [GPU {gpu_id} Dynamic Worker] 모델 로딩 완료 ({load_time:.2f}초)")
        
        processed_count = 0
        
        # 동적 작업 처리 루프
        while True:
            try:
                # 작업 큐에서 다음 작업 가져오기 (타임아웃 30초)
                task = task_queue.get(timeout=30)
                
                if task is None:  # 종료 신호
                    print(f"🏁 [GPU {gpu_id} Dynamic Worker] 종료 신호 수신, 총 {processed_count}개 세그먼트 처리 완료")
                    break
                
                segment_id = task['segment_id']
                print(f"🎬 [GPU {gpu_id} Dynamic Worker] 세그먼트 {segment_id} 처리 시작 ({processed_count+1}번째 작업)")
                
                # 세그먼트 처리
                start_time = time.time()
                result = worker_instance.process_segment(task)
                process_time = time.time() - start_time
                processed_count += 1
                
                # 결과를 결과 큐에 전송 (타임아웃 방지)
                try:
                    result_queue.put(result, timeout=10)  # 10초 타임아웃
                    if result['success']:
                        print(f"✅ [GPU {gpu_id} Dynamic Worker] 세그먼트 {segment_id} 완료 ({process_time:.2f}초) - 결과 전송 성공")
                    else:
                        print(f"❌ [GPU {gpu_id} Dynamic Worker] 세그먼트 {segment_id} 실패 - 결과 전송 성공")
                except queue.Full:
                    print(f"⚠️ [GPU {gpu_id} Dynamic Worker] 결과 큐 가득참 - 세그먼트 {segment_id} 결과 손실")
                
            except queue.Empty:
                print(f"⏰ [GPU {gpu_id} Dynamic Worker] 30초간 새 작업 없음, 종료")
                break
            except Exception as e:
                print(f"💥 [GPU {gpu_id} Dynamic Worker] 작업 처리 중 오류: {e}")
                try:
                    result_queue.put({
                        'segment_id': task.get('segment_id', -1),
                        'gpu_id': gpu_id,
                        'error': str(e),
                        'success': False
                    }, timeout=10)
                    print(f"🔄 [GPU {gpu_id} Dynamic Worker] 에러 결과 전송 완료")
                except queue.Full:
                    print(f"⚠️ [GPU {gpu_id} Dynamic Worker] 에러 결과 전송 실패 - 큐 가득참")
                
    except Exception as e:
        print(f"💥 [GPU {gpu_id} Dynamic Worker] 워커 초기화 실패: {e}")
        import traceback
        traceback.print_exc()

class DynamicMultiGPUManager:
    """
    동적 작업 분배를 지원하는 멀티 GPU 매니저
    
    특징:
    - GPU가 작업을 완료하면 즉시 다음 작업 할당
    - 8개 GPU로 긴 영상도 효율적으로 처리
    - 작업 완료 순서와 상관없이 결과를 올바른 순서로 정렬
    """
    
    def __init__(self, num_gpus: int = None, segment_duration: int = 30):
        """
        동적 멀티 GPU 매니저 초기화
        """
        available_gpus = torch.cuda.device_count()
        
        if num_gpus is None:
            self.num_gpus = available_gpus
        else:
            self.num_gpus = min(num_gpus, available_gpus)
            
        if self.num_gpus == 0:
            raise RuntimeError("사용 가능한 GPU가 없습니다")
            
        self.segment_duration = segment_duration
        self.workers = []
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        
        print(f"🚀 [DynamicMultiGPU] 동적 멀티 GPU 매니저 초기화:")
        print(f"   - 사용할 GPU: {self.num_gpus}개")
        print(f"   - 세그먼트 길이: {self.segment_duration}초")
        print(f"   - 작업 분배 방식: 동적 큐 기반")
        
    def initialize_workers(self, model_paths: Dict[str, str], use_float16: bool = True, main_gpu_id: int = None):
        """
        모든 GPU에서 동적 워커 프로세스 시작
        
        Args:
            main_gpu_id: 메인 프로세스가 사용 중인 GPU ID
        """
        print(f"⚙️ [DynamicMultiGPU] {self.num_gpus}개 GPU 워커 시작 중...")
        if main_gpu_id is not None:
            print(f"   - 메인 프로세스 GPU: {main_gpu_id} (메모리 충돌 주의)")
        
        self.model_paths = model_paths
        self.use_float16 = use_float16
        
        # 메인 프로세스가 사용하는 GPU를 제외하고 워커 프로세스 시작
        available_gpus = []
        for gpu_id in range(self.num_gpus):
            if gpu_id != main_gpu_id:  # 메인 프로세스 GPU 제외
                available_gpus.append(gpu_id)
        
        print(f"   - 사용 가능한 워커 GPU: {available_gpus}")
        
        # 사용 가능한 GPU에서만 워커 프로세스 시작
        for gpu_id in available_gpus:
            # 🔥 CRITICAL: 환경 변수를 사전 설정하여 워커 프로세스 시작
            import os
            
            # 현재 프로세스의 환경 변수를 임시로 설정 (자식 프로세스가 상속)
            original_env = {}
            env_vars = {
                'MUSETALK_WORKER_PROCESS': 'TRUE',
                'CUDA_VISIBLE_DEVICES': str(gpu_id),
                'WORKER_GPU_ID': str(gpu_id),
                'FORCE_WORKER_GPU': str(gpu_id)
            }
            
            # 기존 환경 변수 백업 및 새 환경 변수 설정
            for key, value in env_vars.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value
            
            print(f"🔥 [GPU {gpu_id}] 워커 환경 변수 설정:")
            print(f"   - MUSETALK_WORKER_PROCESS: {os.environ.get('MUSETALK_WORKER_PROCESS')}")
            print(f"   - CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
            print(f"   - WORKER_GPU_ID: {os.environ.get('WORKER_GPU_ID')}")
            
            # multiprocessing.Process로 워커 시작 (환경 변수 상속됨)
            worker = mp.Process(
                target=worker_env_wrapper,
                args=(gpu_id, model_paths, use_float16, self.task_queue, self.result_queue, main_gpu_id)
            )
            worker.start()
            self.workers.append(worker)
            
            # 환경 변수 복원
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
            
            print(f"   ✅ GPU {gpu_id} 동적 워커 시작 (PID: {worker.pid})")
            print(f"      - 환경 변수 상속 완료")
        
        # 실제 워커 수 업데이트
        actual_worker_count = len(available_gpus)
        print(f"   - 실제 워커 수: {actual_worker_count}개 (메인 GPU {main_gpu_id} 제외)")
            
        print(f"🎯 [DynamicMultiGPU] 모든 워커 시작 완료")

    def process_audio_video(self, audio_path: str, video_frames: List[np.ndarray],
                           coord_list: List, model_config: Dict) -> List[np.ndarray]:
        """
        오디오와 비디오를 멀티 GPU로 병렬 처리 (기존 방식)
        
        AudioProcessor의 split_audio_for_multi_gpu를 사용하여 세그먼트를 미리 분할하고
        각 GPU에 정적으로 할당하는 방식입니다.
        
        Args:
            audio_path: 오디오 파일 경로
            video_frames: 비디오 프레임 리스트
            coord_list: 좌표 리스트
            model_config: 모델 설정
            
        Returns:
            List[np.ndarray]: 처리된 프레임 리스트
        """
        # 1. 오디오를 세그먼트로 분할
        print(f"🎵 [DynamicMultiGPU] AudioProcessor 초기화 중...")
        try:
            audio_processor = AudioProcessor()
            segments = audio_processor.split_audio_for_multi_gpu(
                audio_path, self.segment_duration, self.num_gpus
            )
        except Exception as e:
            print(f"💥 [DynamicMultiGPU] AudioProcessor 초기화 실패: {e}")
            raise
        
        # 2. 각 세그먼트를 해당 GPU에 할당
        print(f"📋 [DynamicMultiGPU] 작업을 GPU에 분산 배치 중...")
        total_frames = len(video_frames)
        print(f"   - 총 비디오 프레임: {total_frames}개")
        print(f"   - FPS: {model_config.get('fps', 50)}")
        
        # 비디오 프레임을 numpy 배열로 변환 (pickle 가능하도록)
        video_frames_serializable = []
        for frame in video_frames:
            if isinstance(frame, np.ndarray):
                video_frames_serializable.append(frame)
            else:
                video_frames_serializable.append(np.array(frame))
        
        # 모든 세그먼트를 작업 큐에 추가
        task_count = 0
        for segment in segments:
            gpu_id = segment['gpu_id']
            
            # 해당 세그먼트에 대응하는 비디오 프레임 계산
            start_frame = int(segment['start_time'] * model_config['fps'])
            end_frame = int(segment['end_time'] * model_config['fps'])
            
            # 순환 리스트에서 프레임 추출
            segment_frames = []
            segment_coords = []
            total_frames = len(video_frames_serializable)
            
            for frame_idx in range(start_frame, end_frame):
                cycle_idx = frame_idx % total_frames
                segment_frames.append(video_frames_serializable[cycle_idx])
                segment_coords.append(coord_list[cycle_idx])
            
            # 작업 패키지 생성
            # 세그먼트의 실제 길이 정보를 config에 추가 (강사 강의 영상 처리를 위한 정확한 시간 동기화)
            segment_duration = segment['end_time'] - segment['start_time']
            task_config = model_config.copy()
            task_config['segment_duration'] = segment_duration  # 실제 세그먼트 길이 (초)
            
            task = {
                'segment_id': segment['segment_id'],
                'audio_segment': segment,
                'video_frames': segment_frames,
                'coord_list': segment_coords,
                'config': task_config,  # 세그먼트 길이 정보가 포함된 config
                'start_frame_global': start_frame,
                'end_frame_global': end_frame
            }
            
            # 해당 GPU의 작업 큐에 추가
            self.task_queue.put(task)
            task_count += 1
            
            # 처음 3개와 마지막 3개 세그먼트 정보만 출력
            if task_count <= 3 or task_count > len(segments) - 3:
                print(f"   📦 태스크 {segment['segment_id']}: {segment['start_time']:.1f}-{segment['end_time']:.1f}초 "
                      f"({len(segment_frames)}프레임) → GPU {gpu_id}")
            elif task_count == 4:
                print(f"   📦 ... (중간 태스크 생략)")
            
        print(f"✅ [DynamicMultiGPU] 총 {len(segments)}개 세그먼트를 {self.num_gpus}개 GPU에 분산 배치 완료")
        
        # 3. 결과 수집 및 정렬
        results = {}
        completed_segments = 0
        start_time = time.time()
        
        print(f"📊 [DynamicMultiGPU] GPU 처리 결과 수집 시작...")
        print(f"   - 대기 중인 세그먼트: {len(segments)}개")
        
        while completed_segments < len(segments):
            try:
                # 타임아웃을 더 길게 설정 (강의 영상 처리에는 충분한 시간)
                result = self.result_queue.get(timeout=1800)  # 30분 타임아웃
                
                if result['success']:
                    results[result['segment_id']] = result
                    completed_segments += 1
                    elapsed_time = time.time() - start_time
                    progress_percent = (completed_segments / len(segments)) * 100
                    
                    print(f"✅ [Result] 세그먼트 {result['segment_id']} 완료 "
                          f"(GPU {result['gpu_id']}) - "
                          f"{completed_segments}/{len(segments)} ({progress_percent:.1f}%) "
                          f"[경과시간: {elapsed_time:.1f}초]")
                    
                    # 예상 완료 시간 계산
                    if completed_segments > 0:
                        avg_time_per_segment = elapsed_time / completed_segments
                        remaining_segments = len(segments) - completed_segments
                        eta = remaining_segments * avg_time_per_segment
                        print(f"   ⏱️  예상 완료까지: {eta:.1f}초")
                else:
                    print(f"❌ [Result] 세그먼트 {result['segment_id']} 처리 실패: {result['error']}")
                    # 실패한 세그먼트도 완료로 카운트 (무한 대기 방지)
                    completed_segments += 1
                    
            except queue.Empty:
                elapsed_time = time.time() - start_time
                print(f"⏰ [DynamicMultiGPU] 결과 수집 타임아웃 ({elapsed_time:.1f}초 경과) - 워커 프로세스 상태 확인 중...")
                
                # 살아있는 워커 프로세스 확인
                alive_workers = [i for i, worker in enumerate(self.workers) if worker.is_alive()]
                dead_workers = [i for i, worker in enumerate(self.workers) if not worker.is_alive()]
                
                print(f"   - 살아있는 워커: GPU {alive_workers}")
                print(f"   - 종료된 워커: GPU {dead_workers}")
                print(f"   - 완료된 세그먼트: {completed_segments}/{len(segments)}")
                print(f"   - 결과 큐 크기: {self.result_queue.qsize()}")
                print(f"   - 작업 큐 크기: {self.task_queue.qsize()}")
                
                if not alive_workers:
                    print("   ❌ 모든 워커 프로세스가 종료됨, 결과 수집 중단")
                    break
                elif completed_segments == 0 and elapsed_time > 120:  # 2분 후에도 결과 없음
                    print("   ❌ 2분 후에도 결과 없음, 처리 실패로 판단")
                    break
                else:
                    print(f"   🔄 {len(alive_workers)}개 워커가 여전히 실행 중, 계속 대기...")
                    continue
                
        total_time = time.time() - start_time
        print(f"📊 [DynamicMultiGPU] 결과 수집 완료 (총 {total_time:.2f}초 소요)")
        
        # 4. 결과를 시간 순서대로 병합
        print(f"🔗 [DynamicMultiGPU] 결과 병합 중...")
        final_frames = []
        
        missing_segments = []
        for segment_id in range(len(segments)):
            if segment_id in results:
                result = results[segment_id]
                frames_count = len(result['processed_frames'])
                final_frames.extend(result['processed_frames'])
                print(f"   ✅ 세그먼트 {segment_id}: {frames_count}개 프레임 병합")
            else:
                missing_segments.append(segment_id)
                print(f"   ❌ 세그먼트 {segment_id}: 누락됨")
        
        if missing_segments:
            print(f"⚠️  [DynamicMultiGPU] 누락된 세그먼트: {missing_segments}")
        
        print(f"🎉 [DynamicMultiGPU] 멀티 GPU 처리 완료:")
        print(f"   - 총 처리 시간: {total_time:.2f}초")
        print(f"   - 생성된 프레임: {len(final_frames)}개")
        print(f"   - 성공한 세그먼트: {len(results)}/{len(segments)}개")
        
        return final_frames
        
    def process_audio_video_dynamic(self, audio_path: str, video_frames: List[np.ndarray],
                                   coord_list: List, model_config: Dict) -> List[np.ndarray]:
        """
        동적 작업 분배로 오디오-비디오 처리
        """
        # 1. 오디오를 세그먼트로 분할
        print(f"🎵 [DynamicMultiGPU] 오디오 분할 시작...")
        audio_processor = AudioProcessor()
        
        # 전체 오디오 길이 계산
        import librosa
        audio_data, sr = librosa.load(audio_path, sr=16000)
        total_duration = len(audio_data) / sr
        
        # 세그먼트 개수 계산
        total_segments = math.ceil(total_duration / self.segment_duration)
        print(f"   - 전체 오디오 길이: {total_duration:.2f}초")
        print(f"   - 세그먼트 길이: {self.segment_duration}초")
        print(f"   - 총 세그먼트 수: {total_segments}개")
        print(f"   - 사용 가능한 GPU: {self.num_gpus}개")
        
        # 2. 모든 세그먼트를 작업 큐에 추가
        print(f"📋 [DynamicMultiGPU] 작업 큐에 {total_segments}개 세그먼트 추가 중...")
        
        for segment_id in range(total_segments):
            start_time = segment_id * self.segment_duration
            end_time = min((segment_id + 1) * self.segment_duration, total_duration)
            
            # 해당 시간 구간의 오디오 데이터 추출
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = audio_data[start_sample:end_sample]
            
            # 비디오 프레임 추출
            start_frame = int(start_time * model_config['fps'])
            end_frame = int(end_time * model_config['fps'])
            
            segment_frames = []
            segment_coords = []
            total_frames = len(video_frames)
            
            for frame_idx in range(start_frame, end_frame):
                cycle_idx = frame_idx % total_frames
                segment_frames.append(video_frames[cycle_idx])
                segment_coords.append(coord_list[cycle_idx])
            
            # 작업 패키지 생성
            # 세그먼트의 실제 길이 정보를 config에 추가 (강사 강의 영상 처리를 위한 정확한 시간 동기화)
            segment_duration = end_time - start_time
            task_config = model_config.copy()
            task_config['segment_duration'] = segment_duration  # 실제 세그먼트 길이 (초)
            
            task = {
                'segment_id': segment_id,
                'audio_segment': {
                    'segment_id': segment_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': segment_duration,
                    'audio_data': segment_audio
                },
                'video_frames': segment_frames,
                'coord_list': segment_coords,
                'config': task_config  # 세그먼트 길이 정보가 포함된 config
            }
            
            # 작업 큐에 추가
            self.task_queue.put(task)
            
            if segment_id < 5 or segment_id >= total_segments - 5:
                print(f"   📦 세그먼트 {segment_id}: {start_time:.1f}-{end_time:.1f}초 ({len(segment_frames)}프레임)")
            elif segment_id == 5:
                print(f"   📦 ... (중간 세그먼트 생략)")
        
        print(f"✅ [DynamicMultiGPU] 모든 작업을 큐에 추가 완료")
        
        # 3. 결과 수집
        print(f"📊 [DynamicMultiGPU] 결과 수집 시작 (총 {total_segments}개 대기)...")
        results = {}
        completed_count = 0
        start_time = time.time()
        
        while completed_count < total_segments:
            try:
                result = self.result_queue.get(timeout=1800)  # 30분 타임아웃 (세그먼트당 충분한 시간)
                
                if result['success']:
                    results[result['segment_id']] = result
                    completed_count += 1
                    elapsed = time.time() - start_time
                    progress = (completed_count / total_segments) * 100
                    
                    print(f"✅ [Result] 세그먼트 {result['segment_id']} 완료 "
                          f"(GPU {result['gpu_id']}) - "
                          f"{completed_count}/{total_segments} ({progress:.1f}%) "
                          f"[경과: {elapsed:.1f}초]")
                    
                    # ETA 계산
                    if completed_count > 0:
                        avg_time = elapsed / completed_count
                        remaining = total_segments - completed_count
                        eta = remaining * avg_time
                        print(f"   ⏱️  예상 완료까지: {eta:.1f}초")
                else:
                    print(f"❌ [Result] 세그먼트 {result['segment_id']} 실패: {result.get('error', 'Unknown')}")
                    
            except queue.Empty:
                print("⏰ [DynamicMultiGPU] 결과 수집 타임아웃")
                break
        
        # 4. 워커 종료 신호 전송
        print(f"🔄 [DynamicMultiGPU] 워커 프로세스 종료 신호 전송...")
        for _ in range(self.num_gpus):
            self.task_queue.put(None)
        
        # 5. 결과 정렬 및 병합
        print(f"🔗 [DynamicMultiGPU] 결과 정렬 및 병합 중...")
        final_frames = []
        
        for segment_id in range(total_segments):
            if segment_id in results:
                result = results[segment_id]
                final_frames.extend(result['processed_frames'])
                print(f"   ✅ 세그먼트 {segment_id}: {len(result['processed_frames'])}개 프레임 병합")
            else:
                print(f"   ❌ 세그먼트 {segment_id}: 누락됨")
        
        total_time = time.time() - start_time
        print(f"🎉 [DynamicMultiGPU] 동적 병렬 처리 완료:")
        print(f"   - 총 처리 시간: {total_time:.2f}초")
        print(f"   - 생성된 프레임: {len(final_frames)}개")
        print(f"   - 성공률: {len(results)}/{total_segments} ({len(results)/total_segments*100:.1f}%)")
        print(f"   - 평균 GPU 활용률: {len(results)/total_segments/self.num_gpus*100:.1f}%")
        
        return final_frames
    
    def process_with_full_dynamic_queue(self, audio_path: str, video_frames: List[np.ndarray],
                                       coord_list: List, model_config: Dict) -> List[np.ndarray]:
        """
        완전한 동적 작업 분배로 오디오-비디오 처리
        
        사용자 요구사항:
        - 1분 영상, 30초 세그먼트, 4개 GPU의 경우
        - 1-10, 11-20, 21-30, 31-40 순서로 세그먼트 생성
        - 1,2,3,4번 세그먼트를 4개 GPU에서 병렬 처리
        - 3번 GPU가 먼저 끝나면 5번 세그먼트 할당
        - 1번 GPU가 끝나면 6번 세그먼트 할당
        - 이런 식으로 동적 할당하여 모든 GPU가 최대한 활용되도록 함
        
        Args:
            audio_path: 오디오 파일 경로
            video_frames: 비디오 프레임 리스트
            coord_list: 좌표 리스트
            model_config: 모델 설정
            
        Returns:
            List[np.ndarray]: 처리된 프레임 리스트
        """
        # 1. 오디오를 세그먼트로 분할 (GPU 할당 없이)
        print(f"🎵 [FullDynamic] 오디오 세그먼트 생성 시작...")
        try:
            audio_processor = AudioProcessor()
            segments = audio_processor.create_dynamic_segments(
                audio_path, self.segment_duration
            )
        except Exception as e:
            print(f"💥 [FullDynamic] 오디오 세그먼트 생성 실패: {e}")
            raise
        
        total_segments = len(segments)
        print(f"📊 [FullDynamic] 동적 처리 시작:")
        print(f"   - 총 세그먼트: {total_segments}개")
        print(f"   - 사용 가능한 GPU: {self.num_gpus}개")
        print(f"   - 세그먼트 길이: {self.segment_duration}초")
        print(f"   - 예상 처리 방식: 먼저 끝나는 GPU가 다음 작업 즉시 처리")
        
        # 2. 비디오 프레임을 numpy 배열로 변환 (pickle 가능하도록)
        print(f"🖼️  [FullDynamic] 비디오 프레임 직렬화 중...")
        video_frames_serializable = []
        for frame in video_frames:
            if isinstance(frame, np.ndarray):
                video_frames_serializable.append(frame)
            else:
                video_frames_serializable.append(np.array(frame))
        
        # 3. 모든 세그먼트를 동적 작업 큐에 추가
        print(f"📋 [FullDynamic] 모든 세그먼트를 동적 큐에 추가 중...")
        
        for segment in segments:
            # 해당 세그먼트에 대응하는 비디오 프레임 계산
            start_frame = int(segment['start_time'] * model_config['fps'])
            end_frame = int(segment['end_time'] * model_config['fps'])
            
            # 순환 리스트에서 프레임 추출
            segment_frames = []
            segment_coords = []
            total_frames = len(video_frames_serializable)
            
            for frame_idx in range(start_frame, end_frame):
                cycle_idx = frame_idx % total_frames
                segment_frames.append(video_frames_serializable[cycle_idx])
                segment_coords.append(coord_list[cycle_idx])
            
            # 작업 패키지 생성 (GPU 할당 없음)
            # 세그먼트의 실제 길이 정보를 config에 추가 (강사 강의 영상 처리를 위한 정확한 시간 동기화)
            segment_duration = segment['end_time'] - segment['start_time']
            task_config = model_config.copy()
            task_config['segment_duration'] = segment_duration  # 실제 세그먼트 길이 (초)
            
            task = {
                'segment_id': segment['segment_id'],
                'audio_segment': segment,
                'video_frames': segment_frames,
                'coord_list': segment_coords,
                'config': task_config,  # 세그먼트 길이 정보가 포함된 config
                'start_frame_global': start_frame,
                'end_frame_global': end_frame
            }
            
            # 동적 작업 큐에 추가
            self.task_queue.put(task)
            
            # 처음 몇 개와 마지막 몇 개 세그먼트 정보만 출력
            if segment['segment_id'] < 5 or segment['segment_id'] >= total_segments - 5:
                print(f"   📦 세그먼트 {segment['segment_id']}: {segment['start_time']:.1f}-{segment['end_time']:.1f}초 "
                      f"({len(segment_frames)}프레임) → 동적 큐")
            elif segment['segment_id'] == 5:
                print(f"   📦 ... (중간 세그먼트 생략)")
        
        print(f"✅ [FullDynamic] 총 {total_segments}개 세그먼트를 동적 큐에 추가 완료")
        print(f"🚀 [FullDynamic] {self.num_gpus}개 GPU가 동적으로 작업을 가져가서 처리 시작...")
        
        # 4. 결과 수집 및 정렬
        results = {}
        completed_segments = 0
        start_time = time.time()
        
        # GPU별 처리 현황 추적
        gpu_processing_count = {i: 0 for i in range(self.num_gpus)}
        
        print(f"📊 [FullDynamic] GPU 처리 결과 수집 시작...")
        print(f"   - 대기 중인 세그먼트: {total_segments}개")
        
        while completed_segments < total_segments:
            try:
                result = self.result_queue.get(timeout=1800)  # 30분 타임아웃 (세그먼트당 충분한 시간)
                
                if result['success']:
                    results[result['segment_id']] = result
                    completed_segments += 1
                    gpu_id = result['gpu_id']
                    gpu_processing_count[gpu_id] += 1
                    
                    elapsed_time = time.time() - start_time
                    progress_percent = (completed_segments / total_segments) * 100
                    
                    print(f"✅ [Result] 세그먼트 {result['segment_id']} 완료 "
                          f"(GPU {gpu_id}, {gpu_processing_count[gpu_id]}번째 작업) - "
                          f"{completed_segments}/{total_segments} ({progress_percent:.1f}%) "
                          f"[경과시간: {elapsed_time:.1f}초]")
                    
                    # 예상 완료 시간 계산
                    if completed_segments > 0:
                        avg_time_per_segment = elapsed_time / completed_segments
                        remaining_segments = total_segments - completed_segments
                        eta = remaining_segments * avg_time_per_segment
                        print(f"   ⏱️  예상 완료까지: {eta:.1f}초")
                        
                        # GPU별 활용률 표시
                        if completed_segments % 5 == 0 or completed_segments == total_segments:
                            gpu_utilization = [f"GPU{i}:{gpu_processing_count[i]}" for i in range(self.num_gpus)]
                            print(f"   📈 GPU 활용 현황: {', '.join(gpu_utilization)}")
                else:
                    print(f"❌ [Result] 세그먼트 {result['segment_id']} 처리 실패: {result['error']}")
                    
            except queue.Empty:
                print("⏰ [FullDynamic] 결과 수집 타임아웃 - 일부 GPU가 응답하지 않을 수 있습니다")
                break
                
        total_time = time.time() - start_time
        print(f"📊 [FullDynamic] 결과 수집 완료 (총 {total_time:.2f}초 소요)")
        
        # 5. 워커 종료 신호 전송
        print(f"🔄 [FullDynamic] 워커 프로세스 종료 신호 전송...")
        for _ in range(self.num_gpus):
            self.task_queue.put(None)
        
        # 6. 결과를 시간 순서대로 병합
        print(f"🔗 [FullDynamic] 결과 병합 중...")
        final_frames = []
        
        missing_segments = []
        for segment_id in range(total_segments):
            if segment_id in results:
                result = results[segment_id]
                frames_count = len(result['processed_frames'])
                final_frames.extend(result['processed_frames'])
                print(f"   ✅ 세그먼트 {segment_id}: {frames_count}개 프레임 병합")
            else:
                missing_segments.append(segment_id)
                print(f"   ❌ 세그먼트 {segment_id}: 누락됨")
        
        if missing_segments:
            print(f"⚠️  [FullDynamic] 누락된 세그먼트: {missing_segments}")
        
        # 최종 통계 출력
        print(f"🎉 [FullDynamic] 완전 동적 멀티 GPU 처리 완료:")
        print(f"   - 총 처리 시간: {total_time:.2f}초")
        print(f"   - 생성된 프레임: {len(final_frames)}개")
        print(f"   - 성공한 세그먼트: {len(results)}/{total_segments}개")
        print(f"   - 성공률: {len(results)/total_segments*100:.1f}%")
        
        # GPU별 최종 활용 통계
        print(f"   - GPU별 처리 현황:")
        for gpu_id in range(self.num_gpus):
            utilization = (gpu_processing_count[gpu_id] / total_segments) * 100
            print(f"     * GPU {gpu_id}: {gpu_processing_count[gpu_id]}개 세그먼트 ({utilization:.1f}%)")
        
        return final_frames
    
    def shutdown(self):
        """
        모든 워커 프로세스 정리
        """
        print("🔄 [DynamicMultiGPU] 워커 프로세스 정리 중...")
        
        # 모든 워커 종료 대기
        for i, worker in enumerate(self.workers):
            worker.join(timeout=10)
            if worker.is_alive():
                print(f"   ⚠️  GPU {i} 워커 강제 종료")
                worker.terminate()
            else:
                print(f"   ✅ GPU {i} 워커 정상 종료")
        
        print("🏁 [DynamicMultiGPU] 모든 워커 정리 완료")


class GPUWorker:
    """
    개별 GPU에서 실행되는 워커 클래스
    """
    
    def __init__(self, device: torch.device, model_paths: Dict[str, str], 
                 use_float16: bool = True):
        """
        GPU 워커 초기화
        
        Args:
            device: GPU 디바이스
            model_paths: 모델 파일 경로
            use_float16: float16 사용 여부
        """
        self.device = device
        self.use_float16 = use_float16
        self.weight_dtype = torch.float16 if use_float16 else torch.float32
        
        # 🔧 실제 물리 GPU ID 저장 (로그에서 CUDA_VISIBLE_DEVICES 매핑 문제 해결)
        import os
        self.physical_gpu_id = int(os.environ.get('WORKER_GPU_ID', device.index))
        print(f"🔧 [GPUWorker] 물리 GPU ID 저장: {self.physical_gpu_id} (논리 디바이스: {device})")
        

        # 모델 로드 (이 부분은 실제 모델 로딩 코드로 교체 필요)
        self._load_models(model_paths)
        
    def _load_models(self, model_paths: Dict[str, str]):
        """
        모델들을 GPU에 로드
        
        Args:
            model_paths: 모델 파일 경로들
        """
        # MuseTalk 모델 로딩
        print(f"🔧 [GPU {self.device.index} Worker] MuseTalk 모델 로딩 시작...")
        print(f"   - unet_model_path: {model_paths.get('unet_model_path', './models/musetalkV15/unet.pth')}")
        print(f"   - vae_type: {model_paths.get('vae_type', 'sd-vae')}")
        print(f"   - unet_config: {model_paths.get('unet_config', './models/musetalkV15/musetalk.json')}")
        
        try:
            # 워커 프로세스에서는 명시적으로 디바이스를 강제 설정
            print(f"🔧 [GPU {self.device.index} Worker] 강제 디바이스 설정: {self.device}")
            torch.cuda.set_device(self.device)
            
            self.vae, self.unet, self.pe = load_all_model(
                unet_model_path=model_paths.get('unet_model_path', "./models/musetalkV15/unet.pth"),
                vae_type=model_paths.get('vae_type', "sd-vae"),
                unet_config=model_paths.get('unet_config', "./models/musetalkV15/musetalk.json"),
                device=self.device
            )
            print(f"✅ [GPU {self.device.index} Worker] MuseTalk 모델 로딩 성공")
        except Exception as e:
            print(f"💥 [GPU {self.device.index} Worker] MuseTalk 모델 로딩 실패: {e}")
            raise
        
        # 데이터 타입 변환
        if self.use_float16:
            self.pe = self.pe.half()
            self.vae.vae = self.vae.vae.half()
            self.unet.model = self.unet.model.half()
        
        # 모델을 GPU로 이동
        self.pe = self.pe.to(self.device)
        self.vae.vae = self.vae.vae.to(self.device)
        self.unet.model = self.unet.model.to(self.device)
        
        # Whisper 모델 로딩
        whisper_path = model_paths.get('whisper_path', "openai/whisper-tiny")
        print(f"🔧 [GPU {self.device.index} Worker] Whisper 모델 로딩 시작...")
        print(f"   - whisper_path: {whisper_path}")
        
        try:
            self.whisper = WhisperModel.from_pretrained(whisper_path)
            self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
            self.whisper.requires_grad_(False)
            print(f"✅ [GPU {self.device.index} Worker] Whisper 모델 로딩 성공")
        except Exception as e:
            print(f"💥 [GPU {self.device.index} Worker] Whisper 모델 로딩 실패: {e}")
            raise
        
        # 타임스텝 설정
        self.timesteps = torch.tensor([0], device=self.device)
        
        print(f"🎯 [GPU {self.device.index} Worker] 모델 로딩 완료:")
        print(f"   - VAE: {'✅' if self.vae else '❌'}")
        print(f"   - UNet: {'✅' if self.unet else '❌'}")
        print(f"   - PE: {'✅' if self.pe else '❌'}")
        print(f"   - Whisper: {'✅' if self.whisper else '❌'}")
        print(f"   - 데이터 타입: {self.weight_dtype}")
        
        # 모델 로딩 완료 후 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"   - GPU 메모리 사용량: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f}GB")
        
    def process_segment(self, task: Dict) -> Dict:
        """
        세그먼트 처리
        
        Args:
            task: 처리할 작업 정보
            
        Returns:
            Dict: 처리 결과
        """
        try:
            segment_id = task['segment_id']
            audio_segment = task['audio_segment']
            video_frames = task['video_frames']
            config = task['config']
            
            gpu_id = self.device.index
            print(f"🎬 [GPU {gpu_id} Worker] 세그먼트 {segment_id} 처리 시작:")
            print(f"   - 오디오 길이: {audio_segment['duration']:.2f}초")
            print(f"   - 비디오 프레임: {len(video_frames)}개")
            print(f"   - 시간 범위: {audio_segment['start_time']:.1f}-{audio_segment['end_time']:.1f}초")
            
            # 1. 오디오 특징 추출
            # 새로운 동적 세그먼트 구조에서 오디오 데이터 추출
            audio_processor = AudioProcessor()
            audio_features = audio_processor.get_audio_feature_from_segment(
                audio_segment['audio_data'], self.weight_dtype
            )
            
            # 2. 비디오 프레임 처리 (실제 추론 로직)
            print(f"   🔄 [GPU {gpu_id} Worker] 프레임 처리 시작...")
            processed_frames = self._process_frames(
                audio_features, video_frames, config
            )
            
            print(f"   ✅ [GPU {gpu_id} Worker] 세그먼트 {segment_id} 처리 완료:")
            print(f"      - 생성된 프레임: {len(processed_frames)}개")
            print(f"      - GPU 메모리 사용량: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f}GB")
            
            # 세그먼트 처리 완료 후 메모리 정리 (다음 작업을 위해)
            torch.cuda.empty_cache()
            
            return {
                'segment_id': segment_id,
                'gpu_id': self.physical_gpu_id,  # 실제 물리 GPU ID 사용 (워커에서는 device.index가 항상 0)
                'processed_frames': processed_frames,
                'success': True,
                'processing_time': time.time()
            }
            
        except Exception as e:
            gpu_id = self.physical_gpu_id  # 실제 물리 GPU ID 사용
            print(f"💥 [GPU {gpu_id} Worker] 세그먼트 처리 오류: {e}")
            import traceback
            traceback.print_exc()
            return {
                'segment_id': task.get('segment_id', -1),
                'gpu_id': self.physical_gpu_id,  # 실제 물리 GPU ID 사용 (워커에서는 device.index가 항상 0)
                'error': str(e),
                'success': False
            }
            
    def _process_frames(self, audio_features: torch.Tensor, 
                       video_frames: List[np.ndarray], 
                       config: Dict) -> List[np.ndarray]:
        """
        실제 프레임 처리 로직
        
        Args:
            audio_features: 오디오 특징
            video_frames: 비디오 프레임들
            config: 설정
            
        Returns:
            List[np.ndarray]: 처리된 프레임들
        """
        # 1. 오디오 특징을 Whisper 청크로 변환
        gpu_id = self.device.index
        print(f"      🎵 [GPU {gpu_id}] 오디오 특징 처리 시작...")
        
        # 각 워커에서 독립적으로 AudioProcessor 생성
        try:
            audio_processor = AudioProcessor()
            print(f"      ✅ [GPU {gpu_id}] AudioProcessor 초기화 성공")
        except Exception as e:
            print(f"      💥 [GPU {gpu_id}] AudioProcessor 초기화 실패: {e}")
            raise
        
        # 오디오 특징을 리스트 형태로 변환 (30초 세그먼트)
        whisper_input_features = [audio_features]
        
        # 실제 세그먼트 오디오 길이를 config에서 가져오기 (초 단위)
        segment_duration = config.get('segment_duration', 5.0)  # 기본값 5초
        librosa_length = int(segment_duration * 16000)  # 샘플 수로 변환 (16kHz 기준)
        
        print(f"         - 오디오 특징 크기: {audio_features.shape}")
        print(f"         - 세그먼트 길이: {segment_duration}초")
        print(f"         - 실제 오디오 길이 (샘플): {librosa_length:,}개")
        print(f"         - 실제 오디오 길이 (초): {librosa_length/16000:.2f}초")
        
        # Whisper 청크 생성
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_length,
            fps=config.get('fps', 25),  # FPS를 25로 제한하여 청크 수 감소
            audio_padding_length_left=config.get('audio_padding_length_left', 0),
            audio_padding_length_right=config.get('audio_padding_length_right', 0),
        )
        
        # AudioProcessor 사용 후 메모리 정리
        del audio_processor, whisper_input_features
        torch.cuda.empty_cache()
        
        # 2. 비디오 프레임을 VAE 잠재 공간으로 변환
        print(f"      🖼️  [GPU {gpu_id}] 비디오 프레임 처리 시작...")
        input_latent_list = []
        coord_list = config.get('coord_list', [])
        print(f"         - 입력 프레임 수: {len(video_frames)}")
        print(f"         - 좌표 리스트 길이: {len(coord_list)}")
        
        for i, frame in enumerate(video_frames):
            if i < len(coord_list):
                # 좌표가 있는 경우 얼굴 영역 크롭
                bbox = coord_list[i]
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    extra_margin = config.get('extra_margin', 10)
                    y2 = y2 + extra_margin
                    y2 = min(y2, frame.shape[0])
                    
                    # 얼굴 영역 크롭 및 리사이즈
                    crop_frame = frame[y1:y2, x1:x2]
                    crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                else:
                    # 좌표가 없는 경우 전체 프레임 사용
                    crop_frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            else:
                # 좌표 리스트가 부족한 경우 전체 프레임 사용
                crop_frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            # VAE를 사용하여 잠재 벡터로 변환
            latents = self.vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)
        
        # 3. 배치 단위로 추론 실행
        print(f"      🔄 [GPU {gpu_id}] 배치 추론 시작...")
        
        # 멀티 GPU 환경에서 메모리 효율성을 위해 배치 크기 동적 조정
        available_memory = torch.cuda.get_device_properties(self.device).total_memory
        allocated_memory = torch.cuda.memory_allocated(self.device)
        free_memory = available_memory - allocated_memory
        
        # 사용 가능한 메모리에 따라 배치 크기 동적 조정 (보수적 접근)
        if free_memory > 8 * 1024**3:  # 8GB 이상 여유
            batch_size = 8
        elif free_memory > 4 * 1024**3:  # 4GB 이상 여유
            batch_size = 4
        else:  # 4GB 미만 여유
            batch_size = 2
            
        print(f"         💾 [GPU {gpu_id}] 메모리 상태:")
        print(f"            - 전체 메모리: {available_memory / 1024**3:.2f}GB")
        print(f"            - 할당된 메모리: {allocated_memory / 1024**3:.2f}GB")
        print(f"            - 여유 메모리: {free_memory / 1024**3:.2f}GB")
        print(f"            - 동적 배치 크기: {batch_size}")
        
        video_num = len(whisper_chunks)
        
        print(f"         - Whisper 청크 수: {video_num}")
        print(f"         - 배치 크기: {batch_size}")
        print(f"         - 예상 배치 수: {int(np.ceil(video_num / batch_size))}")
        
        # 데이터 생성기 생성
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=input_latent_list,
            batch_size=batch_size,
            delay_frame=0,
            device=self.device,
        )
        
        processed_frames = []
        batch_count = 0
        
        # 배치별로 처리
        for i, (whisper_batch, latent_batch) in enumerate(gen):
            batch_count += 1
            # print(f"         🎯 배치 {batch_count}: {whisper_batch.shape[0]}개 프레임 처리 중...")
            
            # 오디오 특징을 위치 인코딩
            audio_feature_batch = self.pe(whisper_batch)
            
            # 잠재 벡터 데이터 타입 변환
            latent_batch = latent_batch.to(dtype=self.weight_dtype)
            
            # UNet 모델로 새로운 잠재 벡터 생성
            pred_latents = self.unet.model(
                latent_batch, 
                self.timesteps, 
                encoder_hidden_states=audio_feature_batch
            ).sample
            
            # VAE로 이미지 디코딩 (메모리 효율적 처리)
            try:
                recon = self.vae.decode_latents(pred_latents)
                
                # 결과 프레임 추가
                batch_frames = 0
                for res_frame in recon:
                    processed_frames.append(res_frame)
                    batch_frames += 1
                
                # 배치 처리 후 즉시 메모리 정리 (OOM 방지)
                del pred_latents, recon
                torch.cuda.empty_cache()
                
                # print(f"         ✅ 배치 {batch_count} 완료: {batch_frames}개 프레임 생성")
                
            except torch.cuda.OutOfMemoryError as e:
                # OOM 발생 시 메모리 정리 후 더 작은 배치로 재시도
                print(f"         ⚠️ [GPU {gpu_id}] 배치 {batch_count} OOM 발생, 메모리 정리 후 개별 처리")
                torch.cuda.empty_cache()
                
                # 개별 프레임 단위로 처리 (안전 모드)
                for i in range(len(latent_batch)):
                    try:
                        single_latent = latent_batch[i:i+1]
                        single_audio = audio_feature_batch[i:i+1]
                        
                        single_pred = self.unet.model(
                            single_latent, 
                            self.timesteps, 
                            encoder_hidden_states=single_audio
                        ).sample
                        
                        single_recon = self.vae.decode_latents(single_pred)
                        processed_frames.append(single_recon[0])
                        
                        # 각 프레임 처리 후 메모리 정리
                        del single_latent, single_audio, single_pred, single_recon
                        torch.cuda.empty_cache()
                        
                    except Exception as single_e:
                        print(f"         ❌ [GPU {gpu_id}] 개별 프레임 {i} 처리 실패: {single_e}")
                        # 실패한 프레임은 건너뛰고 계속 진행
                        continue
        
        print(f"      🎉 [GPU {gpu_id}] 모든 배치 처리 완료: 총 {len(processed_frames)}개 프레임")
        
        return processed_frames 