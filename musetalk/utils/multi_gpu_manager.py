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
from transformers import WhisperModel, WhisperForConditionalGeneration
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
            task = None  # 🔧 [FIX] task 변수 초기화로 UnboundLocalError 방지
            try:
                # 작업 큐에서 다음 작업 가져오기 (타임아웃 30초)
                task = task_queue.get(timeout=30)
                
                if task is None:  # 종료 신호
                    print(f"🏁 [GPU {gpu_id} Dynamic Worker] 종료 신호 수신, 총 {processed_count}개 작업 처리 완료")
                    break
                
                # 작업 타입 확인 (세그먼트 vs 배치)
                if 'segment_id' in task:
                    # 기존 세그먼트 처리
                    segment_id = task['segment_id']
                    print(f"🎬 [GPU {gpu_id} Dynamic Worker] 세그먼트 {segment_id} 처리 시작 ({processed_count+1}번째 작업)")
                    
                    start_time = time.time()
                    result = worker_instance.process_segment(task)
                    process_time = time.time() - start_time
                    processed_count += 1
                elif 'batch_id' in task:
                    # 새로운 배치 처리
                    batch_id = task['batch_id']
                    print(f"📦 [GPU {gpu_id} Dynamic Worker] 배치 {batch_id} 처리 시작 ({processed_count+1}번째 작업)")
                    
                    start_time = time.time()
                    result = worker_instance.process_batch(task)
                    process_time = time.time() - start_time
                    processed_count += 1
                else:
                    print(f"❌ [GPU {gpu_id} Dynamic Worker] 알 수 없는 작업 타입: {task}")
                    continue
                
                # 결과를 결과 큐에 전송 (타입별 로그)
                try:
                    result_queue.put(result, timeout=10)  # 10초 타임아웃
                    if result['success']:
                        if 'segment_id' in task:
                            print(f"✅ [GPU {gpu_id} Dynamic Worker] 세그먼트 {task['segment_id']} 완료 ({process_time:.2f}초) - 결과 전송 성공")
                        elif 'batch_id' in task:
                            print(f"✅ [GPU {gpu_id} Dynamic Worker] 배치 {task['batch_id']} 완료 ({process_time:.2f}초) - 결과 전송 성공")
                    else:
                        if 'segment_id' in task:
                            print(f"❌ [GPU {gpu_id} Dynamic Worker] 세그먼트 {task['segment_id']} 실패 - 결과 전송 성공")
                        elif 'batch_id' in task:
                            print(f"❌ [GPU {gpu_id} Dynamic Worker] 배치 {task['batch_id']} 실패 - 결과 전송 성공")
                except queue.Full:
                    if 'segment_id' in task:
                        print(f"⚠️ [GPU {gpu_id} Dynamic Worker] 결과 큐 가득참 - 세그먼트 {task['segment_id']} 결과 손실")
                    elif 'batch_id' in task:
                        print(f"⚠️ [GPU {gpu_id} Dynamic Worker] 결과 큐 가득참 - 배치 {task['batch_id']} 결과 손실")
                
            except queue.Empty:
                print(f"⏰ [GPU {gpu_id} Dynamic Worker] 30초간 새 작업 없음, 종료")
                break
            except Exception as e:
                print(f"💥 [GPU {gpu_id} Dynamic Worker] 작업 처리 중 오류: {e}")
                # 🔧 [FIX] task가 None일 수 있으므로 안전하게 처리
                if task is not None:
                    if 'segment_id' in task:
                        task_id = task.get('segment_id', -1)
                        id_key = 'segment_id'
                    elif 'batch_id' in task:
                        task_id = task.get('batch_id', -1)
                        id_key = 'batch_id'
                    else:
                        task_id = -1
                        id_key = 'segment_id'  # 기본값
                else:
                    task_id = -1
                    id_key = 'segment_id'  # 기본값
                    
                try:
                    result_queue.put({
                        id_key: task_id,
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
    
    def _interpolate_coordinate(self, target_idx: int, valid_indices: List[int], coord_list: List) -> tuple:
        """
        고급 좌표 보간 함수 - 강사 움직임을 고려한 스마트 보간
        
        강사 강의 영상의 특성을 고려한 개선사항:
        1. 시간적 가중치를 적용한 스무스 보간
        2. 얼굴 크기 변화 감지 및 보정
        3. 급격한 움직임 감지 시 보수적 접근
        4. 핸드마이크 등으로 인한 가림 상황 대응
        
        Args:
            target_idx: 보간할 프레임 인덱스
            valid_indices: 유효한 좌표가 있는 인덱스 리스트
            coord_list: 전체 좌표 리스트
            
        Returns:
            tuple: 보간된 좌표 (x1, y1, x2, y2)
        """
        if not valid_indices:
            # 유효한 좌표가 없는 경우 기본값 반환
            return None
        
        # 🔧 [COORDINATE STABILITY] 강사 강의 영상을 위한 안정적 좌표 선택
        # 가장 가까운 유효한 좌표 찾기 (시간적 거리 기반)
        closest_idx = min(valid_indices, key=lambda x: abs(x - target_idx))
        
        # 📍 [ULTRA STABILITY] 덜덜거림 완전 제거를 위한 엄격한 좌표 제한
        # 연속된 프레임 간 좌표 변화를 엄격히 제한하여 안정성 우선
        max_coord_change = 15  # 픽셀 단위 최대 변화량을 더욱 엄격하게 (50 → 15)
        
        # 이전 프레임과의 좌표 차이 검증
        if target_idx > 0 and (target_idx - 1) in valid_indices:
            prev_coord = coord_list[target_idx - 1]
            current_coord = coord_list[closest_idx]
            
            if prev_coord is not None and current_coord is not None:
                # 좌표 변화량 계산
                coord_diff = abs(current_coord[0] - prev_coord[0]) + abs(current_coord[1] - prev_coord[1])
                
                # 급격한 변화 감지 시 이전 좌표 유지 (안정성 우선)
                if coord_diff > max_coord_change:
                    # 🎯 [ULTRA STABLE COORD] 덜덜거림 완전 제거를 위한 초안정 좌표
                    # 블렌딩 비율을 매우 보수적으로 조정 (0.8 → 0.95)
                    smoothed_coord = (
                        int(0.95 * prev_coord[0] + 0.05 * current_coord[0]),  # X1 (95% 이전, 5% 현재)
                        int(0.95 * prev_coord[1] + 0.05 * current_coord[1]),  # Y1
                        int(0.95 * prev_coord[2] + 0.05 * current_coord[2]),  # X2
                        int(0.95 * prev_coord[3] + 0.05 * current_coord[3])   # Y2
                    )
                    return smoothed_coord
        
        # 앞뒤 좌표 찾기
        before_indices = [idx for idx in valid_indices if idx < target_idx]
        after_indices = [idx for idx in valid_indices if idx > target_idx]
        
        if before_indices and after_indices:
            # 앞뒤 좌표가 모두 있는 경우 - 고급 보간 적용
            before_idx = max(before_indices)
            after_idx = min(after_indices)
            
            before_coord = coord_list[before_idx]
            after_coord = coord_list[after_idx]
            
            # 좌표 유효성 재검증
            if before_coord is None or after_coord is None:
                return coord_list[closest_idx]
            
            # 얼굴 크기 변화 감지 (강사 움직임 분석)
            before_width = before_coord[2] - before_coord[0]
            before_height = before_coord[3] - before_coord[1]
            after_width = after_coord[2] - after_coord[0]
            after_height = after_coord[3] - after_coord[1]
            
            # 급격한 크기 변화 감지 (50% 이상 변화 시 보수적 접근)
            size_change_ratio = max(
                abs(after_width - before_width) / max(before_width, 1),
                abs(after_height - before_height) / max(before_height, 1)
            )
            
            if size_change_ratio > 0.5:
                # 급격한 변화 시 가장 가까운 좌표 사용 (안전 모드)
                return coord_list[closest_idx]
            
            # 시간적 거리 기반 가중치 계산 (가까운 시점에 더 높은 가중치)
            time_distance = after_idx - before_idx
            if time_distance > 10:  # 10프레임 이상 차이 시 보수적 접근
                return coord_list[closest_idx]
            
            # 스무스 보간 계산 (코사인 보간으로 자연스러운 움직임)
            import math
            linear_weight = (target_idx - before_idx) / (after_idx - before_idx)
            smooth_weight = (1 - math.cos(linear_weight * math.pi)) / 2  # 코사인 보간
            
            interpolated = []
            for i in range(4):  # x1, y1, x2, y2
                interpolated_val = before_coord[i] + smooth_weight * (after_coord[i] - before_coord[i])
                interpolated.append(int(interpolated_val))
            
            return tuple(interpolated)
        else:
            # 한쪽만 있는 경우 - 시간적 거리 고려
            if before_indices:
                # 이전 좌표만 있는 경우
                recent_idx = max(before_indices)
                if target_idx - recent_idx <= 5:  # 5프레임 이내면 사용
                    return coord_list[recent_idx]
            
            if after_indices:
                # 이후 좌표만 있는 경우
                next_idx = min(after_indices)
                if next_idx - target_idx <= 5:  # 5프레임 이내면 사용
                    return coord_list[next_idx]
            
            # 가장 가까운 좌표 사용 (최종 폴백)
            return coord_list[closest_idx]
        
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
        
        # 🔧 [CRITICAL FIX] 싱글 GPU와 동일한 순환 리스트 방식 적용
        # 멀티 GPU에서도 연속성 보장을 위해 순환 리스트 사용
        print(f"🔄 [COORD SYNC] 싱글 GPU와 동일한 순환 리스트 방식 적용")
        
        # 순환 리스트 생성 (싱글 GPU와 동일)
        frame_list_cycle = video_frames_serializable + video_frames_serializable[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        
        print(f"   - 원본 프레임 수: {len(video_frames_serializable)}개")
        print(f"   - 순환 프레임 수: {len(frame_list_cycle)}개")
        print(f"   - 원본 좌표 수: {len(coord_list)}개")
        print(f"   - 순환 좌표 수: {len(coord_list_cycle)}개")
        
        # 모든 세그먼트를 작업 큐에 추가
        task_count = 0
        for segment in segments:
            gpu_id = segment['gpu_id']
            
            # 해당 세그먼트에 대응하는 비디오 프레임 계산 (순환 리스트 기반)
            start_frame = int(segment['start_time'] * model_config['fps'])
            end_frame = int(segment['end_time'] * model_config['fps'])
            
            # 🔧 [CRITICAL FIX] 순환 리스트 기반 프레임 추출 (싱글 GPU와 동일)
            segment_frames = []
            segment_coords = []
            total_cycle_frames = len(frame_list_cycle)
            video_duration = len(video_frames_serializable) / model_config['fps']  # 실제 비디오 길이 (초)
            
            print(f"   🎬 [COORD MAPPING] 세그먼트 {segment['segment_id']}: {segment['start_time']:.1f}-{segment['end_time']:.1f}초")
            print(f"      - 비디오 길이: {video_duration:.1f}초 ({len(video_frames_serializable)}프레임)")
            print(f"      - 요청 프레임 범위: {start_frame}-{end_frame-1}")
            
            # 좌표 보간을 위한 유효한 좌표 인덱스 수집 (강사 움직임 추적 안정성 향상)
            valid_coord_indices = []
            for idx in range(min(len(video_frames_serializable), end_frame)):
                if idx < len(coord_list) and coord_list[idx] is not None:
                    valid_coord_indices.append(idx)
            
            # 🔧 [CRITICAL FIX] 순환 리스트 기반 세그먼트 범위 처리 (싱글 GPU와 동일)
            for frame_idx in range(start_frame, end_frame):
                # 순환 리스트에서 프레임과 좌표 추출 (연속성 보장)
                cycle_idx = frame_idx % total_cycle_frames
                segment_frames.append(frame_list_cycle[cycle_idx])
                segment_coords.append(coord_list_cycle[cycle_idx])
                
                # 디버깅: 처음 몇 개와 마지막 몇 개 프레임의 매핑 정보 출력
                if (frame_idx - start_frame < 3) or (frame_idx >= end_frame - 3):
                    original_idx = cycle_idx % len(video_frames_serializable)
                    is_reversed = cycle_idx >= len(video_frames_serializable)
                    print(f"      🔄 프레임 {frame_idx}: 순환[{cycle_idx}] -> 원본[{original_idx}] {'(역순)' if is_reversed else '(정순)'}")
            
            # 오버랩 처리 제거 (정확한 동기화 우선)
            
            # 좌표 매핑 검증 (강사 강의 영상 품질 보장)
            valid_coords = [coord for coord in segment_coords if coord is not None]
            if len(valid_coords) > 0:
                print(f"      ✅ 유효한 좌표: {len(valid_coords)}/{len(segment_coords)}개")
                # 첫 번째와 마지막 좌표 샘플 출력
                first_coord = segment_coords[0][:2] if segment_coords[0] else 'None'
                last_coord = segment_coords[-1][:2] if segment_coords[-1] else 'None'
                print(f"      📍 좌표 범위: {first_coord} → {last_coord}")
            else:
                print(f"      ❌ 경고: 유효한 좌표가 없음, 기본값 사용 필요")
            
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
    
    def process_with_single_gpu_style(self, audio_path: str, video_frames: List[np.ndarray],
                                     coord_list: List, model_config: Dict) -> List[np.ndarray]:
        """
        🎯 [REVOLUTIONARY] 단일 GPU 방식을 멀티 GPU에 완전 모방
        
        핵심 혁신:
        1. 세그먼트 분할 완전 제거 - 전체 비디오를 연속적으로 처리
        2. 통합 VAE 인코딩 - 모든 프레임을 동일한 컨텍스트에서 인코딩
        3. 배치 단위 분산 - GPU들이 배치만 나눠서 처리
        4. 시간적 일관성 완벽 유지 - 첫 프레임부터 마지막까지 연속성 보장
        
        Args:
            audio_path: 오디오 파일 경로
            video_frames: 비디오 프레임 리스트  
            coord_list: 좌표 리스트
            model_config: 모델 설정
            
        Returns:
            List[np.ndarray]: 단일 GPU 품질의 처리된 프레임 리스트
        """
        print(f"🚀 ========== 단일 GPU 스타일 멀티 GPU 처리 시작 ==========")
        print(f"   혁신적 접근: 세그먼트 분할 없이 배치만 분산")
        print(f"   목표: 단일 GPU 품질 + 멀티 GPU 속도")
        print(f"=======================================================")
        
        # 1. 전체 오디오를 한 번에 처리 (세그먼트 분할 없음)
        print(f"🎵 [Unified] 전체 오디오 통합 처리...")
        try:
            # 먼저 모델들을 로드해야 함 (메인 GPU에서)
            from musetalk.utils.utils import load_all_model
            device = torch.device("cuda:0")
            weight_dtype = torch.float16
            
            # 모델 로드
            print(f"🔧 [Unified] 메인 GPU에서 모델 로딩...")
            vae, unet, pe = load_all_model(
                model_config['unet_model_path'],
                model_config['vae_type'], 
                model_config['unet_config'],
                device
            )
            
            # 🔧 [데이터 타입 통일] 모든 모델을 float16으로 변환
            # VAE는 커스텀 래퍼 클래스이므로 내부 vae 모델을 직접 변환
            vae.vae = vae.vae.to(device, dtype=weight_dtype)
            vae._use_float16 = True  # float16 사용 플래그 설정
            
            # UNet도 커스텀 래퍼 클래스이므로 내부 model을 직접 변환
            unet.model = unet.model.to(device, dtype=weight_dtype)
            
            # PE는 표준 PyTorch 모듈
            pe = pe.to(device, dtype=weight_dtype)
            
            # Whisper 모델 로드 (올바른 클래스 및 데이터 타입 사용)
            from transformers import WhisperModel
            whisper = WhisperModel.from_pretrained(model_config.get('whisper_path', 'openai/whisper-tiny')).to(device, dtype=weight_dtype)
            
            # 오디오 특징 추출
            audio_processor = AudioProcessor()
            whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
            
            # 전체 오디오의 whisper 특징을 프레임 단위로 분할
            whisper_chunks = audio_processor.get_whisper_chunk(
                whisper_input_features,  # 변환된 음성 특징들
                device,                  # GPU 디바이스
                weight_dtype,            # 데이터 타입
                whisper,                 # Whisper 모델
                librosa_length,          # 전체 음성 길이
                fps=model_config.get('fps', 50),  # 비디오 프레임 레이트
                audio_padding_length_left=2,     # 왼쪽 패딩
                audio_padding_length_right=2,    # 오른쪽 패딩
            )
            
            # 🔧 [P2P FIX] Whisper 청크들을 CPU로 이동하여 P2P 오류 방지
            if isinstance(whisper_chunks, list):
                whisper_chunks = [chunk.cpu() if hasattr(chunk, 'cpu') else chunk for chunk in whisper_chunks]
            elif hasattr(whisper_chunks, 'cpu'):
                whisper_chunks = whisper_chunks.cpu()
            print(f"   - 전체 Whisper 청크: {len(whisper_chunks)}개")
            print(f"   - 비디오 프레임: {len(video_frames)}개")
            print(f"   - 프레임-청크 비율: {len(whisper_chunks)}/{len(video_frames)} = {len(whisper_chunks)/len(video_frames):.3f}")
        except Exception as e:
            print(f"❌ [Unified] 오디오 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        # 2. 전체 비디오 프레임을 한 번에 VAE 인코딩 (단일 GPU 방식)
        print(f"🖼️  [Unified] 전체 프레임 통합 VAE 인코딩...")
        
        # 순환 리스트 생성 (단일 GPU와 동일)
        frame_list_cycle = video_frames + video_frames[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        
        # 모든 프레임을 동일한 기준으로 VAE 인코딩
        all_input_latents = []
        for i, frame in enumerate(video_frames):
            # 순환 좌표 사용
            cycle_idx = i % len(coord_list_cycle)
            bbox = coord_list_cycle[cycle_idx]
            
            if bbox is not None and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                if x2 > x1 and y2 > y1:
                    extra_margin = model_config.get('extra_margin', 10)
                    y2 = min(y2 + extra_margin, frame.shape[0])
                    crop_frame = frame[y1:y2, x1:x2]
                    crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                else:
                    crop_frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            else:
                crop_frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            # 🔧 [8-CHANNEL FIX] 단일 GPU와 동일하게 get_latents_for_unet 사용
            with torch.no_grad():
                # 단일 GPU와 동일한 방식으로 8채널 latent 생성
                latent_8ch = vae.get_latents_for_unet(crop_frame)  # [1, 8, 32, 32]
                all_input_latents.append(latent_8ch.cpu())
        
        print(f"   - 통합 VAE 인코딩 완료: {len(all_input_latents)}개 latent")
        
        # 3. 배치 단위로 GPU들에게 분산 처리
        print(f"🔄 [BatchDistribution] 배치 단위 GPU 분산 처리...")
        batch_size = model_config.get('batch_size', 8)
        total_batches = (len(whisper_chunks) + batch_size - 1) // batch_size
        
        print(f"   - 배치 크기: {batch_size}")
        print(f"   - 총 배치 수: {total_batches}")
        print(f"   - 사용 GPU: {self.num_gpus}개")
        
        # 배치들을 GPU들에게 동적 할당
        batch_tasks = []
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(whisper_chunks))
            
            batch_whisper = whisper_chunks[start_idx:end_idx]
            batch_latents = all_input_latents[start_idx:end_idx]
            
            task = {
                'batch_id': batch_idx,
                'whisper_chunks': batch_whisper,
                'input_latents': batch_latents,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'config': model_config
            }
            batch_tasks.append(task)
        
        # 4. GPU 워커들이 배치 작업을 동적으로 처리
        print(f"⚡ [DynamicBatch] {self.num_gpus}개 GPU가 {len(batch_tasks)}개 배치를 동적 처리...")
        
        # 배치 작업을 큐에 추가
        for task in batch_tasks:
            self.task_queue.put(task)
        
        # 결과 수집
        batch_results = {}
        completed_batches = 0
        
        while completed_batches < len(batch_tasks):
            try:
                result = self.result_queue.get(timeout=300)
                if result['success']:
                    batch_results[result['batch_id']] = result
                    completed_batches += 1
                    print(f"   ✅ 배치 {result['batch_id']} 완료 ({completed_batches}/{len(batch_tasks)})")
                else:
                    print(f"   ❌ 배치 {result['batch_id']} 실패: {result.get('error', 'Unknown')}")
                    return []
            except:
                print(f"   ⏰ 배치 처리 타임아웃")
                return []
        
        # 5. 결과 정렬 및 병합 (시간 순서 보장)
        print(f"🔗 [Merge] 배치 결과를 시간 순서대로 병합...")
        final_frames = []
        
        for batch_id in range(len(batch_tasks)):
            if batch_id in batch_results:
                batch_frames = batch_results[batch_id]['processed_frames']
                final_frames.extend(batch_frames)
                print(f"   📦 배치 {batch_id}: {len(batch_frames)}개 프레임 병합")
        
        print(f"🎉 [Success] 단일 GPU 스타일 멀티 GPU 처리 완료!")
        print(f"   - 총 처리 프레임: {len(final_frames)}개")
        print(f"   - 세그먼트 분할: 없음 (완전 연속)")
        print(f"   - VAE 일관성: 100% 보장")
        
        return final_frames
    
    def process_with_full_dynamic_queue(self, audio_path: str, video_frames: List[np.ndarray],
                                       coord_list: List, model_config: Dict) -> List[np.ndarray]:
        """
        🎯 [SINGLE GPU STYLE] 단일 GPU 방식을 멀티 GPU에 완전히 모방
        
        핵심 변경사항:
        - 세그먼트 분할 제거: 전체 비디오를 한 번에 처리
        - 전체 컨텍스트 유지: 모든 프레임이 서로 연관성을 가짐
        - 배치만 분산: 각 GPU가 배치 단위로만 작업 분담
        - 단일 VAE 인코딩: 모든 프레임을 동일한 기준으로 인코딩
        
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
        
        # 🎯 [GLOBAL REFERENCE] 첫 세그먼트 기준으로 전역 일관성 확보
        print(f"🎯 [GLOBAL REFERENCE] 첫 세그먼트 기준 설정으로 덜덜거림 완전 제거")
        
        # 첫 번째 세그먼트의 첫 번째 프레임을 전역 기준으로 설정
        if len(video_frames_serializable) > 0:
            global_reference_frame = video_frames_serializable[0].copy()
            # 전역 기준 프레임의 통계 계산
            ref_mean = np.mean(global_reference_frame, axis=(0, 1))
            ref_std = np.std(global_reference_frame, axis=(0, 1))
            model_config['global_reference_frame'] = global_reference_frame
            model_config['global_ref_mean'] = ref_mean
            model_config['global_ref_std'] = ref_std
            print(f"   - 전역 기준 프레임 설정 완료: 평균 {ref_mean}, 표준편차 {ref_std}")
        
        # 🔧 [CRITICAL FIX] 싱글 GPU와 동일한 순환 리스트 방식 적용
        # 완전 동적 처리에서도 연속성 보장을 위해 순환 리스트 사용
        print(f"🔄 [FULL DYNAMIC COORD SYNC] 싱글 GPU와 동일한 순환 리스트 방식 적용")
        
        # 순환 리스트 생성 (싱글 GPU와 동일)
        frame_list_cycle = video_frames_serializable + video_frames_serializable[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        
        print(f"   - 원본 프레임 수: {len(video_frames_serializable)}개")
        print(f"   - 순환 프레임 수: {len(frame_list_cycle)}개")
        print(f"   - 원본 좌표 수: {len(coord_list)}개")
        print(f"   - 순환 좌표 수: {len(coord_list_cycle)}개")
        
        # 3. 모든 세그먼트를 동적 작업 큐에 추가
        print(f"📋 [FullDynamic] 모든 세그먼트를 동적 큐에 추가 중...")
        
        for segment in segments:
            # 해당 세그먼트에 대응하는 비디오 프레임 계산 (순환 리스트 기반)
            start_frame = int(segment['start_time'] * model_config['fps'])
            end_frame = int(segment['end_time'] * model_config['fps'])
            
            # 🔧 [CRITICAL FIX] 순환 리스트 기반 프레임 추출 (싱글 GPU와 동일)
            segment_frames = []
            segment_coords = []
            total_cycle_frames = len(frame_list_cycle)
            video_duration = len(video_frames_serializable) / model_config['fps']  # 실제 비디오 길이 (초)
            
            # 세그먼트별 상세 매핑 정보 출력 (처음 5개와 마지막 5개만)
            if segment['segment_id'] < 5 or segment['segment_id'] >= len(segments) - 5:
                print(f"   🎬 [DYNAMIC COORD] 세그먼트 {segment['segment_id']}: {segment['start_time']:.1f}-{segment['end_time']:.1f}초")
                print(f"      - 비디오 길이: {video_duration:.1f}초 ({len(video_frames_serializable)}프레임)")
                print(f"      - 요청 프레임 범위: {start_frame}-{end_frame-1}")
            
            # 좌표 보간을 위한 유효한 좌표 인덱스 수집 (강사 움직임 추적 안정성 향상)
            valid_coord_indices = []
            for idx in range(min(len(video_frames_serializable), end_frame)):
                if idx < len(coord_list) and coord_list[idx] is not None:
                    valid_coord_indices.append(idx)
            
            # 🔧 [CRITICAL FIX] 순환 리스트 기반 세그먼트 범위 처리 (싱글 GPU와 동일)
            for frame_idx in range(start_frame, end_frame):
                # 순환 리스트에서 프레임과 좌표 추출 (연속성 보장)
                cycle_idx = frame_idx % total_cycle_frames
                segment_frames.append(frame_list_cycle[cycle_idx])
                segment_coords.append(coord_list_cycle[cycle_idx])
                
                # 디버깅: 처음 몇 개와 마지막 몇 개 세그먼트의 매핑 정보만 출력
                if (segment['segment_id'] < 5 or segment['segment_id'] >= len(segments) - 5) and \
                   ((frame_idx - start_frame < 3) or (frame_idx >= end_frame - 3)):
                    original_idx = cycle_idx % len(video_frames_serializable)
                    is_reversed = cycle_idx >= len(video_frames_serializable)
                    print(f"      🔄 프레임 {frame_idx}: 순환[{cycle_idx}] -> 원본[{original_idx}] {'(역순)' if is_reversed else '(정순)'}")
            
            # 오버랩 처리 제거 (정확한 동기화 우선)
            
            # 좌표 매핑 검증 (강사 강의 영상 품질 보장) - 상세 로그는 일부만
            valid_coords = [coord for coord in segment_coords if coord is not None]
            if segment['segment_id'] < 5 or segment['segment_id'] >= len(segments) - 5:
                if len(valid_coords) > 0:
                    print(f"      ✅ 유효한 좌표: {len(valid_coords)}/{len(segment_coords)}개")
                    # 첫 번째와 마지막 좌표 샘플 출력
                    first_coord = segment_coords[0][:2] if segment_coords[0] else 'None'
                    last_coord = segment_coords[-1][:2] if segment_coords[-1] else 'None'
                    print(f"      📍 좌표 범위: {first_coord} → {last_coord}")
                else:
                    print(f"      ❌ 경고: 유효한 좌표가 없음, 기본값 사용 필요")
            
            # 작업 패키지 생성 (GPU 할당 없음)
            # 세그먼트의 실제 길이 정보를 config에 추가 (강사 강의 영상 처리를 위한 정확한 시간 동기화)
            segment_duration = segment['end_time'] - segment['start_time']
            task_config = model_config.copy()
            task_config['segment_duration'] = segment_duration  # 실제 세그먼트 길이 (초)
            task_config['start_frame_global'] = start_frame  # 🔧 [CRITICAL FIX] 전역 프레임 인덱스 추가
            
            # 🔗 [OVERLAP FOR STABILITY] 덜덜거림 제거를 위한 오버랩 재활성화
            if segment['segment_id'] > 0:  # 첫 번째 세그먼트가 아닌 경우
                task_config['has_overlap'] = True
                task_config['overlap_frames'] = 5  # 적당한 오버랩으로 부드러운 전환
            else:
                task_config['has_overlap'] = False
                task_config['overlap_frames'] = 0
            
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
                processed_frames = result['processed_frames']
                final_frames.extend(processed_frames)
                print(f"   ✅ 세그먼트 {segment_id}: {len(processed_frames)}개 프레임 병합")
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
        
        # 🔧 [데이터 타입 통일] 모든 모델을 동일한 디바이스와 데이터 타입으로 변환
        self.pe = self.pe.to(self.device, dtype=self.weight_dtype)
        self.vae.vae = self.vae.vae.to(self.device, dtype=self.weight_dtype)
        self.unet.model = self.unet.model.to(self.device, dtype=self.weight_dtype)
        
        # Whisper 모델 로딩
        whisper_path = model_paths.get('whisper_path', "openai/whisper-tiny")
        print(f"🔧 [GPU {self.device.index} Worker] Whisper 모델 로딩 시작...")
        print(f"   - whisper_path: {whisper_path}")
        
        try:
            from transformers import WhisperForConditionalGeneration
            self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_path)
            self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
            for param in self.whisper.parameters():
                param.requires_grad = False
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
            
            # 🔗 [SEGMENT CONTINUITY] 이전 세그먼트와의 연속성 보장
            # 첫 번째 세그먼트가 아닌 경우, 이전 세그먼트의 마지막 프레임 참조
            prev_frame_context = task.get('prev_frame_context', None)
            if prev_frame_context is not None:
                print(f"      🔗 [GPU {gpu_id}] 이전 세그먼트 맥락 사용 (연속성 보장)")
                config['prev_frame_context'] = prev_frame_context
            
            # 세그먼트별 좌표 리스트 사용 (강사 강의 영상의 정확한 얼굴 위치 매핑)
            segment_coords = task.get('coord_list', [])
            processed_frames = self._process_frames(
                audio_features, video_frames, config, segment_coords
            )
            
            print(f"   ✅ [GPU {gpu_id} Worker] 세그먼트 {segment_id} 처리 완료:")
            print(f"      - 생성된 프레임: {len(processed_frames)}개")
            print(f"      - GPU 메모리 사용량: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f}GB")
            
            # 세그먼트 처리 완료 후 메모리 정리 (다음 작업을 위해)
            torch.cuda.empty_cache()
            
            # 🎯 [GLOBAL CONSISTENCY] 전역 기준 프레임으로 일관성 보장
            enhanced_frames = self._apply_global_consistency(processed_frames, config, segment_id)
            
            # 🔧 [OVERLAP PROCESSING] 오버랩 처리를 위한 후처리
            # 첫 번째 세그먼트가 아닌 경우, 오버랩 프레임 제거
            final_processed_frames = enhanced_frames
            has_overlap = config.get('has_overlap', False)
            overlap_frames = config.get('overlap_frames', 0)
            
            if has_overlap and overlap_frames > 0 and len(enhanced_frames) > overlap_frames:
                # 오버랩 프레임 제거 (중복 방지)
                final_processed_frames = enhanced_frames[overlap_frames:]
                print(f"   🔗 [GPU {gpu_id} Worker] 오버랩 프레임 제거: {len(enhanced_frames)} -> {len(final_processed_frames)}개")
            
            # 🔗 [SEGMENT CONTINUITY] 다음 세그먼트를 위한 마지막 프레임 맥락 저장
            last_frame_context = None
            if len(final_processed_frames) > 0:
                last_frame_context = final_processed_frames[-1].copy()  # 마지막 프레임 복사
            
            # 오버랩 정보 포함하여 결과 반환 (연속성 보장)
            result = {
                'segment_id': segment_id,
                'gpu_id': self.physical_gpu_id,  # 실제 물리 GPU ID 사용 (워커에서는 device.index가 항상 0)
                'processed_frames': final_processed_frames,
                'last_frame_context': last_frame_context,  # 다음 세그먼트를 위한 맥락
                'success': True,
                'processing_time': time.time(),
                'has_overlap': has_overlap,
                'overlap_frames_removed': overlap_frames if has_overlap else 0
            }
            
            # 🔧 [OVERLAP REMOVED] 오버랩 처리는 워커에서 수행됨
            
            return result
            
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
    
    def process_batch(self, task: Dict) -> Dict:
        """
        배치 처리 (새로운 단일 GPU 스타일 멀티 GPU 방식)
        
        Args:
            task: 배치 작업 정보
                - batch_id: 배치 ID
                - whisper_chunks: Whisper 청크들
                - input_latents: VAE 인코딩된 latent들
                - config: 모델 설정
                
        Returns:
            Dict: 처리 결과
        """
        try:
            batch_id = task['batch_id']
            whisper_chunks = task['whisper_chunks']
            input_latents = task['input_latents']
            config = task['config']
            
            gpu_id = self.physical_gpu_id
            print(f"🎬 [GPU {gpu_id} Worker] 배치 {batch_id} 처리 시작:")
            print(f"   - Whisper 청크: {len(whisper_chunks)}개")
            print(f"   - Input latent: {len(input_latents)}개")
            
            processed_frames = []
            
            # 🔧 [BATCH FIX] 단일 GPU 방식과 동일하게 배치 처리
            with torch.no_grad():
                # 1. 모든 whisper 청크를 배치로 변환
                whisper_batch = []
                for whisper_chunk in whisper_chunks:
                    if isinstance(whisper_chunk, np.ndarray):
                        whisper_tensor = torch.from_numpy(whisper_chunk).to(self.device, dtype=self.weight_dtype)
                    elif hasattr(whisper_chunk, 'to'):
                        whisper_tensor = whisper_chunk.cpu().to(self.device, dtype=self.weight_dtype)
                    else:
                        whisper_tensor = torch.tensor(whisper_chunk).to(self.device, dtype=self.weight_dtype)
                    whisper_batch.append(whisper_tensor)
                
                whisper_batch = torch.stack(whisper_batch)
                
                # 2. 모든 input latent를 배치로 변환
                latent_batch = []
                for input_latent in input_latents:
                    latent = input_latent.cpu().to(self.device, dtype=self.weight_dtype)
                    latent_batch.append(latent)
                
                latent_batch = torch.cat(latent_batch, dim=0)
                
                # 3. 오디오 특징을 위치 인코딩 (단일 GPU와 동일)
                audio_feature_batch = self.pe(whisper_batch)
                
                # 4. UNet 추론 (단일 GPU와 동일한 방식)
                pred_latents = self.unet.model(
                    latent_batch, 
                    self.timesteps, 
                    encoder_hidden_states=audio_feature_batch
                ).sample
                
                # 5. VAE 디코딩 (단일 GPU와 동일)
                decoded_frames = self.vae.vae.decode(pred_latents / self.vae.vae.config.scaling_factor).sample
                
                # 6. 배치 결과를 개별 프레임으로 분리
                decoded_frames = (decoded_frames / 2 + 0.5).clamp(0, 1)
                decoded_frames = decoded_frames.cpu().permute(0, 2, 3, 1).numpy()
                
                for decoded_frame in decoded_frames:
                    decoded_frame = (decoded_frame * 255).astype(np.uint8)
                    processed_frames.append(decoded_frame)
            
            print(f"   ✅ [GPU {gpu_id} Worker] 배치 {batch_id} 처리 완료:")
            print(f"      - 생성된 프레임: {len(processed_frames)}개")
            print(f"      - GPU 메모리 사용량: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f}GB")
            
            # 메모리 정리
            torch.cuda.empty_cache()
            
            return {
                'batch_id': batch_id,
                'gpu_id': self.physical_gpu_id,
                'processed_frames': processed_frames,
                'success': True,
                'processing_time': time.time()
            }
            
        except Exception as e:
            gpu_id = self.physical_gpu_id
            print(f"💥 [GPU {gpu_id} Worker] 배치 처리 오류: {e}")
            import traceback
            traceback.print_exc()
            return {
                'batch_id': task.get('batch_id', -1),
                'gpu_id': self.physical_gpu_id,
                'error': str(e),
                'success': False
            }
            
    def _process_frames(self, audio_features: torch.Tensor, 
                       video_frames: List[np.ndarray], 
                       config: Dict, segment_coords: List = None) -> List[np.ndarray]:
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
        
        # 오디오 특징을 리스트 형태로 변환 (세그먼트별 처리)
        whisper_input_features = [audio_features]
        
        # 실제 세그먼트 오디오 길이를 config에서 가져오기 (초 단위)
        segment_duration = config.get('segment_duration', 5.0)  # 기본값 5초
        librosa_length = int(segment_duration * 16000)  # 샘플 수로 변환 (16kHz 기준)
        
        # 🎨 [NATURAL AUDIO] 자연스러운 오디오 처리를 위한 개선사항
        # 오디오 특징에 시간적 가중치 적용으로 더 자연스러운 립싱크
        audio_features = self._enhance_audio_features_for_naturalness(audio_features, config)
        
        print(f"         - 오디오 특징 크기: {audio_features.shape}")
        print(f"         - 세그먼트 길이: {segment_duration}초")
        print(f"         - 비디오 프레임 수: {len(video_frames)}개")
        print(f"         - 오디오 길이 (샘플): {librosa_length:,}개")
        print(f"         - 오디오 길이 (초): {librosa_length/16000:.2f}초")
        
        # 🔧 [FRAME-AUDIO SYNC] 비디오 프레임 수에 맞춰 오디오 길이 조정
        # 실제 비디오 프레임 수에 정확히 맞춰서 Whisper 청크 생성
        video_frame_count = len(video_frames)
        video_duration_actual = video_frame_count / config['fps']  # 실제 비디오 길이 (초)
        final_librosa_length = int(video_duration_actual * 16000)  # 실제 프레임 수에 맞춘 오디오 길이
        
        print(f"         - 원본 오디오 길이: {librosa_length:,}샘플")
        print(f"         - 프레임 동기화 길이: {final_librosa_length:,}샘플 (프레임: {video_frame_count}개)")
        print(f"         - 실제 세그먼트 길이: {video_duration_actual:.3f}초")
        
        # Whisper 청크 생성
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            final_librosa_length,
            fps=config['fps'],  # 실제 비디오 fps 전달
            audio_padding_length_left=config.get('audio_padding_length_left', 0),
            audio_padding_length_right=config.get('audio_padding_length_right', 0),
        )
        
        print(f"         - 생성된 Whisper 청크 수: {len(whisper_chunks)}")
        print(f"         - 프레임-청크 비율: {len(whisper_chunks)}/{len(video_frames)} = {len(whisper_chunks)/len(video_frames):.3f}")
        
        # AudioProcessor 사용 후 메모리 정리
        del audio_processor, whisper_input_features
        torch.cuda.empty_cache()
        
        # 2. 비디오 프레임을 VAE 잠재 공간으로 변환
        print(f"      🖼️  [GPU {gpu_id}] 비디오 프레임 처리 시작...")
        
        # 🔧 [CRITICAL FIX] 세그먼트별 좌표에 맞는 얼굴 크롭 적용
        # 단일 GPU와 동일하게 각 프레임별로 해당 좌표를 사용해서 latent 생성
        input_latent_list = []
        
        # 세그먼트별 좌표 사용 (강사 강의 영상의 정확한 얼굴 매핑)
        coord_list = segment_coords if segment_coords is not None else config.get('coord_list', [])
        
        print(f"      🎯 [GPU {gpu_id}] 세그먼트별 좌표 기반 VAE 인코딩 시작")
        print(f"         - 세그먼트 프레임 수: {len(video_frames)}")
        print(f"         - 세그먼트 좌표 수: {len(coord_list)}")
        
        for i, frame in enumerate(video_frames):
            if i < len(coord_list):
                # 좌표가 있는 경우 얼굴 영역 크롭 (단일 GPU와 동일한 방식)
                bbox = coord_list[i]
                if bbox is not None and len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    
                    # 좌표 유효성 검증
                    if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                        # 디버깅: 처음 3개와 마지막 3개 프레임의 좌표 정보 출력
                        if i < 3 or i >= len(video_frames) - 3:
                            print(f"         - 프레임 {i}: 세그먼트 좌표 ({x1}, {y1}, {x2}, {y2})")
                        
                        # 🔧 [SIMPLE CROP] 단일 GPU와 동일한 단순한 얼굴 크롭
                        extra_margin = config.get('extra_margin', 10)
                        y2 = y2 + extra_margin
                        y2 = min(y2, frame.shape[0])
                        
                        # 단순한 얼굴 크롭 (단일 GPU와 동일)
                        crop_frame = frame[y1:y2, x1:x2]
                        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                    else:
                        # 잘못된 좌표인 경우 전체 프레임 사용
                        if i < 3 or i >= len(video_frames) - 3:
                            print(f"         - 프레임 {i}: 잘못된 좌표, 전체 프레임 사용")
                        crop_frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                else:
                    # 좌표가 없는 경우 전체 프레임 사용
                    if i < 3 or i >= len(video_frames) - 3:
                        print(f"         - 프레임 {i}: 좌표 없음, 전체 프레임 사용")
                    crop_frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            else:
                # 좌표 리스트가 부족한 경우 전체 프레임 사용
                if i < 3 or i >= len(video_frames) - 3:
                    print(f"         - 프레임 {i}: 좌표 리스트 부족, 전체 프레임 사용")
                crop_frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            # 🎯 [NATURAL VAE ENCODING] 원본 프레임 그대로 사용하여 자연스러운 인코딩
            # 정규화 없이 원본 crop_frame을 직접 사용 (단일 GPU와 동일한 방식)
            
            # 🎯 [VAE ENCODING] 세그먼트별 정확한 latent 생성
            latents = self.vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)
        
        print(f"      ✅ [GPU {gpu_id}] 세그먼트별 VAE 인코딩 완료: {len(input_latent_list)}개 latent 생성")
        
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
            
            # 🎨 [NATURAL BATCH] 배치 단위로 자연스러운 처리 적용
            # 배치 내 프레임 간 연속성도 고려
            if batch_count > 1 and len(processed_frames) > 0:
                # 이전 배치의 마지막 프레임과의 연속성 고려
                prev_frame_context = processed_frames[-1] if processed_frames else None
            else:
                prev_frame_context = None
            
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
                
                # 🎨 [NATURAL FRAMES] 배치 결과에 자연스러운 후처리 적용
                batch_frames = list(recon)
                
                # 이전 배치와의 연속성 보장
                if prev_frame_context is not None and len(batch_frames) > 0:
                    # 첫 번째 프레임에 이전 배치와의 블렌딩 적용
                    first_frame = batch_frames[0]
                    blended_first = (
                        0.3 * prev_frame_context.astype(np.float32) +
                        0.7 * first_frame.astype(np.float32)
                    ).astype(np.uint8)
                    batch_frames[0] = blended_first
                
                # 배치 내 프레임들 추가
                batch_frames_count = 0
                for frame in batch_frames:
                    processed_frames.append(frame)
                    batch_frames_count += 1
                
                # 배치 처리 후 즉시 메모리 정리 (OOM 방지)
                del pred_latents, recon
                torch.cuda.empty_cache()
                
                # print(f"         ✅ 배치 {batch_count} 완료: {batch_frames_count}개 프레임 생성")
                
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
                        
                        # 안전 모드에서도 연속성 보장
                        frame = single_recon[0]
                        if len(processed_frames) > 0:
                            prev_frame = processed_frames[-1]
                            # 간단한 블렌딩으로 연속성 보장
                            frame = (
                                0.1 * prev_frame.astype(np.float32) +
                                0.9 * frame.astype(np.float32)
                            ).astype(np.uint8)
                        
                        processed_frames.append(frame)
                        
                        # 각 프레임 처리 후 메모리 정리
                        del single_latent, single_audio, single_pred, single_recon
                        torch.cuda.empty_cache()
                        
                    except Exception as single_e:
                        print(f"         ❌ [GPU {gpu_id}] 개별 프레임 {i} 처리 실패: {single_e}")
                        # 실패한 프레임은 건너뛰고 계속 진행
                        continue
        
        print(f"      🎉 [GPU {gpu_id}] 모든 배치 처리 완료: 총 {len(processed_frames)}개 프레임")
        
        return processed_frames
    
    def _apply_natural_enhancements(self, frames: List[np.ndarray], config: Dict, segment_id: int) -> List[np.ndarray]:
        """
        자연스러운 립싱크를 위한 고급 후처리 시스템
        
        강사 강의 영상의 자연스러운 표현을 위한 개선사항:
        1. 시간적 스무딩 (프레임 간 부드러운 전환)
        2. 표정 연속성 보장 (급격한 변화 방지)
        3. 입술 움직임 자연성 향상
        4. 눈 깜박임 및 미세 표정 보정
        
        Args:
            frames: 처리된 프레임 리스트
            config: 설정 정보
            segment_id: 세그먼트 ID
            
        Returns:
            List[np.ndarray]: 자연스러운 후처리가 적용된 프레임 리스트
        """
        if len(frames) < 2:
            return frames
        
        gpu_id = self.device.index
        print(f"      🎨 [GPU {gpu_id}] 자연스러운 후처리 시작...")
        
        # 0. 밝기 일관성은 매우 부드럽게만 적용 (자연스러운 결과 우선)
        # brightness_consistent_frames = self._ensure_brightness_consistency(frames)
        brightness_consistent_frames = frames  # 원본 그대로 사용하여 자연스러움 우선
        
        # 1. 시간적 스무딩 적용
        smoothed_frames = self._apply_temporal_smoothing(brightness_consistent_frames)
        
        # 2. 표정 연속성 보장
        continuous_frames = self._ensure_expression_continuity(smoothed_frames)
        
        # 3. 입술 움직임 자연성 향상
        natural_frames = self._enhance_lip_movement_naturalness(continuous_frames)
        
        # 4. 미세 표정 보정 (눈 깜박임, 눈동자 움직임 등)
        final_frames = self._refine_micro_expressions(natural_frames)
        
        print(f"      ✨ [GPU {gpu_id}] 자연스러운 후처리 완료: {len(final_frames)}개 프레임")
        
        return final_frames
    
    def _ensure_brightness_consistency(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        🌟 [BRIGHTNESS CONSISTENCY] 세그먼트 간 밝기 및 색상 일관성 보장
        
        멀티 GPU 환경에서 세그먼트별로 다른 VAE 인코딩으로 인한
        밝기 변화와 색상 차이를 보정하여 단일 GPU와 동일한 결과를 생성
        
        Args:
            frames: 입력 프레임 리스트
            
        Returns:
            밝기가 일관된 프레임 리스트
        """
        if len(frames) < 2:
            return frames
        
        gpu_id = self.device.index
        print(f"         🌟 [GPU {gpu_id}] 밝기 일관성 보정 시작...")
        
        consistent_frames = []
        
        # 첫 번째 프레임을 기준으로 설정
        reference_frame = frames[0]
        consistent_frames.append(reference_frame)
        
        # 기준 프레임의 밝기 및 색상 통계 계산
        ref_mean = np.mean(reference_frame, axis=(0, 1))  # RGB 각 채널별 평균
        ref_std = np.std(reference_frame, axis=(0, 1))    # RGB 각 채널별 표준편차
        
        for i in range(1, len(frames)):
            current_frame = frames[i]
            
            # 현재 프레임의 밝기 및 색상 통계 계산
            curr_mean = np.mean(current_frame, axis=(0, 1))
            curr_std = np.std(current_frame, axis=(0, 1))
            
            # 🎯 [HISTOGRAM MATCHING] 히스토그램 매칭을 통한 색상 보정
            corrected_frame = current_frame.astype(np.float32)
            
            for channel in range(3):  # RGB 각 채널별로 처리
                # 표준편차가 0인 경우 방지
                if curr_std[channel] > 0:
                    # 정규화 후 기준 프레임의 분포로 조정
                    corrected_frame[:, :, channel] = (
                        (corrected_frame[:, :, channel] - curr_mean[channel]) / curr_std[channel]
                    ) * ref_std[channel] + ref_mean[channel]
                else:
                    # 표준편차가 0인 경우 평균만 조정
                    corrected_frame[:, :, channel] = (
                        corrected_frame[:, :, channel] - curr_mean[channel] + ref_mean[channel]
                    )
            
            # 픽셀 값 범위 제한 (0-255)
            corrected_frame = np.clip(corrected_frame, 0, 255).astype(np.uint8)
            
            # 🔗 [SMOOTH TRANSITION] 급격한 변화 방지를 위한 부드러운 전환
            # 이전 프레임과의 차이가 큰 경우 점진적 적용
            prev_frame = consistent_frames[-1]
            frame_diff = np.mean(np.abs(corrected_frame.astype(np.float32) - prev_frame.astype(np.float32)))
            
            if frame_diff > 15.0:  # 급격한 밝기 변화 감지
                # 🎨 [GENTLE CORRECTION] 색상 보존을 위한 부드러운 보정
                # 50% 보정 + 50% 원본으로 자연스러운 전환 (70% → 50%)
                blend_ratio = 0.5
                final_frame = (
                    blend_ratio * corrected_frame.astype(np.float32) +
                    (1 - blend_ratio) * current_frame.astype(np.float32)
                ).astype(np.uint8)
            else:
                # 작은 변화도 30% 원본 유지로 색상 보존
                final_frame = (
                    0.7 * corrected_frame.astype(np.float32) +
                    0.3 * current_frame.astype(np.float32)
                ).astype(np.uint8)
            
            consistent_frames.append(final_frame)
        
        print(f"         ✅ [GPU {gpu_id}] 밝기 일관성 보정 완료: {len(consistent_frames)}개 프레임")
        return consistent_frames
    
    def _normalize_for_vae_consistency(self, frame: np.ndarray) -> np.ndarray:
        """
        🎯 [GENTLE VAE CONSISTENCY] 색상 보존을 위한 부드러운 VAE 정규화
        
        강한 정규화로 인한 색상 왜곡(초록색 피부)을 방지하면서도
        세그먼트 간 VAE 인코딩 일관성을 확보합니다.
        
        Args:
            frame: 입력 프레임 (256x256x3)
            
        Returns:
            부드럽게 정규화된 프레임
        """
        # 🎨 [COLOR PRESERVATION] 원본 색상 보존을 위한 부드러운 정규화
        # 강한 정규화 대신 미세한 조정만 적용
        
        normalized_frame = frame.astype(np.float32)
        
        # 🔧 [MILD NORMALIZATION] 색상 왜곡 방지를 위한 약한 정규화
        # 전체 프레임의 밝기만 약간 조정 (채널별 강한 정규화 제거)
        overall_mean = np.mean(normalized_frame)
        target_mean = 127.5  # 중간 밝기 목표
        
        # 밝기 차이가 클 때만 미세 조정 적용
        brightness_diff = abs(overall_mean - target_mean)
        if brightness_diff > 30:  # 큰 차이가 있을 때만
            # 20% 정도만 목표 밝기로 조정 (80% 원본 유지)
            adjustment_factor = 0.2
            brightness_adjustment = (target_mean - overall_mean) * adjustment_factor
            normalized_frame = normalized_frame + brightness_adjustment
        
        # 픽셀 값 범위 제한
        normalized_frame = np.clip(normalized_frame, 0, 255).astype(np.uint8)
        
        return normalized_frame
    
    def _apply_global_consistency(self, frames: List[np.ndarray], config: Dict, segment_id: int) -> List[np.ndarray]:
        """
        🎯 [GLOBAL CONSISTENCY] 전역 기준 프레임으로 모든 세그먼트의 일관성 보장
        
        첫 번째 세그먼트의 기준을 모든 세그먼트에 적용하여
        세그먼트 간 덜덜거림을 완전히 제거합니다.
        
        Args:
            frames: 처리된 프레임 리스트
            config: 설정 (전역 기준 정보 포함)
            segment_id: 세그먼트 ID
            
        Returns:
            전역 기준으로 일관성이 보장된 프레임 리스트
        """
        if len(frames) == 0:
            return frames
        
        gpu_id = self.device.index
        
        # 첫 번째 세그먼트는 그대로 사용 (기준이 됨)
        if segment_id == 0:
            print(f"         🎯 [GPU {gpu_id}] 세그먼트 0: 전역 기준으로 사용 (변경 없음)")
            return frames
        
        # 전역 기준 정보 가져오기
        global_ref_mean = config.get('global_ref_mean')
        global_ref_std = config.get('global_ref_std')
        
        if global_ref_mean is None or global_ref_std is None:
            print(f"         ⚠️ [GPU {gpu_id}] 전역 기준 없음, 원본 그대로 사용")
            return frames
        
        print(f"         🎯 [GPU {gpu_id}] 세그먼트 {segment_id}: 전역 기준으로 일관성 보정 시작...")
        
        consistent_frames = []
        
        for i, frame in enumerate(frames):
            # 현재 프레임의 통계 계산
            curr_mean = np.mean(frame, axis=(0, 1))
            curr_std = np.std(frame, axis=(0, 1))
            
            # 🎯 [COLOR PRESERVING ADJUSTMENT] 색상 보존을 위한 부드러운 밝기 조정
            # RGB 채널별 강한 정규화 대신 전체 밝기만 조정하여 색상 왜곡 방지
            corrected_frame = frame.astype(np.float32)
            
            # 전체 밝기 차이만 계산 (색상 균형 유지)
            curr_brightness = np.mean(curr_mean)
            ref_brightness = np.mean(global_ref_mean)
            brightness_diff = ref_brightness - curr_brightness
            
            # 밝기 차이가 클 때만 조정 (색상 왜곡 최소화)
            if abs(brightness_diff) > 10:  # 임계값 설정
                # 모든 채널에 동일한 밝기 조정 적용 (색상 균형 유지)
                corrected_frame = corrected_frame + (brightness_diff * 0.5)  # 50% 적용
            
            # 픽셀 값 범위 제한
            corrected_frame = np.clip(corrected_frame, 0, 255).astype(np.uint8)
            
            # 🔗 [STRONG CONSISTENCY] 덜덜거림 제거를 위한 강화된 적용
            # 70% 보정 + 30% 원본으로 강한 일관성 확보 (30% → 70%)
            final_frame = (
                0.7 * corrected_frame.astype(np.float32) +
                0.3 * frame.astype(np.float32)
            ).astype(np.uint8)
            
            consistent_frames.append(final_frame)
        
        print(f"         ✅ [GPU {gpu_id}] 전역 기준 일관성 보정 완료: {len(consistent_frames)}개 프레임")
        return consistent_frames
    
    def _apply_temporal_smoothing(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        시간적 스무딩 적용 - 프레임 간 부드러운 전환
        
        강사 강의 영상에서 자주 발생하는 문제들:
        - 갑작스러운 입술 모양 변화
        - 프레임 간 일관성 부족
        - 인위적인 느낌의 전환
        
        해결 방법:
        - 가우시안 가중 평균으로 프레임 블렌딩
        - 전후 프레임의 영향력을 고려한 스무딩
        - 얼굴 영역별 차별화된 스무딩 강도
        """
        if len(frames) < 3:
            return frames
        
        smoothed_frames = []
        
        # 🔗 [SEGMENT BOUNDARY] 세그먼트 시작 부분 강화 스무딩
        # 첫 번째 프레임도 두 번째 프레임과 블렌딩하여 급격한 변화 완화
        if len(frames) >= 2:
            first_frame = frames[0].astype(np.float32)
            second_frame = frames[1].astype(np.float32)
            # 첫 프레임에 약간의 스무딩 적용 (세그먼트 경계 완화)
            blended_first = (0.8 * first_frame + 0.2 * second_frame).astype(np.uint8)
            smoothed_frames.append(blended_first)
        else:
            smoothed_frames.append(frames[0])
        
        # 중간 프레임들에 초강화된 스무딩 적용
        for i in range(1, len(frames) - 1):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]
            next_frame = frames[i + 1]
            
            # 🎨 [ULTRA SMOOTHING] 덜덜거림 완전 제거를 위한 초강화 스무딩
            # 가중치를 (0.25, 0.5, 0.25)로 조정하여 훨씬 더 부드러운 전환
            smoothed_frame = (
                0.25 * prev_frame.astype(np.float32) +
                0.5 * curr_frame.astype(np.float32) +
                0.25 * next_frame.astype(np.float32)
            ).astype(np.uint8)
            
            smoothed_frames.append(smoothed_frame)
        
        # 🔗 [SEGMENT BOUNDARY] 세그먼트 끝 부분 강화 스무딩  
        # 마지막 프레임도 이전 프레임과 블렌딩하여 다음 세그먼트와의 연결 준비
        if len(frames) >= 2:
            last_frame = frames[-1].astype(np.float32)
            second_last_frame = frames[-2].astype(np.float32)
            # 마지막 프레임에 약간의 스무딩 적용 (다음 세그먼트와의 연결성 향상)
            blended_last = (0.8 * last_frame + 0.2 * second_last_frame).astype(np.uint8)
            smoothed_frames.append(blended_last)
        else:
            smoothed_frames.append(frames[-1])
        
        return smoothed_frames
    
    def _ensure_expression_continuity(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        표정 연속성 보장 - 급격한 표정 변화 방지
        
        강사 강의 영상에서 중요한 요소:
        - 자연스러운 표정 변화
        - 갑작스러운 얼굴 경직 방지
        - 눈과 입 주변의 자연스러운 움직임
        
        개선 방법:
        - 프레임 간 차이 분석
        - 임계값 기반 변화 제한
        - 점진적 변화 유도
        """
        if len(frames) < 2:
            return frames
        
        continuous_frames = [frames[0]]  # 첫 번째 프레임
        
        for i in range(1, len(frames)):
            prev_frame = continuous_frames[-1]
            curr_frame = frames[i]
            
            # 프레임 간 차이 계산 (평균 픽셀 차이)
            diff = np.mean(np.abs(curr_frame.astype(np.float32) - prev_frame.astype(np.float32)))
            
            # 🔗 [MULTI-GPU CONTINUITY] 멀티 GPU 환경을 위한 더욱 강화된 연속성 보장
            # 세그먼트 경계에서 발생하는 급격한 변화를 더 적극적으로 완화
            threshold = 8.0  # 임계값을 더 낮춰서 훨씬 민감하게 감지 (12.0 → 8.0)
            
            if diff > threshold:
                # 🎨 [ULTRA ENHANCED BLENDING] 덜덜거림 완전 제거를 위한 초강화 블렌딩
                # 변화 정도에 따라 블렌딩 비율을 더욱 보수적으로 조정
                if diff > 20.0:  # 매우 급격한 변화
                    blend_ratio = 0.9  # 90% 이전 프레임, 10% 현재 프레임 (더 보수적)
                elif diff > 15.0:  # 중간 정도 변화
                    blend_ratio = 0.85  # 85% 이전 프레임, 15% 현재 프레임
                elif diff > 10.0:  # 약간의 변화
                    blend_ratio = 0.8  # 80% 이전 프레임, 20% 현재 프레임
                else:  # 미세한 변화
                    blend_ratio = 0.75  # 75% 이전 프레임, 25% 현재 프레임
                
                blended_frame = (
                    blend_ratio * prev_frame.astype(np.float32) +
                    (1 - blend_ratio) * curr_frame.astype(np.float32)
                ).astype(np.uint8)
                continuous_frames.append(blended_frame)
            else:
                # 자연스러운 변화도 약간의 스무딩 적용 (덜덜거림 완전 제거)
                light_blend = (
                    0.1 * prev_frame.astype(np.float32) +
                    0.9 * curr_frame.astype(np.float32)
                ).astype(np.uint8)
                continuous_frames.append(light_blend)
        
        return continuous_frames
    
    def _enhance_lip_movement_naturalness(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        입술 움직임 자연성 향상
        
        립싱크에서 가장 중요한 부분:
        - 입술의 자연스러운 열고 닫힘
        - 발음에 따른 입모양 변화
        - 입가 근육의 자연스러운 움직임
        
        개선 방법:
        - 입술 영역 집중 처리
        - 입모양 변화의 연속성 보장
        - 미세한 입술 디테일 보정
        """
        if len(frames) < 3:
            return frames
        
        enhanced_frames = []
        
        for i, frame in enumerate(frames):
            enhanced_frame = frame.copy()
            
            # 입술 영역에 추가 스무딩 적용 (하단 1/3 영역)
            height = frame.shape[0]
            lip_region_start = int(height * 0.67)  # 하단 33% 영역
            
            # 이전/다음 프레임과의 블렌딩으로 입술 움직임 부드럽게
            if i > 0 and i < len(frames) - 1:
                prev_lip = frames[i-1][lip_region_start:, :]
                curr_lip = frame[lip_region_start:, :]
                next_lip = frames[i+1][lip_region_start:, :]
                
                # 입술 영역에 더 강한 스무딩 (0.1, 0.8, 0.1)
                smoothed_lip = (
                    0.1 * prev_lip.astype(np.float32) +
                    0.8 * curr_lip.astype(np.float32) +
                    0.1 * next_lip.astype(np.float32)
                ).astype(np.uint8)
                
                enhanced_frame[lip_region_start:, :] = smoothed_lip
            
            enhanced_frames.append(enhanced_frame)
        
        return enhanced_frames
    
    def _refine_micro_expressions(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        미세 표정 보정 - 눈 깜박임, 눈동자 움직임 등
        
        자연스러운 표정을 위한 미세 조정:
        - 눈 주변 영역의 자연스러운 움직임
        - 갑작스러운 밝기 변화 방지
        - 전체적인 표정의 일관성 유지
        
        개선 방법:
        - 눈 주변 영역 안정화
        - 전체적인 밝기 일관성 보장
        - 미세한 노이즈 제거
        """
        if len(frames) < 2:
            return frames
        
        refined_frames = []
        
        for i, frame in enumerate(frames):
            refined_frame = frame.copy()
            
            # 눈 주변 영역 안정화 (상단 40% 영역)
            height = frame.shape[0]
            eye_region_end = int(height * 0.4)
            
            # 이전 프레임과의 차이가 큰 경우 안정화
            if i > 0:
                prev_eye = frames[i-1][:eye_region_end, :]
                curr_eye = frame[:eye_region_end, :]
                
                # 눈 주변 영역의 갑작스러운 변화 감지
                eye_diff = np.mean(np.abs(curr_eye.astype(np.float32) - prev_eye.astype(np.float32)))
                
                if eye_diff > 10.0:  # 눈 주변 변화 임계값
                    # 눈 주변 영역만 안정화 (90% 이전, 10% 현재)
                    stabilized_eye = (
                        0.9 * prev_eye.astype(np.float32) +
                        0.1 * curr_eye.astype(np.float32)
                    ).astype(np.uint8)
                    
                    refined_frame[:eye_region_end, :] = stabilized_eye
            
            # 전체적인 밝기 일관성 보장
            if i > 0:
                prev_brightness = np.mean(frames[i-1])
                curr_brightness = np.mean(refined_frame)
                brightness_diff = abs(curr_brightness - prev_brightness)
                
                # 갑작스러운 밝기 변화 보정
                if brightness_diff > 5.0:  # 밝기 변화 임계값
                    brightness_ratio = prev_brightness / max(curr_brightness, 1)
                    brightness_ratio = np.clip(brightness_ratio, 0.9, 1.1)  # 밝기 변화 제한
                    
                    refined_frame = np.clip(
                        refined_frame.astype(np.float32) * brightness_ratio,
                        0, 255
                    ).astype(np.uint8)
            
            refined_frames.append(refined_frame)
        
        return refined_frames
    
    def _enhance_audio_features_for_naturalness(self, audio_features: torch.Tensor, config: Dict) -> torch.Tensor:
        """
        자연스러운 립싱크를 위한 오디오 특징 개선
        
        강사 강의 영상에서 중요한 오디오 처리:
        1. 음성 강약 변화에 따른 입모양 조절
        2. 묵음 구간에서의 자연스러운 입모양 유지
        3. 발음 전환 시점에서의 부드러운 변화
        4. 강사의 말하기 패턴에 맞는 입술 움직임
        
        Args:
            audio_features: 원본 오디오 특징
            config: 설정 정보
            
        Returns:
            torch.Tensor: 개선된 오디오 특징
        """
        if audio_features.dim() < 2:
            return audio_features
        
        # 1. 시간적 스무딩 적용 (오디오 특징에도)
        # 갑작스러운 음성 변화를 부드럽게 만들어 입모양 전환도 자연스럽게
        smoothed_features = audio_features.clone()
        
        # 시간 차원에서 스무딩 (1D convolution 효과)
        if audio_features.shape[0] > 2:
            for i in range(1, audio_features.shape[0] - 1):
                # 이전, 현재, 다음 프레임의 가중 평균
                smoothed_features[i] = (
                    0.2 * audio_features[i-1] +
                    0.6 * audio_features[i] +
                    0.2 * audio_features[i+1]
                )
        
        # 2. 동적 범위 조정 (음성 강약에 따른 입모양 조절)
        # 음성이 강할 때는 입모양 변화를 더 크게, 약할 때는 더 작게
        feature_magnitude = torch.norm(smoothed_features, dim=-1, keepdim=True)
        # Half(float16) 타입 호환성을 위해 GPU에서 처리하거나 float32로 변환
        if feature_magnitude.dtype == torch.float16:
            # GPU에서 처리하거나 float32로 변환 후 다시 원래 타입으로 복원
            if feature_magnitude.is_cuda:
                normalized_magnitude = torch.tanh(feature_magnitude * 0.5)  # GPU에서는 Half 지원
            else:
                # CPU에서는 float32로 변환 후 처리
                normalized_magnitude = torch.tanh(feature_magnitude.float() * 0.5).half()
        else:
            normalized_magnitude = torch.tanh(feature_magnitude * 0.5)  # 0-1 범위로 정규화
        
        # 음성 강도에 비례하여 특징 강도 조절
        enhanced_features = smoothed_features * (0.7 + 0.6 * normalized_magnitude)
        
        # 3. 묵음 구간 감지 및 처리
        # 음성 에너지가 낮은 구간에서는 입모양 변화를 최소화
        silence_threshold = 0.1
        silence_mask = feature_magnitude.squeeze(-1) < silence_threshold
        
        if silence_mask.any():
            # 묵음 구간에서는 이전 프레임과의 변화를 최소화
            for i in range(1, enhanced_features.shape[0]):
                if silence_mask[i]:
                    enhanced_features[i] = 0.8 * enhanced_features[i-1] + 0.2 * enhanced_features[i]
        
        return enhanced_features 