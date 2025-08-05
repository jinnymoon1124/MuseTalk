import os
import re
import gradio as gr
import numpy as np
import sys
import subprocess
import argparse
import cv2
import torch
import glob
import pickle
from tqdm import tqdm
import copy
from argparse import Namespace
import imageio
from moviepy.editor import *
from transformers import WhisperModel
from datetime import datetime

# 사용X
# import time
# import pdb
# from huggingface_hub import snapshot_download
# import requests
# from omegaconf import OmegaConf
# import shutil
# import gdown
# import ffmpeg



ProjectDir = os.path.abspath(os.path.dirname(__file__))
CheckpointsDir = os.path.join(ProjectDir, "models")

@torch.no_grad()
def debug_inpainting(video_path, bbox_shift, extra_margin=10, parsing_mode="jaw", 
                    left_cheek_width=90, right_cheek_width=90, 
                    enable_occlusion_detection=True, occlusion_sensitivity=0.3):
    """Debug inpainting parameters, only process the first frame"""
    # Set default parameters
    args_dict = {
        "result_dir": './results/debug', 
        "fps": 50,  
        "batch_size": 1, 
        "output_vid_name": '', 
        "use_saved_coord": False,
        "audio_padding_length_left": 2,
        "audio_padding_length_right": 2,
        "version": "v15",
        "extra_margin": extra_margin,
        "parsing_mode": parsing_mode,
        "left_cheek_width": left_cheek_width,
        "right_cheek_width": right_cheek_width
    }
    args = Namespace(**args_dict)

    # Create debug directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Read first frame
    if get_file_type(video_path) == "video":
        reader = imageio.get_reader(video_path)
        first_frame = reader.get_data(0)
        reader.close()
    else:
        first_frame = cv2.imread(video_path)
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    # Save first frame
    debug_frame_path = os.path.join(args.result_dir, "debug_frame.png")
    cv2.imwrite(debug_frame_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
    
    # Get face coordinates
    coord_list, frame_list = get_landmark_and_bbox([debug_frame_path], bbox_shift)
    bbox = coord_list[0]
    frame = frame_list[0]
    
    if bbox == coord_placeholder:
        return None, "No face detected, please adjust bbox_shift parameter"
    
    # Initialize face parser
    fp = FaceParsing(
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width
    )
    
    # Process first frame
    x1, y1, x2, y2 = bbox
    y2 = y2 + args.extra_margin
    y2 = min(y2, frame.shape[0])
    crop_frame = frame[y1:y2, x1:x2]
    crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
    
    # Generate random audio features
    random_audio = torch.randn(1, 50, 384, device=device, dtype=weight_dtype)
    audio_feature = pe(random_audio)
    
    # Get latents
    latents = vae.get_latents_for_unet(crop_frame)
    latents = latents.to(dtype=weight_dtype)
    
    # Generate prediction results
    pred_latents = unet.model(latents, timesteps, encoder_hidden_states=audio_feature).sample
    recon = vae.decode_latents(pred_latents)
    
    # Inpaint back to original image
    res_frame = recon[0]
    res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
    
    # 가림 감지 기능을 포함한 블렌딩 적용
    combine_frame = get_image(frame, res_frame, [x1, y1, x2, y2], 
                             mode=args.parsing_mode, fp=fp,
                             enable_occlusion_detection=enable_occlusion_detection,
                             occlusion_sensitivity=occlusion_sensitivity)
    # Inpaint back to original image
    res_frame = recon[0]
    res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
    combine_frame = get_image(frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
    
    # Save results (no need to convert color space again since get_image already returns RGB format)
    debug_result_path = os.path.join(args.result_dir, "debug_result.png")
    cv2.imwrite(debug_result_path, combine_frame)
    
    # Create information text
    info_text = f"Parameter information:\n" + \
                f"bbox_shift: {bbox_shift}\n" + \
                f"extra_margin: {extra_margin}\n" + \
                f"parsing_mode: {parsing_mode}\n" + \
                f"left_cheek_width: {left_cheek_width}\n" + \
                f"right_cheek_width: {right_cheek_width}\n" + \
                f"enable_occlusion_detection: {enable_occlusion_detection}\n" + \
                f"occlusion_sensitivity: {occlusion_sensitivity}\n" + \
                f"Detected face coordinates: [{x1}, {y1}, {x2}, {y2}]"
    # Create information text
    info_text = f"Parameter information:\n" + \
                f"bbox_shift: {bbox_shift}\n" + \
                f"extra_margin: {extra_margin}\n" + \
                f"parsing_mode: {parsing_mode}\n" + \
                f"left_cheek_width: {left_cheek_width}\n" + \
                f"right_cheek_width: {right_cheek_width}\n" + \
                f"Detected face coordinates: [{x1}, {y1}, {x2}, {y2}]"
    
    return cv2.cvtColor(combine_frame, cv2.COLOR_RGB2BGR), info_text

def print_directory_contents(path):
    for child in os.listdir(path):
        child_path = os.path.join(path, child)
        if os.path.isdir(child_path):
            print(child_path)

def download_model():
    """
    필수 모델 파일들의 존재 여부를 확인하는 함수
    - MuseTalk, SD VAE, Whisper, DWPose, SyncNet, Face Parse, ResNet 모델들이 필요
    - 누락된 모델이 있으면 다운로드 스크립트 실행을 안내
    """
    # 检查必需的模型文件是否存在
    required_models = {
        "MuseTalk": f"{CheckpointsDir}/musetalkV15/unet.pth",
        "MuseTalk": f"{CheckpointsDir}/musetalkV15/musetalk.json",
        "SD VAE": f"{CheckpointsDir}/sd-vae/config.json",
        "Whisper": f"{CheckpointsDir}/whisper/config.json",
        "DWPose": f"{CheckpointsDir}/dwpose/dw-ll_ucoco_384.pth",
        "SyncNet": f"{CheckpointsDir}/syncnet/latentsync_syncnet.pt",
        "Face Parse": f"{CheckpointsDir}/face-parse-bisent/79999_iter.pth",
        "ResNet": f"{CheckpointsDir}/face-parse-bisent/resnet18-5c106cde.pth"
    }
    
    missing_models = []
    for model_name, model_path in required_models.items():
        if not os.path.exists(model_path):
            missing_models.append(model_name)
    
    if missing_models:
        # 全用英文
        print("The following required model files are missing:")
        for model in missing_models:
            print(f"- {model}")
        print("\nPlease run the download script to download the missing models:")
        if sys.platform == "win32":
            print("Windows: Run download_weights.bat")
        else:
            print("Linux/Mac: Run ./download_weights.sh")
        sys.exit(1)
    else:
        print("All required model files exist.")


# ==================== 앱 실행 시 초기화 및 필요한 모델들 로딩 ========================

# 모델 다운로드 확인
download_model()   # 필수 모델 파일들 존재 여부 확인

from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder, get_bbox_range


def fast_check_ffmpeg():
    """
    ffmpeg 설치 여부를 빠르게 확인하는 함수
    - 비디오 처리에 필수적인 ffmpeg가 시스템에 설치되어 있는지 확인
    """
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


@torch.no_grad()
def inference(audio_path, video_path, bbox_shift, extra_margin=10, parsing_mode="jaw", 
              left_cheek_width=90, right_cheek_width=90, 
              enable_occlusion_detection=True, occlusion_sensitivity=0.3,
              use_multi_gpu=True, num_gpus=None, batch_size=4,
              progress=gr.Progress(track_tqdm=True)):
    """
    메인 추론 함수 - 오디오와 비디오를 입력받아 말하는 얼굴을 생성
    
    동작 순서:
    1. 파라미터 설정 및 초기화
    2. 입력 비디오에서 프레임 추출
    3. 오디오에서 특징 추출
    4. 입력 이미지 전처리 (랜드마크 및 바운딩박스 추출)
    5. 배치 단위로 추론 실행 (멀티 GPU 지원)
    6. 생성된 이미지를 원본 비디오에 합성
    7. 최종 비디오 생성 및 오디오 합성
    
    Args:
        use_multi_gpu: 멀티 GPU 병렬 처리 사용 여부
        num_gpus: 사용할 GPU 개수 (None이면 자동 감지)
        batch_size: 각 GPU가 처리할 배치 크기 (기본값 4프레임)
    """
    
    # ===== 1단계: 파라미터 설정 및 초기화 =====
    # inference.py와 동일한 기본 파라미터 설정
    args_dict = {
        "result_dir": './results/output', 
        "fps": 50, 
        "batch_size": batch_size,  # UI에서 전달된 배치 크기 사용
        "output_vid_name": '', 
        "use_saved_coord": False,
        "audio_padding_length_left": 0,
        "audio_padding_length_right": 0,
        "version": "v15",  # v15 버전 고정 사용
        "extra_margin": extra_margin,
        "parsing_mode": parsing_mode,
        "left_cheek_width": left_cheek_width,
        "right_cheek_width": right_cheek_width
    }
    args = Namespace(**args_dict)

    # ⏰ [TIME] 전체 처리 시작 시간 기록
    start_time = datetime.now()
    print(f"\n⏰ ========== 강사 강의 영상 립싱크 처리 시작 ==========")
    print(f"   🕐 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   📁 입력 비디오: {os.path.basename(video_path)}")
    print(f"   🎵 입력 오디오: {os.path.basename(audio_path)}")
    print(f"=======================================================\n")

    # ffmpeg 설치 여부 확인
    if not fast_check_ffmpeg():
        print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

    # 출력 파일명 생성
    input_basename = os.path.basename(video_path).split('.')[0]
    audio_basename = os.path.basename(audio_path).split('.')[0]
    output_basename = f"{input_basename}_{audio_basename}"
    
    # 임시 디렉토리 생성 및 정리
    temp_dir = os.path.join(args.result_dir, f"{args.version}")
    
    # 🔧 [CLEANUP] 시작 시 전체 임시 디렉토리 정리 (강사 강의 영상 처리 시 깨끗한 시작)
    if os.path.exists(temp_dir):
        print(f"🧹 [CLEANUP] 시작 시 전체 임시 디렉토리 정리: {temp_dir}")
        import shutil
        for item in os.listdir(temp_dir):
            item_path = os.path.join(temp_dir, item)
            try:
                if os.path.isdir(item_path):
                    print(f"   - 폴더 삭제: {item}")
                    shutil.rmtree(item_path)
                else:
                    print(f"   - 파일 삭제: {item}")
                    os.remove(item_path)
            except Exception as e:
                print(f"   - 삭제 실패 (무시): {item} - {e}")
        print(f"✅ [CLEANUP] 시작 시 전체 정리 완료")
    
    os.makedirs(temp_dir, exist_ok=True)
    
    # 결과 저장 경로 설정
    result_img_save_path = os.path.join(temp_dir, output_basename)
    crop_coord_save_path = os.path.join(args.result_dir, "../", input_basename+".pkl")
    
    # 결과 저장 폴더 생성 (이미 위에서 전체 정리 완료)
    os.makedirs(result_img_save_path, exist_ok=True)

    if args.output_vid_name == "":
        output_vid_name = os.path.join(temp_dir, output_basename+".mp4")
    else:
        output_vid_name = os.path.join(temp_dir, args.output_vid_name)
        
    # ===== 2단계: 입력 비디오에서 프레임 추출 =====
    if get_file_type(video_path) == "video":
        # 비디오 파일인 경우: 프레임을 추출하여 이미지로 저장
        save_dir_full = os.path.join(temp_dir, input_basename)
        
        # 프레임 저장 폴더 생성 (이미 위에서 전체 정리 완료)
        os.makedirs(save_dir_full, exist_ok=True)
        # 비디오 읽기
        reader = imageio.get_reader(video_path)

        # 각 프레임을 이미지로 저장
        for i, im in enumerate(reader):
            imageio.imwrite(f"{save_dir_full}/{i:08d}.png", im)
        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
        fps = get_video_fps(video_path)
    else: 
        # 이미지 폴더인 경우: 기존 이미지 파일들을 사용
        input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        fps = args.fps
        
    # ===== 3단계: 오디오에서 특징 추출 =====
    # Whisper 모델을 사용하여 오디오에서 특징 추출
    # 이 단계는 립싱크의 핵심입니다 - 음성의 특징을 분석하여 입술 움직임을 예측할 수 있는 정보를 추출합니다
    
    # 1. 음성 파일을 AI가 이해할 수 있는 형태로 변환
    # - 음성 파일을 읽어서 16kHz로 변환 (표준 주파수)
    # - 30초씩 나누어서 처리 (AI 모델이 한 번에 처리할 수 있는 최적 길이)
    # - 각 구간을 숫자 형태로 변환 (AI가 이해할 수 있는 형태)
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    
    # 2. 음성 특징을 비디오 프레임과 동기화
    # - 비디오는 1초에 50장의 이미지(프레임)로 구성됩니다
    # - 음성도 같은 시간에 맞춰서 50개 구간으로 나눕니다
    # - 각 프레임마다 해당 시간의 음성 특징을 제공합니다
    # - 이렇게 하면 "이 시간에 이런 소리가 나면 입술이 이렇게 움직여야 한다"는 정보를 얻을 수 있습니다
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features,  # 변환된 음성 특징들
        device,                  # GPU 또는 CPU 사용 여부
        weight_dtype,            # 데이터 타입 (float16 또는 float32)
        whisper,                 # Whisper AI 모델
        librosa_length,          # 전체 음성 길이
        fps=fps,                 # 비디오 프레임 레이트 (초당 프레임 수)
        audio_padding_length_left=args.audio_padding_length_left,    # 왼쪽 패딩 (이전 프레임과의 연결)
        audio_padding_length_right=args.audio_padding_length_right,  # 오른쪽 패딩 (다음 프레임과의 연결)
    )
        
    # ===== 4단계: 입력 이미지 전처리 =====
    if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        # 저장된 좌표가 있으면 재사용 (시간 절약)
        print("using extracted coordinates")
        with open(crop_coord_save_path,'rb') as f:
            coord_list = pickle.load(f)
        frame_list = read_imgs(input_img_list)
    else:
        # 랜드마크 및 바운딩박스 추출 (시간이 오래 걸림)
        print("extracting landmarks...time consuming")
        
        preprocessing_start = datetime.now()
        print(f"⏰ [TIME] 얼굴 전처리 시작: {preprocessing_start.strftime('%H:%M:%S')}")
        
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        
        preprocessing_end = datetime.now()
        preprocessing_duration = preprocessing_end - preprocessing_start
        print(f"⏰ [TIME] 얼굴 전처리 완료: {preprocessing_duration.total_seconds():.1f}초 소요")
        
        # 추출된 좌표를 파일로 저장 (다음 실행 시 재사용)
        with open(crop_coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)
    bbox_shift_text = get_bbox_range(input_img_list, bbox_shift)
    
    # 얼굴 파싱 모델 초기화
    fp = FaceParsing(
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width
    )
    
    # ===== 5단계: 입력 이미지를 VAE 잠재 공간으로 변환 =====
    i = 0
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        y2 = y2 + args.extra_margin
        y2 = min(y2, frame.shape[0])
        # 얼굴 영역을 크롭하고 256x256으로 리사이즈
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
        # VAE를 사용하여 잠재 벡터로 변환
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)

    # 첫 프레임과 마지막 프레임을 부드럽게 연결하기 위해 순환 리스트 생성
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    
    # ===== 6단계: 배치 단위로 추론 실행 (멀티 GPU 지원) =====
    print("start inference")
    
    # 멀티 GPU 파라미터 디버깅
    print(f"🔍 [DEBUG] 멀티 GPU 파라미터 확인:")
    print(f"   - use_multi_gpu: {use_multi_gpu}")
    print(f"   - num_gpus: {num_gpus}")
    print(f"   - batch_size: {batch_size}")
    print(f"   - torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"   - use_multi_gpu and torch.cuda.device_count() > 1: {use_multi_gpu and torch.cuda.device_count() > 1}")
    
    # 멀티 GPU 성공 여부 플래그 초기화 (단일 GPU 모드에서도 사용)
    multi_gpu_success = False
    
    if use_multi_gpu and torch.cuda.device_count() > 1:
        # 멀티 GPU 병렬 처리
        multigpu_start = datetime.now()
        print(f"\n🚀 ============ 멀티 GPU 병렬 처리 시작 ============")
        print(f"   ⏰ 시작 시간: {multigpu_start.strftime('%H:%M:%S')}")
        print(f"   시스템 정보:")
        print(f"   - 사용 가능한 GPU: {torch.cuda.device_count()}개")
        print(f"   - 실제 사용할 GPU: {num_gpus if num_gpus else torch.cuda.device_count()}개")
        print(f"   - 배치 크기: {batch_size}프레임")
        print(f"   - 오디오 파일: {os.path.basename(audio_path)}")
        print(f"   - 비디오 파일: {os.path.basename(video_path)}")
        print(f"=================================================\n")
        
        from musetalk.utils.multi_gpu_manager import DynamicMultiGPUManager
        
        # 동적 멀티 GPU 매니저 초기화 (세그먼트 분할 제거)
        # Gradio에서 전달된 num_gpus가 float일 수 있으므로 int로 변환
        num_gpus_int = int(num_gpus) if num_gpus is not None else None
        batch_size_int = int(batch_size) if batch_size is not None else 4
        
        print(f"🔍 [DEBUG] 변환된 파라미터:")
        print(f"   - num_gpus_int: {num_gpus_int}")
        print(f"   - batch_size_int: {batch_size_int}")
        
        multi_gpu_manager = DynamicMultiGPUManager(
            num_gpus=num_gpus_int,
            segment_duration=0  # 세그먼트 분할 비활성화
        )
        
        # 모델 경로 설정
        model_paths = {
            'unet_model_path': "./models/musetalkV15/unet.pth",
            'vae_type': "sd-vae",
            'unet_config': "./models/musetalkV15/musetalk.json",
            'whisper_path': "openai/whisper-tiny"  # 기존과 동일한 경로 사용
        }
        
        # GPU 워커 초기화
        # 메인 프로세스가 사용 중인 GPU ID 추출 (cuda:1 -> 1)
        main_gpu_id = int(str(device).split(':')[1]) if ':' in str(device) else 0
        print(f"🔍 [DEBUG] 메인 프로세스 GPU ID: {main_gpu_id}")
        
        multi_gpu_manager.initialize_workers(model_paths, use_float16=True, main_gpu_id=main_gpu_id)
        
        # 모델 설정 준비 (강사 강의 영상 처리를 위한 실제 fps 사용)
        print(f"🔍 [DEBUG] FPS 설정 확인:")
        print(f"   - 실제 비디오 fps: {fps}")
        print(f"   - args.fps (기본값): {args.fps}")
        
        model_config = {
            'fps': fps,  # 실제 비디오 fps 사용 (args.fps 대신)
            'batch_size': batch_size_int,  # UI에서 설정한 배치 크기 사용
            'audio_padding_length_left': args.audio_padding_length_left,
            'audio_padding_length_right': args.audio_padding_length_right,
            'extra_margin': args.extra_margin,
            # 🔧 [MODEL PATHS] 멀티 GPU 배치 처리를 위한 모델 경로 추가
            'unet_model_path': "./models/musetalkV15/unet.pth",
            'vae_type': "sd-vae",
            'unet_config': "./models/musetalkV15/musetalk.json",
            'whisper_path': 'openai/whisper-tiny'
            # 🎯 [NO SEGMENTS] 세그먼트 분할 완전 제거 - 배치 단위로만 처리
        }
        
        multi_gpu_success = False
        try:
            # 🎯 [REVOLUTIONARY] 단일 GPU 스타일 멀티 GPU 처리
            # 세그먼트 분할 없이 배치만 분산하여 단일 GPU 품질 + 멀티 GPU 속도 달성
            res_frame_list = multi_gpu_manager.process_with_single_gpu_style(
                audio_path=audio_path,
                video_frames=frame_list,  # 순환 리스트 대신 원본 사용
                coord_list=coord_list,    # 순환 리스트 대신 원본 사용
                model_config=model_config
            )
            
            # 멀티 GPU 처리 완료 시간 측정
            multigpu_end = datetime.now()
            multigpu_duration = multigpu_end - multigpu_start
            
            # 멀티 GPU 처리 성공 여부 확인
            if len(res_frame_list) > 0:
                multi_gpu_success = True
                print(f"\n🎉 ========== 멀티 GPU 처리 성공 ==========")
                print(f"   ⏰ 완료 시간: {multigpu_end.strftime('%H:%M:%S')}")
                print(f"   ⏱️  소요 시간: {multigpu_duration.total_seconds():.1f}초")
                print(f"   - 생성된 프레임: {len(res_frame_list)}개")
                print(f"   - 사용된 fps: {fps}")
                print(f"   - 예상 영상 길이: {len(res_frame_list)/fps:.2f}초")
                print(f"   - 처리 방식: 멀티 GPU 병렬 처리")
                print(f"   - 사용된 GPU: {num_gpus if num_gpus else torch.cuda.device_count()}개")
                print(f"==========================================\n")
            else:
                print(f"\n⚠️ ========== 멀티 GPU 처리 실패 ==========")
                print(f"   ⏰ 완료 시간: {multigpu_end.strftime('%H:%M:%S')}")
                print(f"   ⏱️  소요 시간: {multigpu_duration.total_seconds():.1f}초")
                print(f"   - 생성된 프레임: 0개")
                print(f"   - 단일 GPU 폴백 모드로 전환")
                print(f"==========================================\n")
            
        except Exception as e:
            print(f"\n💥 ========== 멀티 GPU 처리 오류 ==========")
            print(f"   - 오류: {e}")
            print(f"   - 단일 GPU 폴백 모드로 전환")
            print(f"==========================================\n")
            
        finally:
            # 멀티 GPU 매니저 종료
            print(f"🔄 [MultiGPU] 워커 프로세스 정리 중...")
            multi_gpu_manager.shutdown()
            print(f"✅ [MultiGPU] 모든 워커 프로세스 정리 완료")
            
        # 멀티 GPU 처리가 실패한 경우 단일 GPU로 폴백
        if not multi_gpu_success:
            print(f"\n💻 ============ 단일 GPU 폴백 처리 시작 ============")
            print(f"   - 멀티 GPU 실패로 인한 폴백 모드")
            print(f"   - 원본 프레임 사용 (순환 리스트 없음)")
            print(f"   - 사용 디바이스: {device}")
            print(f"================================================\n")
            
            # 순환 리스트 대신 원본 리스트 사용
            video_num = len(whisper_chunks)
            batch_size = args.batch_size
            
            # 데이터 생성기 초기화 (순환 리스트 없이)
            gen = datagen(
                whisper_chunks=whisper_chunks,
                vae_encode_latents=input_latent_list,  # 순환 리스트 대신 원본 사용
                batch_size=batch_size,
                delay_frame=0,
                device=device,
            )
            res_frame_list = []
            
            # 배치 단위로 추론 실행
            for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
                audio_feature_batch = pe(whisper_batch)
                latent_batch = latent_batch.to(dtype=weight_dtype)
                pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                recon = vae.decode_latents(pred_latents)
                for res_frame in recon:
                    res_frame_list.append(res_frame)
                    
            print(f"✅ [SingleGPU] 단일 GPU 폴백 처리 완료: {len(res_frame_list)}개 프레임 생성")
            
    else:
        # 단일 GPU 처리 (기존 방식)
        singlegpu_start = datetime.now()
        print(f"\n💻 ============ 단일 GPU 처리 시작 ============")
        print(f"   ⏰ 시작 시간: {singlegpu_start.strftime('%H:%M:%S')}")
        if torch.cuda.device_count() <= 1:
            print(f"   - 사용 가능한 GPU: {torch.cuda.device_count()}개 (멀티 GPU 불가)")
        else:
            print(f"   - 멀티 GPU 옵션이 비활성화됨")
        print(f"   - 사용 디바이스: {device}")
        print(f"   - 오디오 파일: {os.path.basename(audio_path)}")
        print(f"   - 비디오 파일: {os.path.basename(video_path)}")
        print(f"=============================================\n")
        
        video_num = len(whisper_chunks)
        batch_size = args.batch_size
        
        # 데이터 생성기 초기화
        # 이 생성기는 음성 특징과 이미지 특징을 함께 제공하여 AI 모델이 처리할 수 있도록 합니다
        gen = datagen(
            whisper_chunks=whisper_chunks,        # 각 프레임에 해당하는 음성 특징들
            vae_encode_latents=input_latent_list,  # 원본 이미지 특징 (순환 리스트 없음)
            batch_size=batch_size,                # 한 번에 처리할 프레임 수 (메모리 효율성을 위해)
            delay_frame=0,                        # 지연 프레임 (현재는 사용하지 않음)
            device=device,                        # GPU 또는 CPU 사용 여부
        )
        res_frame_list = []
        
        # 배치 단위로 추론 실행
        # 이 부분이 실제 립싱크를 생성하는 핵심입니다!
        for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
            # 1. 오디오 특징을 위치 인코딩
            # - 음성의 시간적 정보를 AI 모델이 이해할 수 있도록 변환
            # - "이 시간에 이런 소리가 나고 있다"는 정보를 제공
            audio_feature_batch = pe(whisper_batch)
            
            # 2. 잠재 벡터를 모델 가중치 타입과 일치하도록 변환
            # - 이미지 특징을 AI 모델이 처리할 수 있는 형태로 변환
            latent_batch = latent_batch.to(dtype=weight_dtype)
            
            # 3. UNet 모델을 사용하여 새로운 잠재 벡터 생성
            # - 이것이 실제 립싱크를 만드는 마법의 부분입니다!
            # - AI 모델이 "이 음성에 맞는 입술 움직임"을 예측합니다
            # - timesteps는 확산 모델에서 사용하는 시간 단계 (현재는 0으로 고정)
            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            
            # 4. VAE를 사용하여 잠재 벡터를 이미지로 디코딩
            # - AI가 예측한 "숨겨진 특징"을 실제 이미지로 변환
            # - 이제 입술이 움직인 새로운 얼굴 이미지가 생성됩니다
            recon = vae.decode_latents(pred_latents)
            
            # 5. 생성된 이미지들을 결과 리스트에 추가
            for res_frame in recon:
                res_frame_list.append(res_frame)
        
        # 단일 GPU 처리 완료 시간 측정
        singlegpu_end = datetime.now()
        singlegpu_duration = singlegpu_end - singlegpu_start
        print(f"\n💻 ========== 단일 GPU 처리 완료 ==========")
        print(f"   ⏰ 완료 시간: {singlegpu_end.strftime('%H:%M:%S')}")
        print(f"   ⏱️  소요 시간: {singlegpu_duration.total_seconds():.1f}초")
        print(f"   - 생성된 프레임: {len(res_frame_list)}개")
        print(f"   - 사용된 fps: {fps}")
        print(f"   - 예상 영상 길이: {len(res_frame_list)/fps:.2f}초")
        print(f"   - 처리 방식: 단일 GPU 처리")
        print(f"==========================================\n")
            
    # ===== 7단계: 생성된 이미지를 원본 비디오에 합성 =====
    print("pad talking image to original video")
    
    # 가림 감지 비교 이미지 저장을 위한 디렉토리 생성
    check_dir = "./results/check"
    os.makedirs(check_dir, exist_ok=True)
    
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        # 1. 현재 프레임에 해당하는 얼굴 위치 정보 가져오기
        # - 원본 비디오에서 얼굴이 어디에 있는지 알려주는 좌표
        # 순환 리스트 사용 여부 확인 (멀티 GPU 성공 시에만 순환 리스트 사용)
        if 'multi_gpu_success' in locals() and multi_gpu_success:
            bbox = coord_list_cycle[i%(len(coord_list_cycle))]
            ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
        else:
            # 단일 GPU 또는 폴백 모드에서는 원본 리스트 사용
            bbox = coord_list[i%(len(coord_list))]
            ori_frame = copy.deepcopy(frame_list[i%(len(frame_list))])
        
        # 2. 얼굴 영역의 좌표 추출
        x1, y1, x2, y2 = bbox  # x1,y1: 왼쪽 위, x2,y2: 오른쪽 아래
        y2 = y2 + args.extra_margin  # 턱 움직임을 위해 아래쪽 여백 추가
        y2 = min(y2, frame.shape[0])  # 이미지 경계를 벗어나지 않도록 제한
        
        try:
            # 3. 생성된 얼굴 이미지를 원본 크기로 리사이즈
            # - AI가 생성한 256x256 얼굴 이미지를 원본 얼굴 크기로 확대
            res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
        except:
            continue
        
        # 4-1. 진짜 처리 전 이미지 생성 (원본 + 생성된 얼굴만 단순 합성)
        # - 가림 감지, 얼굴 파싱, 블러 등 모든 처리 없이 단순 합성
        combine_frame_raw = ori_frame.copy()
        combine_frame_raw[y1:y2, x1:x2] = res_frame  # 단순 픽셀 대체

        # 4-2. 가림 감지 없는 기본 블렌딩 (얼굴 파싱 + 블러 적용)
        combine_frame_before = get_image(ori_frame, res_frame, [x1, y1, x2, y2], 
                                        mode=args.parsing_mode, fp=fp,
                                        enable_occlusion_detection=False,
                                        occlusion_sensitivity=0.3)

        # 4-3. 가림 감지 기능을 포함한 v15 버전 블렌딩 사용
        combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], 
                                 mode=args.parsing_mode, fp=fp,
                                 enable_occlusion_detection=enable_occlusion_detection,
                                 occlusion_sensitivity=occlusion_sensitivity)

        # 4-4. 가림 감지 비교 이미지 저장 (더 자세한 비교를 위해)
        if i % 50 == 0 or i < 10:
            # 1) 완전 원본 (처리 전)
            cv2.imwrite(f"{check_dir}/frame_{str(i).zfill(8)}_0_original.png", 
                       cv2.cvtColor(ori_frame, cv2.COLOR_RGB2BGR))
            
            # 2) 단순 합성 (얼굴만 대체)
            cv2.imwrite(f"{check_dir}/frame_{str(i).zfill(8)}_1_raw_replace.png", 
                       cv2.cvtColor(combine_frame_raw, cv2.COLOR_RGB2BGR))
            
            # 3) 기본 블렌딩 (가림 감지 없음)
            cv2.imwrite(f"{check_dir}/frame_{str(i).zfill(8)}_2_basic_blending.png", 
                       cv2.cvtColor(combine_frame_before, cv2.COLOR_RGB2BGR))
            
            # 4) 가림 감지 적용
            cv2.imwrite(f"{check_dir}/frame_{str(i).zfill(8)}_3_occlusion_aware.png", 
                       cv2.cvtColor(combine_frame, cv2.COLOR_RGB2BGR))
        
        # 5. 합성된 프레임을 파일로 저장
        # - 각 프레임을 순서대로 저장하여 나중에 비디오로 만들기 위해 준비
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)
        
    # ===== 8단계: 최종 비디오 생성 =====
    video_generation_start = datetime.now()
    print(f"\n🔍 [DEBUG] 최종 비디오 생성 단계:")
    print(f"   ⏰ 시작 시간: {video_generation_start.strftime('%H:%M:%S')}")
    print(f"   - 현재 fps 값: {fps}")
    print(f"   - 생성된 프레임 수: {len(res_frame_list) if multi_gpu_success else '단일 GPU 모드'}")
    
    # ⚠️ 중요: fps 값을 덮어쓰지 않고 실제 비디오 fps 유지 (강사 강의 영상 처리)
    # fps = 50  # ← 이 줄을 제거하여 실제 fps 유지
    
    # 임시 출력 비디오 경로
    output_video = 'temp.mp4'

    # 유효한 이미지 파일 필터링 함수
    # - 8자리 숫자로 된 PNG 파일만 선택 (예: 00000001.png, 00000002.png)
    def is_valid_image(file):
        pattern = re.compile(r'\d{8}\.png')
        return pattern.match(file)

    # 저장된 이미지들을 읽어서 리스트로 변환
    images = []
    # 모든 이미지 파일을 찾아서 숫자 순서대로 정렬
    files = [file for file in os.listdir(result_img_save_path) if is_valid_image(file)]
    files.sort(key=lambda x: int(x.split('.')[0]))

    # 각 이미지 파일을 순서대로 읽어서 리스트에 추가
    for file in files:
        filename = os.path.join(result_img_save_path, file)
        images.append(imageio.imread(filename))
    
    # 🔍 [DEBUG] 이미지 → 비디오 변환 정보
    print(f"🔍 [DEBUG] 이미지 → 비디오 변환:")
    print(f"   - 읽어온 이미지 수: {len(images)}개")
    print(f"   - 멀티 GPU 생성 프레임: {len(res_frame_list) if multi_gpu_success else '단일 GPU 모드'}개")
    print(f"   - 사용할 fps: {fps}")
    print(f"   - 예상 비디오 길이: {len(images)/fps:.2f}초")
    print(f"   - 출력 파일: {output_video}")
    
    # 🔍 [VALIDATION] 이미지 수 검증 (강사 강의 영상 처리 정확성 확인)
    if multi_gpu_success:
        expected_images = len(res_frame_list)
        if len(images) != expected_images:
            print(f"⚠️ [WARNING] 이미지 수 불일치!")
            print(f"   - 예상: {expected_images}개")
            print(f"   - 실제: {len(images)}개")
            print(f"   - 차이: {len(images) - expected_images}개")
        else:
            print(f"✅ [VALIDATION] 이미지 수 일치: {len(images)}개")

    # 이미지들을 비디오로 저장
    # - 여러 장의 이미지를 순서대로 재생하여 비디오로 만듭니다
    # - FFMPEG 코덱을 사용하여 고품질 비디오 생성
    imageio.mimwrite(output_video, images, 'FFMPEG', fps=fps, codec='libx264', pixelformat='yuv420p')
    
    video_generation_end = datetime.now()
    video_generation_duration = video_generation_end - video_generation_start
    print(f"⏰ [TIME] 비디오 생성 완료: {video_generation_duration.total_seconds():.1f}초 소요")

    # ===== 9단계: 오디오와 비디오 합성 =====
    audio_merge_start = datetime.now()
    input_video = './temp.mp4'
    # 입력 비디오와 오디오 파일 존재 여부 확인
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # 비디오 정보 읽기
    reader = imageio.get_reader(input_video)
    temp_video_fps = reader.get_meta_data()['fps']  # temp.mp4의 프레임 레이트
    reader.close() # 윈도우에서 파일 사용 중 오류를 방지하기 위해 즉시 닫기
    
    # 🔍 [DEBUG] 오디오-비디오 합성 단계 fps 확인
    print(f"\n🔍 [DEBUG] 오디오-비디오 합성 단계:")
    print(f"   ⏰ 시작 시간: {audio_merge_start.strftime('%H:%M:%S')}")
    print(f"   - 원본 비디오 fps: {fps}")
    print(f"   - temp.mp4 fps: {temp_video_fps}")
    print(f"   - 최종 저장에 사용할 fps: {fps}")

    # 비디오 클립 로드 (moviepy 라이브러리 사용)
    video_clip = VideoFileClip(input_video)

    # 오디오 클립 로드 (원본 음성 파일)
    audio_clip = AudioFileClip(audio_path)
    
    # 🔍 [DEBUG] 클립 정보 확인
    print(f"🔍 [DEBUG] 클립 정보:")
    print(f"   - 비디오 클립 길이: {video_clip.duration:.2f}초")
    print(f"   - 오디오 클립 길이: {audio_clip.duration:.2f}초")

    # 비디오에 오디오 설정
    # - 이제 립싱크가 적용된 비디오에 원본 음성이 합성됩니다
    video_clip = video_clip.set_audio(audio_clip)

    # 최종 비디오 파일로 저장 (강사 강의 영상 처리를 위한 실제 fps 사용)
    # - libx264: 고품질 비디오 코덱
    # - aac: 고품질 오디오 코덱
    # - fps: 실제 비디오 프레임 레이트 사용 (원본과 동일하게 유지)
    print(f"🔍 [DEBUG] 최종 비디오 저장: {output_vid_name}")
    video_clip.write_videofile(output_vid_name, codec='libx264', audio_codec='aac', fps=fps)
    
    audio_merge_end = datetime.now()
    audio_merge_duration = audio_merge_end - audio_merge_start
    print(f"⏰ [TIME] 오디오-비디오 합성 완료: {audio_merge_duration.total_seconds():.1f}초 소요")

    # 🔧 [CLEANUP] 임시 파일 정리 (강사 강의 영상 처리 후 정리)
    print(f"\n🧹 [CLEANUP] 임시 파일 정리 시작...")
    
    # temp.mp4 삭제
    if os.path.exists("temp.mp4"):
        os.remove("temp.mp4")
        print(f"✅ [CLEANUP] temp.mp4 삭제 완료")
    
    # 임시 이미지 폴더 삭제 (메모리 절약)
    if os.path.exists(result_img_save_path):
        import shutil
        shutil.rmtree(result_img_save_path)
        print(f"✅ [CLEANUP] 임시 이미지 폴더 삭제 완료: {result_img_save_path}")
    
    # ⏰ [TIME] 전체 처리 완료 시간 기록 및 통계 출력
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    # 시간을 시:분:초 형태로 변환
    total_seconds = int(total_duration.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    print(f"\n⏰ ========== 강사 강의 영상 립싱크 처리 완료 ==========")
    print(f"   🕐 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   🕕 종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   ⏱️  총 소요 시간: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"   📊 단계별 소요 시간:")
    if 'preprocessing_duration' in locals():
        print(f"      - 얼굴 전처리: {preprocessing_duration.total_seconds():.1f}초")
    if 'multigpu_duration' in locals():
        print(f"      - 멀티 GPU 처리: {multigpu_duration.total_seconds():.1f}초")
    if 'singlegpu_duration' in locals():
        print(f"      - 단일 GPU 처리: {singlegpu_duration.total_seconds():.1f}초")
    if 'video_generation_duration' in locals():
        print(f"      - 비디오 생성: {video_generation_duration.total_seconds():.1f}초")
    if 'audio_merge_duration' in locals():
        print(f"      - 오디오 합성: {audio_merge_duration.total_seconds():.1f}초")
    print(f"   📁 처리 결과:")
    print(f"      - 입력 비디오: {os.path.basename(video_path)}")
    print(f"      - 입력 오디오: {os.path.basename(audio_path)}")
    print(f"      - 출력 파일: {os.path.basename(output_vid_name)}")
    print(f"   🎉 [SUCCESS] 최종 결과 저장 완료: {output_vid_name}")
    print(f"   🧹 [CLEANUP] 모든 임시 파일 정리 완료")
    print(f"=======================================================")
    
    return output_vid_name,bbox_shift_text



# 멀티프로세싱 워커 프로세스인지 확인 (multiprocessing 호출 스택 기반)
import os
import sys
import traceback

# 🔥 CRITICAL: multiprocessing fork에서 워커 프로세스 감지
# fork 방식에서는 환경 변수나 다른 방법으로 워커 프로세스 판별
def is_multiprocessing_worker():
    """multiprocessing fork 워커 프로세스인지 확인"""
    try:
        # fork 방식에서는 환경 변수 기반으로 판별
        # 워커 프로세스에서 설정된 환경 변수 확인
        worker_env = os.environ.get('MUSETALK_WORKER_PROCESS') == 'TRUE'
        worker_gpu = os.environ.get('WORKER_GPU_ID') is not None
        
        # 호출 스택도 함께 확인 (보조적)
        stack = traceback.extract_stack()
        stack_worker = any('multiprocessing' in frame.filename for frame in stack)
        
        return worker_env or worker_gpu or stack_worker
    except:
        return False

# 🔥 CRITICAL: spawn 방식에서 환경 변수 기반 GPU 할당
# 워커 프로세스는 환경 변수가 설정된 상태로 app.py 실행됨
worker_gpu_id = os.environ.get('WORKER_GPU_ID') or os.environ.get('FORCE_WORKER_GPU')
is_worker = os.environ.get('MUSETALK_WORKER_PROCESS') == 'TRUE'

print(f"🔍 [DEBUG] 프로세스 타입 체크:")
print(f"   - PID: {os.getpid()}")
print(f"   - MUSETALK_WORKER_PROCESS: {os.environ.get('MUSETALK_WORKER_PROCESS')}")
print(f"   - is_worker: {is_worker}")
print(f"   - WORKER_GPU_ID: {os.environ.get('WORKER_GPU_ID')}")
print(f"   - FORCE_WORKER_GPU: {os.environ.get('FORCE_WORKER_GPU')}")
print(f"   - CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

if is_worker and worker_gpu_id:
    # 워커 프로세스: CUDA_VISIBLE_DEVICES 적용으로 인해 항상 cuda:0 사용
    print(f"=============== Worker Process Using GPU {worker_gpu_id} ================")
    print(f"🔥 [Worker] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}로 설정됨")
    print(f"🔥 [Worker] 워커 프로세스에서는 실제 GPU {worker_gpu_id}이 cuda:0으로 매핑됩니다")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # CUDA_VISIBLE_DEVICES로 인해 항상 0
    print(f"🔥 [Worker] 디바이스 설정: {device} (실제 물리 GPU: {worker_gpu_id})")
else:
    # 메인 프로세스: GPU 0 사용
    print(f"=============== Main Process Using device: cuda:0 ================")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 모든 모델 로드 (각 프로세스가 올바른 GPU에서 로딩)
vae, unet, pe = load_all_model(
    unet_model_path="./models/musetalkV15/unet.pth", 
    vae_type="sd-vae",
    unet_config="./models/musetalkV15/musetalk.json",
    device=device
)

# ===== 명령행 인수 파싱 =====
parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_path", type=str, default=r"ffmpeg-master-latest-win64-gpl-shared\bin", help="Path to ffmpeg executable")
parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
parser.add_argument("--share", action="store_true", help="Create a public link")
parser.add_argument("--use_float16", action="store_true", help="Use float16 for faster inference")
args = parser.parse_args()

# ===== 데이터 타입 설정 =====
if args.use_float16:
    # float16을 사용하여 더 빠른 추론을 위해 모델을 반정밀도로 변환
    pe = pe.half()
    vae.vae = vae.vae.half()
    unet.model = unet.model.half()
    weight_dtype = torch.float16
else:
    weight_dtype = torch.float32

# ===== 모델을 지정된 디바이스로 이동 =====
pe = pe.to(device)
vae.vae = vae.vae.to(device)
unet.model = unet.model.to(device)

# 타임스텝 설정 (확산 모델에서 사용)
timesteps = torch.tensor([0], device=device)

# ===== 오디오 프로세서 및 Whisper 모델 초기화 =====
audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
whisper = WhisperModel.from_pretrained("./models/whisper")
whisper = whisper.to(device=device, dtype=weight_dtype).eval()
whisper.requires_grad_(False)


def check_video(video):
    """
    입력 비디오를 처리하여 50fps로 변환하는 함수
    
    동작 과정:
    1. 이미 처리된 비디오인지 확인
    2. 비디오를 프레임 단위로 분해
    3. 50fps로 리샘플링
    4. 새로운 비디오로 저장
    """
    if not isinstance(video, str):
        return video # none 타입인 경우
    # 출력 비디오 파일명 정의
    dir_path, file_name = os.path.split(video)
    if file_name.startswith("outputxxx_"):
        return video
    # 파일명에 출력 접두사 추가
    output_file_name = "outputxxx_" + file_name

    # 결과 디렉토리 생성
    os.makedirs('./results',exist_ok=True)
    os.makedirs('./results/output',exist_ok=True)
    os.makedirs('./results/input',exist_ok=True)

    # 디렉토리 경로와 새 파일명 결합
    output_video = os.path.join('./results/input', output_file_name)

    # 비디오 읽기
    reader = imageio.get_reader(video)
    fps = reader.get_meta_data()['fps']  # 원본 비디오에서 fps 가져오기

    # fps를 원본 fps로 유지 (강사 강의 영상의 자연스러운 프레임 레이트 보존)
    frames = [im for im in reader]
    target_fps = fps  # 원본 fps 사용
    
    L = len(frames)
    L_target = int(L / fps * target_fps)
    original_t = [x / fps for x in range(1, L+1)]
    t_idx = 0
    target_frames = []
    for target_t in range(1, L_target+1):
        while target_t / target_fps > original_t[t_idx]:
            t_idx += 1      # target_t / target_fps <= original_t[t_idx]인 첫 번째 t_idx 찾기
            if t_idx >= L:
                break
        target_frames.append(frames[t_idx])

    # 비디오 저장 (원본 fps로 저장하여 자연스러운 재생 속도 유지)
    imageio.mimwrite(output_video, target_frames, 'FFMPEG', fps=fps, codec='libx264', quality=9, pixelformat='yuv420p')
    return output_video




# ===== Gradio UI CSS 스타일 =====
css = """#input_img {max-width: 1024px !important} #output_vid {max-width: 1024px; max-height: 576px}"""

# ===== Gradio 웹 인터페이스 구성 =====
with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column():
            # 입력 컴포넌트들
            audio = gr.Audio(label="Drving Audio",type="filepath")  # 오디오 입력
            video = gr.Video(label="Reference Video",sources=['upload'])  # 비디오 입력
            bbox_shift = gr.Number(label="BBox_shift value, px", value=0)  # 바운딩박스 이동값
            extra_margin = gr.Slider(label="Extra Margin", minimum=0, maximum=400, value=0, step=1)  # 추가 여백
            parsing_mode = gr.Radio(label="Parsing Mode", choices=["jaw", "raw"], value="jaw")  # 파싱 모드
            left_cheek_width = gr.Slider(label="Left Cheek Width", minimum=0, maximum=200, value=20, step=1)  # 왼쪽 볼 너비
            right_cheek_width = gr.Slider(label="Right Cheek Width", minimum=0, maximum=200, value=20, step=1)  # 오른쪽 볼 너비
            
            # 가림 감지 관련 컨트롤 추가
            with gr.Group():
                gr.Markdown("### 가림 감지 설정 (Occlusion Detection)")
                enable_occlusion_detection = gr.Checkbox(label="가림 감지 활성화 (Enable Occlusion Detection)", value=True)
                occlusion_sensitivity = gr.Slider(label="가림 감지 민감도 (Occlusion Sensitivity)", 
                                                 minimum=0.1, maximum=1.0, value=0.3, step=0.1,
                                                 info="값이 낮을수록 더 민감하게 가림을 감지/값이 높을수록 덜 민감하게 가림을 감지")
            
            # 멀티 GPU 관련 컨트롤 추가
            with gr.Group():
                gr.Markdown("### 🚀 단일 GPU 스타일 멀티 GPU 병렬 처리 (Single-GPU Style Multi-GPU Processing)")
                use_multi_gpu = gr.Checkbox(label="멀티 GPU 병렬 처리 사용 (Enable Multi-GPU Processing)", value=True,
                                          info="🎯 혁신적 처리 방식: 단일 GPU 품질 + 멀티 GPU 속도")
                num_gpus = gr.Slider(label="사용할 GPU 개수 (Number of GPUs)", 
                                   minimum=1, maximum=8, value=torch.cuda.device_count(), step=1,
                                   info="사용할 GPU 개수 (배치 단위로 동적 분산 처리)")
                batch_size = gr.Slider(label="배치 크기 (Batch Size)", 
                                     minimum=1, maximum=16, value=4, step=1,
                                     info="각 GPU가 처리할 배치 크기 (4프레임 권장)")
                gr.Markdown(f"**현재 사용 가능한 GPU: {torch.cuda.device_count()}개**")
                gr.Markdown("**🎯 새로운 동작 방식:** 전체 비디오를 메인 GPU에서 통합 처리 → 배치 단위로 워커 GPU들에 분산 → 완벽한 시간적 연속성 보장")
            
            bbox_shift_scale = gr.Textbox(label="'left_cheek_width'와 'right_cheek_width' 파라미터는 파싱 모델이 'jaw'일 때 좌우 볼 편집 범위를 결정합니다. 'extra_margin' 파라미터는 턱의 움직임 범위를 결정합니다. 사용자는 이 세 파라미터를 자유롭게 조정하여 더 나은 인페인팅 결과를 얻을 수 있습니다. 가림 감지 기능은 마이크 등의 물체에 의해 얼굴이 가려진 부분에서 자연스러운 립싱크를 제공합니다.")

            with gr.Row():
                debug_btn = gr.Button("1. Test Inpainting ")
                btn = gr.Button("Generate")
        with gr.Column():
            debug_image = gr.Image(label="Test Inpainting Result (First Frame)")
            debug_info = gr.Textbox(label="Parameter Information", lines=5)
            out1 = gr.Video()
    
    # 비디오 변경 시 자동으로 50fps로 변환
    video.change(
        fn=check_video, inputs=[video], outputs=[video]
    )
    # 생성 버튼 클릭 시 추론 실행
    btn.click(
        fn=inference,
        inputs=[
            audio,
            video,
            bbox_shift,
            extra_margin,
            parsing_mode,
            left_cheek_width,
            right_cheek_width,
            enable_occlusion_detection,
            occlusion_sensitivity,
            use_multi_gpu,
            num_gpus,
            batch_size
        ],
        outputs=[out1,bbox_shift_scale]
    )
    debug_btn.click(
        fn=debug_inpainting,
        inputs=[
            video,
            bbox_shift,
            extra_margin,
            parsing_mode,
            left_cheek_width,
            right_cheek_width,
            enable_occlusion_detection,
            occlusion_sensitivity
        ],
        outputs=[debug_image, debug_info]
    )
    # debug_btn.click(
    #     fn=debug_inpainting,
    #     inputs=[
    #         video,
    #         bbox_shift,
    #         extra_margin,
    #         parsing_mode,
    #         left_cheek_width,
    #         right_cheek_width
    #     ],
    #     outputs=[debug_image, debug_info]
    # )

# Check ffmpeg and add to PATH
# if not fast_check_ffmpeg():
#     print(f"Adding ffmpeg to PATH: {args.ffmpeg_path}")
#     # According to operating system, choose path separator
#     path_separator = ';' if sys.platform == 'win32' else ':'
#     os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
#     if not fast_check_ffmpeg():
#         print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

# # Solve asynchronous IO issues on Windows
# if sys.platform == 'win32':
#     import asyncio
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ===== Gradio 애플리케이션 시작 =====
# 워커 프로세스에서 실행되지 않도록 보호
if __name__ == "__main__":
    demo.queue().launch(
        share=args.share, 
        debug=True, 
        server_name="10.202.15.248",
        server_port=8000
    )
