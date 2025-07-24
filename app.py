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
# def debug_inpainting(video_path, bbox_shift, extra_margin=10, parsing_mode="jaw", 
#                     left_cheek_width=90, right_cheek_width=90, 
#                     enable_occlusion_detection=True, occlusion_sensitivity=0.3):
#     """Debug inpainting parameters, only process the first frame"""
#     # Set default parameters
#     args_dict = {
#         "result_dir": './results/debug', 
#         "fps": 50,  
#         "batch_size": 1, 
#         "output_vid_name": '', 
#         "use_saved_coord": False,
#         "audio_padding_length_left": 2,
#         "audio_padding_length_right": 2,
#         "version": "v15",
#         "extra_margin": extra_margin,
#         "parsing_mode": parsing_mode,
#         "left_cheek_width": left_cheek_width,
#         "right_cheek_width": right_cheek_width
#     }
#     args = Namespace(**args_dict)
# def debug_inpainting(video_path, bbox_shift, extra_margin=10, parsing_mode="jaw", 
#                     left_cheek_width=90, right_cheek_width=90):
#     """Debug inpainting parameters, only process the first frame"""
#     # Set default parameters
#     args_dict = {
#         "result_dir": './results/debug', 
#         "fps": 50,  
#         "batch_size": 1, 
#         "output_vid_name": '', 
#         "use_saved_coord": False,
#         "audio_padding_length_left": 2,
#         "audio_padding_length_right": 2,
#         "version": "v15",
#         "extra_margin": extra_margin,
#         "parsing_mode": parsing_mode,
#         "left_cheek_width": left_cheek_width,
#         "right_cheek_width": right_cheek_width
#     }
#     args = Namespace(**args_dict)

#     # Create debug directory
#     os.makedirs(args.result_dir, exist_ok=True)
    
#     # Read first frame
#     if get_file_type(video_path) == "video":
#         reader = imageio.get_reader(video_path)
#         first_frame = reader.get_data(0)
#         reader.close()
#     else:
#         first_frame = cv2.imread(video_path)
#         first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
#     # Save first frame
#     debug_frame_path = os.path.join(args.result_dir, "debug_frame.png")
#     cv2.imwrite(debug_frame_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
    
#     # Get face coordinates
#     coord_list, frame_list = get_landmark_and_bbox([debug_frame_path], bbox_shift)
#     bbox = coord_list[0]
#     frame = frame_list[0]
    
#     if bbox == coord_placeholder:
#         return None, "No face detected, please adjust bbox_shift parameter"
    
#     # Initialize face parser
#     fp = FaceParsing(
#         left_cheek_width=args.left_cheek_width,
#         right_cheek_width=args.right_cheek_width
#     )
    
#     # Process first frame
#     x1, y1, x2, y2 = bbox
#     y2 = y2 + args.extra_margin
#     y2 = min(y2, frame.shape[0])
#     crop_frame = frame[y1:y2, x1:x2]
#     crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
    
#     # Generate random audio features
#     random_audio = torch.randn(1, 50, 384, device=device, dtype=weight_dtype)
#     audio_feature = pe(random_audio)
    
#     # Get latents
#     latents = vae.get_latents_for_unet(crop_frame)
#     latents = latents.to(dtype=weight_dtype)
    
#     # Generate prediction results
#     pred_latents = unet.model(latents, timesteps, encoder_hidden_states=audio_feature).sample
#     recon = vae.decode_latents(pred_latents)
    
#     # Inpaint back to original image
#     res_frame = recon[0]
#     res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
    
#     # 가림 감지 기능을 포함한 블렌딩 적용
#     combine_frame = get_image(frame, res_frame, [x1, y1, x2, y2], 
#                              mode=args.parsing_mode, fp=fp,
#                              enable_occlusion_detection=enable_occlusion_detection,
#                              occlusion_sensitivity=occlusion_sensitivity)
#     # Inpaint back to original image
#     res_frame = recon[0]
#     res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
#     combine_frame = get_image(frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
    
#     # Save results (no need to convert color space again since get_image already returns RGB format)
#     debug_result_path = os.path.join(args.result_dir, "debug_result.png")
#     cv2.imwrite(debug_result_path, combine_frame)
    
#     # Create information text
#     info_text = f"Parameter information:\n" + \
#                 f"bbox_shift: {bbox_shift}\n" + \
#                 f"extra_margin: {extra_margin}\n" + \
#                 f"parsing_mode: {parsing_mode}\n" + \
#                 f"left_cheek_width: {left_cheek_width}\n" + \
#                 f"right_cheek_width: {right_cheek_width}\n" + \
#                 f"enable_occlusion_detection: {enable_occlusion_detection}\n" + \
#                 f"occlusion_sensitivity: {occlusion_sensitivity}\n" + \
#                 f"Detected face coordinates: [{x1}, {y1}, {x2}, {y2}]"
#     # Create information text
#     info_text = f"Parameter information:\n" + \
#                 f"bbox_shift: {bbox_shift}\n" + \
#                 f"extra_margin: {extra_margin}\n" + \
#                 f"parsing_mode: {parsing_mode}\n" + \
#                 f"left_cheek_width: {left_cheek_width}\n" + \
#                 f"right_cheek_width: {right_cheek_width}\n" + \
#                 f"Detected face coordinates: [{x1}, {y1}, {x2}, {y2}]"
    
#     return cv2.cvtColor(combine_frame, cv2.COLOR_RGB2BGR), info_text

# def print_directory_contents(path):
#     for child in os.listdir(path):
#         child_path = os.path.join(path, child)
#         if os.path.isdir(child_path):
#             print(child_path)

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
              progress=gr.Progress(track_tqdm=True)):
    """
    메인 추론 함수 - 오디오와 비디오를 입력받아 말하는 얼굴을 생성
    
    동작 순서:
    1. 파라미터 설정 및 초기화
    2. 입력 비디오에서 프레임 추출
    3. 오디오에서 특징 추출
    4. 입력 이미지 전처리 (랜드마크 및 바운딩박스 추출)
    5. 배치 단위로 추론 실행
    6. 생성된 이미지를 원본 비디오에 합성
    7. 최종 비디오 생성 및 오디오 합성
    """
    
    # ===== 1단계: 파라미터 설정 및 초기화 =====
    # inference.py와 동일한 기본 파라미터 설정
    args_dict = {
        "result_dir": './results/output', 
        "fps": 50, 
        "batch_size": 8, 
        "output_vid_name": '', 
        "use_saved_coord": False,
        "audio_padding_length_left": 2,
        "audio_padding_length_right": 2,
        "version": "v15",  # v15 버전 고정 사용
        "extra_margin": extra_margin,
        "parsing_mode": parsing_mode,
        "left_cheek_width": left_cheek_width,
        "right_cheek_width": right_cheek_width
    }
    args = Namespace(**args_dict)

    # ffmpeg 설치 여부 확인
    if not fast_check_ffmpeg():
        print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

    # 출력 파일명 생성
    input_basename = os.path.basename(video_path).split('.')[0]
    audio_basename = os.path.basename(audio_path).split('.')[0]
    output_basename = f"{input_basename}_{audio_basename}"
    
    # 임시 디렉토리 생성
    temp_dir = os.path.join(args.result_dir, f"{args.version}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 결과 저장 경로 설정
    result_img_save_path = os.path.join(temp_dir, output_basename)
    crop_coord_save_path = os.path.join(args.result_dir, "../", input_basename+".pkl")
    os.makedirs(result_img_save_path, exist_ok=True)

    if args.output_vid_name == "":
        output_vid_name = os.path.join(temp_dir, output_basename+".mp4")
    else:
        output_vid_name = os.path.join(temp_dir, args.output_vid_name)
        
    # ===== 2단계: 입력 비디오에서 프레임 추출 =====
    if get_file_type(video_path) == "video":
        # 비디오 파일인 경우: 프레임을 추출하여 이미지로 저장
        save_dir_full = os.path.join(temp_dir, input_basename)
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
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features, 
        device, 
        weight_dtype, 
        whisper, 
        librosa_length,
        fps=fps,
        audio_padding_length_left=args.audio_padding_length_left,
        audio_padding_length_right=args.audio_padding_length_right,
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
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
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
    
    # ===== 6단계: 배치 단위로 추론 실행 =====
    print("start inference")
    video_num = len(whisper_chunks)
    batch_size = args.batch_size
    # 데이터 생성기 초기화
    gen = datagen(
        whisper_chunks=whisper_chunks,
        vae_encode_latents=input_latent_list_cycle,
        batch_size=batch_size,
        delay_frame=0,
        device=device,
    )
    res_frame_list = []
    # 배치 단위로 추론 실행
    for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
        # 오디오 특징을 위치 인코딩
        audio_feature_batch = pe(whisper_batch)
        # 잠재 벡터를 모델 가중치 타입과 일치하도록 변환
        latent_batch = latent_batch.to(dtype=weight_dtype)
        
        # UNet 모델을 사용하여 새로운 잠재 벡터 생성
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        # VAE를 사용하여 잠재 벡터를 이미지로 디코딩
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)
            
    # ===== 7단계: 생성된 이미지를 원본 비디오에 합성 =====
    print("pad talking image to original video")
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        bbox = coord_list_cycle[i%(len(coord_list_cycle))]
        ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
        x1, y1, x2, y2 = bbox
        y2 = y2 + args.extra_margin
        y2 = min(y2, frame.shape[0])
        try:
            # 생성된 이미지를 원본 크기로 리사이즈
            res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
        except:
            continue
        
        # 가림 감지 기능을 포함한 v15 버전 블렌딩 사용
        combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], 
                                 mode=args.parsing_mode, fp=fp,
                                 enable_occlusion_detection=enable_occlusion_detection,
                                 occlusion_sensitivity=occlusion_sensitivity)
        # v15 버전 블렌딩을 사용하여 원본 이미지와 합성
        combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
            
        # 합성된 프레임을 파일로 저장
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)
        
    # ===== 8단계: 최종 비디오 생성 =====
    # 프레임 레이트 설정
    fps = 50
    # 임시 출력 비디오 경로
    output_video = 'temp.mp4'

    # 유효한 이미지 파일 필터링 함수
    def is_valid_image(file):
        pattern = re.compile(r'\d{8}\.png')
        return pattern.match(file)

    # 저장된 이미지들을 읽어서 리스트로 변환
    images = []
    files = [file for file in os.listdir(result_img_save_path) if is_valid_image(file)]
    files.sort(key=lambda x: int(x.split('.')[0]))

    for file in files:
        filename = os.path.join(result_img_save_path, file)
        images.append(imageio.imread(filename))
        

    # 이미지들을 비디오로 저장
    imageio.mimwrite(output_video, images, 'FFMPEG', fps=fps, codec='libx264', pixelformat='yuv420p')

    # ===== 9단계: 오디오와 비디오 합성 =====
    input_video = './temp.mp4'
    # 입력 비디오와 오디오 파일 존재 여부 확인
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # 비디오 정보 읽기
    reader = imageio.get_reader(input_video)
    fps = reader.get_meta_data()['fps']  # Get original video frame rate
    reader.close() # Otherwise, error on win11: PermissionError: [WinError 32] Another program is using this file, process cannot access. : 'temp.mp4'
    # Store frames in list
    frames = images
    
    print(len(frames))

    # 비디오 클립 로드
    video_clip = VideoFileClip(input_video)

    # 오디오 클립 로드
    audio_clip = AudioFileClip(audio_path)

    # 비디오에 오디오 설정
    video_clip = video_clip.set_audio(audio_clip)

    # 최종 비디오 파일로 저장
    video_clip.write_videofile(output_vid_name, codec='libx264', audio_codec='aac',fps=50)

    # 임시 파일 정리
    os.remove("temp.mp4")
    #shutil.rmtree(result_img_save_path)
    print(f"result is save to {output_vid_name}")
    return output_vid_name,bbox_shift_text



# 1번쨰 gpu 사용하도록 임시 지정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"=============== Using device: {device} ================")

# 모든 모델 로드 (VAE, UNet, Position Encoder)
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

    # fps를 50으로 변환
    frames = [im for im in reader]
    target_fps = 50
    
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

    # 비디오 저장
    imageio.mimwrite(output_video, target_frames, 'FFMPEG', fps=50, codec='libx264', quality=9, pixelformat='yuv420p')
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
                                                 info="값이 높을수록 더 민감하게 가림을 감지합니다")
            
            bbox_shift_scale = gr.Textbox(label="'left_cheek_width'와 'right_cheek_width' 파라미터는 파싱 모델이 'jaw'일 때 좌우 볼 편집 범위를 결정합니다. 'extra_margin' 파라미터는 턱의 움직임 범위를 결정합니다. 사용자는 이 세 파라미터를 자유롭게 조정하여 더 나은 인페인팅 결과를 얻을 수 있습니다. 가림 감지 기능은 마이크 등의 물체에 의해 얼굴이 가려진 부분에서 자연스러운 립싱크를 제공합니다.")

            with gr.Row():
                # debug_btn = gr.Button("1. Test Inpainting ")
                btn = gr.Button("Generate")
        with gr.Column():
            # debug_image = gr.Image(label="Test Inpainting Result (First Frame)")
            # debug_info = gr.Textbox(label="Parameter Information", lines=5)
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
            occlusion_sensitivity
        ],
        outputs=[out1,bbox_shift_scale]
    )
    # debug_btn.click(
    #     fn=debug_inpainting,
    #     inputs=[
    #         video,
    #         bbox_shift,
    #         extra_margin,
    #         parsing_mode,
    #         left_cheek_width,
    #         right_cheek_width,
    #         enable_occlusion_detection,
    #         occlusion_sensitivity
    #     ],
    #     outputs=[debug_image, debug_info]
    # )
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
demo.queue().launch(
    share=args.share, 
    debug=True, 
    server_name="10.202.15.248",
    server_port=8000
)
