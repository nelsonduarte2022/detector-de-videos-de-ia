"""
Script para extraer frames del dataset de videos ya organizados
"""

import cv2
import os
from pathlib import Path
from tqdm import tqdm


def extract_frames_from_video(video_path, output_dir, max_frames=30):
    """Extrae frames de un video"""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return 0
    
    # Extraer frames uniformemente distribuidos
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        import numpy as np
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    
    video_name = Path(video_path).stem
    extracted = 0
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            frame_filename = f"{video_name}_frame_{idx:04d}.jpg"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            extracted += 1
    
    cap.release()
    return extracted


def extract_dataset_frames(dataset_dir='dataset', frames_per_video=30):
    """
    Extrae frames de videos en dataset/fake y dataset/real
    Crea dataset_frames/fake y dataset_frames/real con las imágenes
    """
    print("="*70)
    print("🎬 Extrayendo Frames del Dataset")
    print("="*70)
    
    dataset_path = Path(dataset_dir)
    output_path = Path('dataset_frames')
    
    # Procesar fake y real
    for label in ['fake', 'real']:
        video_dir = dataset_path / label
        output_dir = output_path / label
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Buscar videos
        videos = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi')) + \
                 list(video_dir.glob('*.mov')) + list(video_dir.glob('*.mkv'))
        
        print(f"\n📹 Procesando {len(videos)} videos {label.upper()}...")
        
        total_frames = 0
        for video_path in tqdm(videos, desc=f"Extrayendo {label}"):
            try:
                extracted = extract_frames_from_video(
                    video_path, 
                    output_dir, 
                    max_frames=frames_per_video
                )
                total_frames += extracted
            except Exception as e:
                print(f"⚠️ Error en {video_path.name}: {e}")
        
        print(f"✅ {label.upper()}: {total_frames} frames extraídos")
    
    print("\n" + "="*70)
    print("✅ Extracción completada!")
    print(f"📁 Frames guardados en: {output_path.absolute()}")
    print("\n📝 SIGUIENTE PASO:")
    print("   python src/train.py --dataset dataset_frames")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extraer frames del dataset')
    parser.add_argument('--dataset', type=str, default='dataset',
                       help='Directorio con videos (fake/real)')
    parser.add_argument('--frames', type=int, default=30,
                       help='Frames por video')
    
    args = parser.parse_args()
    
    extract_dataset_frames(args.dataset, args.frames)
