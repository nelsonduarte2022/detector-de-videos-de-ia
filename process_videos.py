"""
Script para extraer frames de videos y organizar dataset
Extrae frames de videos fake y real para entrenar el modelo
"""

import cv2
import os
from pathlib import Path
from tqdm import tqdm
import shutil
import zipfile


def extract_frames_from_video(video_path, output_dir, max_frames=30):
    """
    Extrae frames de un video
    
    Args:
        video_path: Ruta al video
        output_dir: Directorio donde guardar los frames
        max_frames: Número máximo de frames a extraer
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return 0
    
    # Extraer frames uniformemente distribuidos
    frame_indices = []
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames / max_frames
        frame_indices = [int(i * step) for i in range(max_frames)]
    
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


def process_dataset(source_zip, output_dir='dataset', frames_per_video=30):
    """
    Procesa el dataset de videos:
    1. Descomprime el ZIP
    2. Extrae frames de cada video
    3. Organiza en carpetas fake/real
    
    Args:
        source_zip: Ruta al archivo archive.zip
        output_dir: Directorio de salida
        frames_per_video: Frames a extraer por video
    """
    print("="*70)
    print("🎬 Procesando Dataset de Videos Deepfake")
    print("="*70)
    
    source_path = Path(source_zip)
    output_path = Path(output_dir)
    
    # Crear directorios
    fake_dir = output_path / 'fake'
    real_dir = output_path / 'real'
    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    
    # Directorio temporal para extraer el ZIP
    temp_dir = Path('temp_extracted')
    temp_dir.mkdir(exist_ok=True)
    
    print(f"\n📦 Descomprimiendo {source_path.name}...")
    
    try:
        with zipfile.ZipFile(source_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        print("✅ Archivo descomprimido")
        
        # Buscar videos en el directorio extraído
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        all_videos = []
        
        for ext in video_extensions:
            all_videos.extend(temp_dir.rglob(f'*{ext}'))
        
        print(f"\n🎥 Encontrados {len(all_videos)} videos")
        
        if len(all_videos) == 0:
            print("❌ No se encontraron videos en el archivo")
            return
        
        # Procesar videos
        # Asumimos que todos los videos del ZIP son fake
        # Si tienes videos reales, agrégalos a la carpeta dataset/real/
        
        print(f"\n🔄 Extrayendo frames de videos fake...")
        total_frames = 0
        
        for video_path in tqdm(all_videos, desc="Procesando videos"):
            try:
                extracted = extract_frames_from_video(
                    video_path, 
                    fake_dir, 
                    max_frames=frames_per_video
                )
                total_frames += extracted
            except Exception as e:
                print(f"⚠️ Error procesando {video_path.name}: {e}")
        
        print(f"\n✅ Extracción completada!")
        print(f"📊 Total de frames extraídos: {total_frames}")
        print(f"📁 Frames fake guardados en: {fake_dir.absolute()}")
        
        # Limpiar directorio temporal
        print(f"\n🧹 Limpiando archivos temporales...")
        shutil.rmtree(temp_dir)
        
        # Instrucciones para videos reales
        print("\n" + "="*70)
        print("📝 SIGUIENTE PASO:")
        print("="*70)
        print(f"1. Agrega videos REALES a una carpeta temporal")
        print(f"2. Ejecuta este script de nuevo apuntando a esos videos")
        print(f"3. O copia frames de videos reales manualmente a: {real_dir.absolute()}")
        print("\n💡 Necesitas un balance 50/50 de fake y real para entrenar bien")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Procesar dataset de videos')
    parser.add_argument('--zip', type=str, default='archive.zip',
                       help='Ruta al archivo ZIP con videos')
    parser.add_argument('--output', type=str, default='dataset',
                       help='Directorio de salida')
    parser.add_argument('--frames', type=int, default=30,
                       help='Frames por video')
    
    args = parser.parse_args()
    
    process_dataset(args.zip, args.output, args.frames)
