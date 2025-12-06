"""
Script para organizar el dataset DFDC en estructura train/val/test
"""
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

def organize_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Organiza imágenes en carpetas train/val/test con subcarpetas real/fake
    
    Asume que los nombres de archivo contienen información sobre real/fake
    o que hay un archivo de metadatos
    """
    print("🔍 Buscando imágenes...")
    source_path = Path(source_dir)
    
    # Buscar todas las imágenes
    all_images = list(source_path.rglob("*.jpg")) + list(source_path.rglob("*.png"))
    print(f"✅ Encontradas {len(all_images)} imágenes")
    
    # Buscar archivo de metadatos si existe
    metadata_file = source_path / "metadata.json"
    labels_file = source_path / "labels.csv"
    
    # Diccionario para almacenar labels
    image_labels = {}
    
    # Intentar cargar metadatos
    if metadata_file.exists():
        print("📄 Cargando metadata.json...")
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            for video_name, info in metadata.items():
                label = info.get('label', 'REAL')
                # Buscar imágenes que correspondan a este video
                for img in all_images:
                    if video_name.replace('.mp4', '') in img.name:
                        image_labels[img] = 'fake' if label == 'FAKE' else 'real'
    elif labels_file.exists():
        print("📄 Cargando labels.csv...")
        import pandas as pd
        df = pd.read_csv(labels_file)
        for _, row in df.iterrows():
            video_name = row.get('filename', row.get('video', ''))
            label = row.get('label', 'REAL')
            for img in all_images:
                if video_name.replace('.mp4', '') in img.name:
                    image_labels[img] = 'fake' if label == 'FAKE' or label == 1 else 'real'
    else:
        print("⚠️  No se encontró archivo de metadatos.")
        print("💡 Asumiendo distribución 50/50 basada en nombres de archivo...")
        
        # Estrategia: usar nombres de archivos únicos de video
        video_names = set()
        for img in all_images:
            # Extraer nombre base del video (ej: "aagfhgtpmv" de "aagfhgtpmv.mp4_face_1.jpg")
            video_name = img.stem.split('_face_')[0].replace('.mp4', '')
            video_names.add(video_name)
        
        video_names = list(video_names)
        random.shuffle(video_names)
        
        # Dividir videos en real/fake (50/50)
        mid = len(video_names) // 2
        fake_videos = set(video_names[:mid])
        real_videos = set(video_names[mid:])
        
        for img in all_images:
            video_name = img.stem.split('_face_')[0].replace('.mp4', '')
            if video_name in fake_videos:
                image_labels[img] = 'fake'
            else:
                image_labels[img] = 'real'
        
        print(f"📊 Videos fake: {len(fake_videos)}, Videos real: {len(real_videos)}")
    
    # Separar por clase
    real_images = [img for img, label in image_labels.items() if label == 'real']
    fake_images = [img for img, label in image_labels.items() if label == 'fake']
    
    print(f"✅ Imágenes reales: {len(real_images)}")
    print(f"✅ Imágenes fake: {len(fake_images)}")
    
    if len(real_images) == 0 or len(fake_images) == 0:
        print("❌ Error: No se pudieron separar imágenes reales y fake")
        return
    
    # Dividir cada clase en train/val/test
    def split_data(images, train_r, val_r, test_r):
        train, temp = train_test_split(images, train_size=train_r, random_state=42)
        val_ratio_adjusted = val_r / (val_r + test_r)
        val, test = train_test_split(temp, train_size=val_ratio_adjusted, random_state=42)
        return train, val, test
    
    real_train, real_val, real_test = split_data(real_images, train_ratio, val_ratio, test_ratio)
    fake_train, fake_val, fake_test = split_data(fake_images, train_ratio, val_ratio, test_ratio)
    
    print(f"\n📊 División del dataset:")
    print(f"   Train: {len(real_train)} real + {len(fake_train)} fake = {len(real_train) + len(fake_train)}")
    print(f"   Val:   {len(real_val)} real + {len(fake_val)} fake = {len(real_val) + len(fake_val)}")
    print(f"   Test:  {len(real_test)} real + {len(fake_test)} fake = {len(real_test) + len(fake_test)}")
    
    # Crear estructura de directorios
    dest_path = Path(dest_dir)
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            (dest_path / split / label).mkdir(parents=True, exist_ok=True)
    
    # Función para copiar imágenes
    def copy_images(images, split, label):
        dest = dest_path / split / label
        print(f"\n📁 Copiando {len(images)} imágenes a {split}/{label}...")
        for img in tqdm(images):
            shutil.copy2(img, dest / img.name)
    
    # Copiar todas las imágenes
    copy_images(real_train, 'train', 'real')
    copy_images(fake_train, 'train', 'fake')
    copy_images(real_val, 'val', 'real')
    copy_images(fake_val, 'val', 'fake')
    copy_images(real_test, 'test', 'real')
    copy_images(fake_test, 'test', 'fake')
    
    print("\n✅ ¡Dataset organizado exitosamente!")
    print(f"📂 Ubicación: {dest_path.absolute()}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Organizar dataset DFDC')
    parser.add_argument('--source', type=str, default='data/raw/DFDCDFDC/DFDCDFDC',
                       help='Directorio fuente con las imágenes')
    parser.add_argument('--dest', type=str, default='data',
                       help='Directorio destino')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Proporción para entrenamiento')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Proporción para validación')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Proporción para test')
    
    args = parser.parse_args()
    
    # Validar proporciones
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 0.01:
        print(f"❌ Error: Las proporciones deben sumar 1.0 (actual: {total})")
        exit(1)
    
    organize_dataset(
        args.source,
        args.dest,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )
