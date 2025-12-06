"""
Crear subset pequeño del dataset para entrenar rápido en CPU
"""
import shutil
from pathlib import Path
from tqdm import tqdm
import random

def create_small_dataset(source_dir, dest_dir, samples_per_class=1000):
    """
    Crea un subset pequeño del dataset
    
    Args:
        source_dir: Directorio con el dataset completo
        dest_dir: Directorio para el subset pequeño
        samples_per_class: Número de muestras por clase (real/fake)
    """
    random.seed(42)
    
    for split in ['train', 'val']:
        print(f"\n📁 Procesando {split}...")
        
        for label in ['real', 'fake']:
            source_path = Path(source_dir) / split / label
            dest_path = Path(dest_dir) / f"{split}_small" / label
            dest_path.mkdir(parents=True, exist_ok=True)
            
            # Obtener todas las imágenes
            all_images = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
            
            # Seleccionar subset aleatorio
            selected = random.sample(all_images, min(samples_per_class, len(all_images)))
            
            print(f"  {split}/{label}: Copiando {len(selected)} de {len(all_images)} imágenes...")
            
            # Copiar imágenes
            for img in tqdm(selected, desc=f"  {split}/{label}"):
                shutil.copy2(img, dest_path / img.name)
    
    print("\n✅ Dataset pequeño creado exitosamente!")
    print(f"📊 Total por split:")
    print(f"   Train: {samples_per_class * 2} imágenes")
    print(f"   Val: {samples_per_class * 2} imágenes")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Crear subset pequeño del dataset')
    parser.add_argument('--source', type=str, default='data',
                       help='Directorio fuente')
    parser.add_argument('--dest', type=str, default='data',
                       help='Directorio destino')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Muestras por clase')
    
    args = parser.parse_args()
    
    create_small_dataset(args.source, args.dest, args.samples)
