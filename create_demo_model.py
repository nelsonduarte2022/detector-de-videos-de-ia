"""
Script para crear un modelo demo sin entrenamiento
Útil para probar la aplicación web sin entrenar
"""

import torch
import torch.nn as nn
from pathlib import Path


def create_demo_model():
    """Crea un modelo demo inicializado aleatoriamente"""
    print("="*70)
    print("🤖 Creando Modelo Demo")
    print("="*70)
    
    # Importar el modelo
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    from models.cnn import get_model
    
    # Crear modelo
    print("\n📦 Creando EfficientNet-B4...")
    model = get_model('efficientnet_b4', pretrained=True)
    
    # Crear directorio de checkpoints
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Guardar modelo
    checkpoint_path = checkpoint_dir / 'demo_model.pth'
    
    print(f"\n💾 Guardando modelo en {checkpoint_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': 0,
        'model_name': 'efficientnet_b4',
        'val_accuracy': 0.0,
    }, checkpoint_path)
    
    print(f"\n✅ Modelo demo creado exitosamente!")
    print(f"📍 Ubicación: {checkpoint_path.absolute()}")
    print("\n⚠️  NOTA: Este es un modelo demo sin entrenar.")
    print("   Para obtener predicciones reales, entrena el modelo con:")
    print("   python src/train.py")
    print("="*70)


if __name__ == "__main__":
    create_demo_model()
