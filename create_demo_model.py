"""
Script para crear un modelo demo pre-inicializado para pruebas rápidas.
Este modelo NO está entrenado, pero permite probar el pipeline completo.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.cnn import get_model


def create_demo_model(output_path: str = "checkpoints/demo_model.pth"):
    """
    Crea un modelo demo con pesos aleatorios para pruebas.
    
    Args:
        output_path: Ruta donde guardar el modelo
    """
    print("🔧 Creando modelo demo EfficientNet-B4...")
    
    # Crear modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model('efficientnet_b4', pretrained=True)
    model = model.to(device)
    
    # Crear checkpoint con la estructura correcta
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': 0,
        'model_name': 'efficientnet_b4',
        'val_accuracy': 0.85,  # Valor demo
        'optimizer_state_dict': None,
        'train_loss': 0.3,
        'val_loss': 0.35
    }
    
    # Crear directorio si no existe
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar modelo
    torch.save(checkpoint, output_path)
    
    print(f"✅ Modelo demo guardado en: {output_path}")
    print(f"📊 Dispositivo: {device}")
    print(f"🔢 Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    print("\n⚠️  NOTA: Este modelo usa pesos pre-entrenados de ImageNet.")
    print("   Para detección real de deepfakes, entrena con: python src/train.py\n")


if __name__ == "__main__":
    create_demo_model()
