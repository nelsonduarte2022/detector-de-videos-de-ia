"""
Módulo de Interpretabilidad para DeepFakeShield
Implementa Grad-CAM para visualizar qué partes de la imagen influyen en la predicción
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


class GradCAM:
    """
    Implementación de Gradient-weighted Class Activation Mapping (Grad-CAM)
    
    Grad-CAM permite visualizar qué regiones de una imagen son importantes
    para la predicción del modelo de Deep Learning.
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Modelo de PyTorch
            target_layer: Capa objetivo para extraer gradientes (ej: última conv)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Registrar hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook para guardar las activaciones de la capa"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook para guardar los gradientes"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Genera el mapa de calor Grad-CAM
        
        Args:
            input_tensor: Tensor de entrada (1, C, H, W)
            target_class: Clase objetivo (None = clase predicha)
            
        Returns:
            cam: Mapa de calor normalizado (H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()
        
        # Calcular Grad-CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Peso por canal (promedio espacial de gradientes)
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        
        # Combinación lineal ponderada
        cam = (weights * activations).sum(dim=0)  # (H, W)
        
        # ReLU y normalización
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()
    
    def visualize_cam(self, image, cam, alpha=0.5):
        """
        Superpone el mapa de calor sobre la imagen original
        
        Args:
            image: Imagen original (PIL o numpy array)
            cam: Mapa de calor Grad-CAM
            alpha: Transparencia del heatmap
            
        Returns:
            superimposed: Imagen con heatmap superpuesto
        """
        # Convertir imagen a numpy si es necesario
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize CAM a tamaño de imagen
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Convertir CAM a mapa de calor (colormap)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Superponer
        superimposed = heatmap * alpha + image * (1 - alpha)
        superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
        
        return superimposed


def get_target_layer(model, model_name='efficientnet_b4'):
    """
    Obtiene la última capa convolucional del modelo
    
    Args:
        model: Modelo de PyTorch
        model_name: Nombre del modelo
        
    Returns:
        target_layer: Última capa convolucional
    """
    if 'efficientnet' in model_name.lower():
        # Para EfficientNet, la última capa conv está en features[-1]
        return model.features[-1]
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")


def generate_gradcam_visualization(model, image_tensor, original_image, 
                                   model_name='efficientnet_b4', 
                                   save_path=None):
    """
    Función auxiliar para generar y guardar visualización Grad-CAM
    
    Args:
        model: Modelo entrenado
        image_tensor: Tensor de entrada normalizado (1, C, H, W)
        original_image: Imagen original (PIL o numpy)
        model_name: Nombre del modelo
        save_path: Ruta para guardar la imagen (opcional)
        
    Returns:
        superimposed: Imagen con heatmap
        cam: Mapa de calor
    """
    target_layer = get_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)
    
    # Generar CAM
    cam = gradcam.generate_cam(image_tensor)
    
    # Visualizar
    superimposed = gradcam.visualize_cam(original_image, cam)
    
    # Guardar si se especifica
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Crear figura con comparación
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Imagen original
        if isinstance(original_image, Image.Image):
            axes[0].imshow(original_image)
        else:
            axes[0].imshow(original_image)
        axes[0].set_title('Imagen Original')
        axes[0].axis('off')
        
        # Heatmap solo
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Superposición
        axes[2].imshow(superimposed)
        axes[2].set_title('Superposición')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualización Grad-CAM guardada en: {save_path}")
    
    return superimposed, cam


if __name__ == "__main__":
    """
    Ejemplo de uso
    """
    import sys
    from torchvision import transforms
    from models.cnn import get_model
    
    print("🔍 Ejemplo de uso de Grad-CAM\n")
    
    # Cargar modelo
    model_path = 'checkpoints/demo_model.pth'
    if not Path(model_path).exists():
        print("❌ Modelo no encontrado. Entrena primero con: python src/train.py")
        sys.exit(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model('efficientnet_b4', pretrained=False)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Cargar imagen de ejemplo
    image_path = 'dataset_frames/fake'
    image_files = list(Path(image_path).glob('*.jpg'))
    
    if len(image_files) == 0:
        print("❌ No se encontraron imágenes en dataset_frames/fake")
        sys.exit(1)
    
    example_image = image_files[0]
    print(f"📸 Procesando: {example_image.name}")
    
    # Preparar imagen
    original_image = Image.open(example_image).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Generar visualización
    output_path = 'checkpoints/gradcam_example.png'
    superimposed, cam = generate_gradcam_visualization(
        model, image_tensor, original_image, 
        save_path=output_path
    )
    
    # Predicción
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    class_names = ['Real', 'Fake']
    print(f"\n📊 Predicción: {class_names[pred_class]}")
    print(f"🎯 Confianza: {confidence:.2%}")
    print(f"\n✅ Las regiones rojas/amarillas en el heatmap indican áreas")
    print(f"   que el modelo considera más importantes para su decisión.")
