"""
Módulo de predicción para DeepFakeShield
Procesa videos y detecta si son deepfakes
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image


class DeepfakePredictor:
    def __init__(self, model_path, device='cpu'):
        """
        Inicializa el predictor
        
        Args:
            model_path: Ruta al modelo entrenado (.pth)
            device: 'cuda' o 'cpu'
        """
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        
        # Transformaciones para las imágenes
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path):
        """Carga el modelo entrenado"""
        from models.cnn import get_model
        
        # Cargar checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Crear modelo
        model = get_model('efficientnet_b4', pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def extract_frames(self, video_path, num_frames=10):
        """
        Extrae frames uniformemente distribuidos del video
        
        Args:
            video_path: Ruta al video
            num_frames: Número de frames a extraer
            
        Returns:
            Lista de frames (numpy arrays)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError("El video no tiene frames")
        
        # Calcular índices de frames a extraer
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convertir BGR a RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        return frames
    
    def predict_frame(self, frame):
        """
        Predice si un frame es fake o real
        
        Args:
            frame: Numpy array (RGB)
            
        Returns:
            dict con prediction y confidence
        """
        # Convertir a PIL Image
        image = Image.fromarray(frame)
        
        # Aplicar transformaciones
        image_tensor = self.transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Predicción
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # 0 = fake, 1 = real (según tu dataset)
        prediction = "FAKE" if predicted.item() == 0 else "REAL"
        confidence_value = confidence.item()
        
        return {
            'prediction': prediction,
            'confidence': confidence_value,
            'probabilities': {
                'fake': probabilities[0][0].item(),
                'real': probabilities[0][1].item()
            }
        }
    
    def predict_video(self, video_path, num_frames=10, threshold=0.5):
        """
        Predice si un video es deepfake
        
        Args:
            video_path: Ruta al video
            num_frames: Número de frames a analizar
            threshold: Umbral para considerar fake (0.0-1.0)
            
        Returns:
            dict con resultados del análisis
        """
        print(f"🎥 Extrayendo {num_frames} frames del video...")
        frames = self.extract_frames(video_path, num_frames)
        
        if not frames:
            return {
                'error': 'No se pudieron extraer frames del video',
                'final_prediction': 'ERROR',
                'confidence': 0.0
            }
        
        print(f"🔍 Analizando {len(frames)} frames...")
        
        # Analizar cada frame
        frame_predictions = []
        fake_count = 0
        
        for i, frame in enumerate(frames):
            result = self.predict_frame(frame)
            frame_predictions.append(result)
            
            if result['prediction'] == 'FAKE':
                fake_count += 1
        
        # Calcular métricas generales
        fake_ratio = fake_count / len(frames)
        
        # Decisión final basada en el threshold
        if fake_ratio >= threshold:
            final_prediction = "FAKE"
            confidence = fake_ratio
        else:
            final_prediction = "REAL"
            confidence = 1 - fake_ratio
        
        return {
            'final_prediction': final_prediction,
            'confidence': confidence,
            'fake_ratio': fake_ratio,
            'total_frames': len(frames),
            'frames_with_face': len(frames),  # Simplificado
            'frame_predictions': frame_predictions
        }
