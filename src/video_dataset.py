import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
import os
from PIL import Image

class VideoFrameDataset(Dataset):
    """
    Dataset que carga frames (imágenes) de videos
    Estructura esperada:
        root_dir/
            fake/
                imagen1.jpg
                imagen2.jpg
            real/
                imagen1.jpg
                imagen2.jpg
    """
    def __init__(self, root_dir, frames_per_video=5, transform=None):
        self.samples = []
        self.transform = transform
        
        # fake = 0, real = 1
        for label, subfolder in enumerate(['fake', 'real']):
            folder = os.path.join(root_dir, subfolder)
            if not os.path.exists(folder):
                print(f"⚠️ Carpeta no encontrada: {folder}")
                continue
            
            # Buscar imágenes (JPG y PNG)
            images = glob(os.path.join(folder, '*.jpg')) + \
                    glob(os.path.join(folder, '*.jpeg')) + \
                    glob(os.path.join(folder, '*.png'))
            
            for img_path in images:
                self.samples.append((img_path, label))
        
        print(f"✅ Dataset cargado: {len(self.samples)} imágenes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Cargar imagen
        image = Image.open(img_path).convert('RGB')
        
        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)
        
        return image, label
