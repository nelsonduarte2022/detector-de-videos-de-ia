
"""
Entrenamiento de modelo para DeepFakeShield
Este script entrena EfficientNet-B4 para clasificar videos como reales o deepfakes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from pathlib import Path
import os
import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, classification_report
import matplotlib.pyplot as plt

from models.cnn import get_model
from video_dataset import VideoFrameDataset

def train_model(dataset_dir='dataset_frames', epochs=10, batch_size=16, lr=1e-4):
    print("="*70)
    print("🚀 Iniciando Entrenamiento - DeepFakeShield")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Dispositivo: {device}")
    
    print(f"\n📦 Cargando modelo EfficientNet-B4...")
    model = get_model('efficientnet_b4', pretrained=True)
    model = model.to(device)
    print(f"✅ Modelo cargado")

    # Transformaciones con Data Augmentation para entrenamiento
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Transformaciones para validación (sin augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print(f"\n📂 Cargando dataset desde: {dataset_dir}")
    full_dataset = VideoFrameDataset(dataset_dir, transform=val_transform)

    # Divide en train/val (80/20)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)), [train_size, val_size]
    )
    
    # Crear datasets separados con sus respectivas transformaciones
    train_dataset = VideoFrameDataset(dataset_dir, transform=train_transform)
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    
    val_dataset = VideoFrameDataset(dataset_dir, transform=val_transform)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    
    print(f"\n📊 Split:")
    print(f"   Entrenamiento: {train_size} imágenes (con augmentation)")
    print(f"   Validación: {val_size} imágenes")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n⚙️ Hiperparámetros:")
    print(f"   Épocas: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print("\n" + "="*70)

    best_acc = 0.0
    for epoch in range(epochs):
        print(f"\n📈 Época {epoch+1}/{epochs}")
        print("-"*70)
        
        # Entrenamiento
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
            if (i + 1) % 10 == 0:
                print(f"   Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # Validación con métricas avanzadas
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                # Probabilidades y predicciones
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                # Guardar para métricas
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probabilidad clase "Fake"
        
        val_acc = correct / total if total > 0 else 0
        val_loss = val_loss / total if total > 0 else 0
        
        # Calcular métricas avanzadas
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        try:
            roc_auc = roc_auc_score(all_labels, all_probs)
        except:
            roc_auc = 0.0
        
        print(f"\n📊 Resultados Época {epoch+1}:")
        print(f"   Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.4f}")
        print(f"\n📈 Métricas Avanzadas (Validación):")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # Guardar mejor modelo con métricas completas
            Path('checkpoints').mkdir(exist_ok=True)
            checkpoint_path = 'checkpoints/demo_model.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch+1,
                'model_name': 'efficientnet_b4',
                'val_accuracy': val_acc,
                'train_accuracy': train_acc,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
                'val_loss': val_loss,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            }, checkpoint_path)
            print(f"   ✅ Mejor modelo guardado! (Val Acc: {val_acc:.4f}, F1: {f1:.4f})")
    
    print("\n" + "="*70)
    print("🎉 Entrenamiento completado!")
    print(f"🏆 Mejor Val Accuracy: {best_acc:.4f}")
    print(f"💾 Modelo guardado en: checkpoints/demo_model.pth")
    print("="*70)
    
    # Generar reporte de clasificación final
    print("\n📋 Reporte de Clasificación Final:")
    print(classification_report(all_labels, all_preds, 
                                target_names=['Real', 'Fake'], 
                                digits=4))
    
    print("\n📝 Siguiente paso:")
    print("   python app.py")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenar modelo DeepFakeShield')
    parser.add_argument('--dataset', type=str, default='dataset_frames',
                       help='Directorio con frames (fake/real)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Tamaño del batch')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    train_model(args.dataset, args.epochs, args.batch_size, args.lr)
