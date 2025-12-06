"""
Script simplificado para probar detección de deepfake en video.
"""

import sys
from pathlib import Path

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import argparse
import torch
from predict import DeepfakePredictor


def main():
    parser = argparse.ArgumentParser(description='Detectar deepfake en video')
    parser.add_argument('--video', type=str, required=True, 
                       help='Ruta al video a analizar')
    parser.add_argument('--model', type=str, 
                       default='checkpoints/demo_model.pth',
                       help='Ruta al modelo entrenado')
    parser.add_argument('--frames', type=int, default=10,
                       help='Número de frames a analizar (default: 10)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Umbral de decisión (default: 0.5)')
    
    args = parser.parse_args()
    
    # Verificar que el video existe
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Error: Video no encontrado: {args.video}")
        print(f"\n💡 Proporciona la ruta completa al video, ejemplo:")
        print(f'   python test_video.py --video "C:\\Users\\missa\\Videos\\mi_video.mp4"')
        return
    
    # Verificar que el modelo existe
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Modelo no encontrado: {args.model}")
        print(f"\n💡 Primero crea el modelo demo:")
        print(f"   python create_demo_model.py")
        return
    
    print("="*70)
    print("🛡️  DeepFakeShield - Detección de Videos Deepfake")
    print("="*70)
    print(f"📹 Video: {video_path.name}")
    print(f"🤖 Modelo: {model_path.name}")
    print(f"🎞️  Frames a analizar: {args.frames}")
    print(f"⚖️  Umbral: {args.threshold}")
    print("-"*70)
    
    # Cargar predictor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Cargando modelo en {device}...")
    
    predictor = DeepfakePredictor(
        model_path=str(model_path),
        device=device
    )
    
    # Hacer predicción
    print(f"🔍 Analizando video...\n")
    
    result = predictor.predict_video(
        video_path=str(video_path),
        num_frames=args.frames,
        threshold=args.threshold
    )
    
    # Mostrar resultados
    print("="*70)
    print("📊 RESULTADOS")
    print("="*70)
    
    prediction = result['final_prediction']
    confidence = result['confidence']
    fake_ratio = result['fake_ratio']
    
    # Emoji según resultado
    emoji = "🔴" if prediction == "FAKE" else "🟢"
    
    print(f"\n{emoji} Predicción Final: {prediction}")
    print(f"📈 Confianza: {confidence:.2%}")
    print(f"📊 Ratio de frames fake: {fake_ratio:.2%}")
    print(f"🎞️  Frames analizados: {result['total_frames']}")
    print(f"🎭 Frames con rostro detectado: {result['frames_with_face']}")
    
    if result['frames_with_face'] < result['total_frames'] * 0.5:
        print(f"\n⚠️  ADVERTENCIA: Se detectaron pocos rostros ({result['frames_with_face']}/{result['total_frames']})")
        print("   El video podría no contener rostros claros o estar en baja calidad.")
    
    print("\n" + "-"*70)
    print("💡 Detalles por frame:")
    print("-"*70)
    
    for i, frame_result in enumerate(result['frame_predictions'][:5], 1):
        pred = frame_result['prediction']
        conf = frame_result['confidence']
        emoji = "🔴" if pred == "FAKE" else "🟢"
        print(f"  Frame {i}: {emoji} {pred} (confianza: {conf:.2%})")
    
    if len(result['frame_predictions']) > 5:
        print(f"  ... y {len(result['frame_predictions']) - 5} frames más")
    
    print("\n" + "="*70)
    
    # Interpretación del resultado
    print("\n🎯 INTERPRETACIÓN:")
    if prediction == "FAKE":
        if confidence > 0.8:
            print("   ⚠️  ALTA probabilidad de ser deepfake")
            print("   ⚠️  Se recomienda verificación adicional")
        else:
            print("   ⚠️  Posible deepfake, pero con confianza moderada")
            print("   💭 Podría ser manipulación sutil o baja calidad")
    else:
        if confidence > 0.8:
            print("   ✅ ALTA probabilidad de ser video auténtico")
            print("   ✅ No se detectaron signos de manipulación")
        else:
            print("   ✅ Probablemente auténtico, pero con cierta incertidumbre")
            print("   💭 Podría tener compresión o artefactos naturales")
    
    print("\n" + "="*70)
    print("✅ Análisis completado")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
