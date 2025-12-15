"""
Servidor web para DeepFakeShield
Aplicación Flask para demo interactiva
"""

from flask import Flask, render_template, request, jsonify
import torch
from pathlib import Path
import sys
import os
from werkzeug.utils import secure_filename
import tempfile

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from predict import DeepfakePredictor
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False
    print("⚠️ Módulo predict no disponible. Modo demo sin predicción real.")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['UPLOAD_FOLDER'] = 'uploads'

# Crear carpeta de uploads si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Extensiones permitidas
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No se proporcionó ningún video'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Formato de video no permitido'}), 400
    
    try:
        # Guardar archivo temporal
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Obtener parámetros
        num_frames = int(request.form.get('frames', 10))
        threshold = float(request.form.get('threshold', 0.5))
        
        if not PREDICTOR_AVAILABLE:
            # Modo demo: respuesta simulada
            import random
            import time
            time.sleep(2)  # Simular procesamiento
            
            is_fake = random.choice([True, False])
            confidence = random.uniform(0.7, 0.99)
            
            result = {
                'status': 'success',
                'prediction': 'FAKE' if is_fake else 'REAL',
                'confidence': confidence,
                'fake_ratio': random.uniform(0.3, 0.7) if is_fake else random.uniform(0.0, 0.3),
                'total_frames': num_frames,
                'frames_with_face': random.randint(int(num_frames * 0.6), num_frames),
                'demo_mode': True,
                'message': '⚠️ Modo demo - Resultados simulados (instala los modelos para análisis real)'
            }
        else:
            # Análisis real
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            model_path = 'checkpoints/demo_model.pth'
            if not Path(model_path).exists():
                return jsonify({
                    'error': 'Modelo no encontrado. Ejecuta: python create_demo_model.py'
                }), 500
            
            predictor = DeepfakePredictor(model_path=model_path, device=device)
            
            analysis_result = predictor.predict_video(
                video_path=filepath,
                num_frames=num_frames,
                threshold=threshold
            )
            
            result = {
                'status': 'success',
                'prediction': analysis_result['final_prediction'],
                'confidence': float(analysis_result['confidence']),
                'fake_ratio': float(analysis_result['fake_ratio']),
                'total_frames': analysis_result['total_frames'],
                'frames_with_face': analysis_result['frames_with_face'],
                'demo_mode': False
            }
        
        # Limpiar archivo temporal
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Error al procesar video: {str(e)}'}), 500

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'predictor_available': PREDICTOR_AVAILABLE,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })

if __name__ == '__main__':
    print("="*70)
    print("🛡️  DeepFakeShield - Servidor Web")
    print("="*70)
    print(f"📍 URL: http://127.0.0.1:5000")
    print(f"🤖 Predictor: {'Disponible' if PREDICTOR_AVAILABLE else 'Modo Demo'}")
    print(f"🔧 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*70)
    print("\n🚀 Abre tu navegador en http://127.0.0.1:5000\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
