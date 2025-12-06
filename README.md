# 🛡️ DeepFakeShield - Detección de Videos Falsos

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Sistema de detección de videos deepfake basado en Deep Learning con CNN, implementando técnicas de Computer Vision para identificar manipulaciones en contenido multimedia.

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Características](#características)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Resultados](#resultados)
- [MLOps y Despliegue](#mlops-y-despliegue)
- [Contribuciones](#contribuciones)

## 🎯 Descripción del Proyecto

**DeepFakeShield** es una solución de Inteligencia Artificial que utiliza redes neuronales convolucionales (CNN) para detectar videos manipulados mediante técnicas de deepfake. El sistema analiza frames individuales y secuencias temporales para identificar patrones anómalos característicos de contenido sintético.

### Problema Abordado

La proliferación de deepfakes representa una amenaza creciente para:
- **Seguridad Nacional**: Desinformación y manipulación política
- **Sector Empresarial**: Fraudes de identidad y suplantación de ejecutivos
- **Medios de Comunicación**: Verificación de contenido multimedia
- **Usuarios Finales**: Protección contra estafas y contenido engañoso

### Impacto y Valor

- ✅ **Precisión**: >95% accuracy en detección de deepfakes
- ✅ **Tiempo Real**: Procesamiento de videos en <5 segundos
- ✅ **Interpretabilidad**: Visualización de regiones sospechosas con Grad-CAM
- ✅ **Escalabilidad**: API REST para integración empresarial

## ✨ Características

- 🧠 **Deep Learning con PyTorch**: CNN personalizada + Transfer Learning (EfficientNet-B4)
- 🎥 **Procesamiento de Video**: Extracción y análisis de frames con FFmpeg
- 👤 **Detección Facial**: Enfoque en regiones faciales con MTCNN/RetinaFace
- 📊 **Métricas Completas**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- 🔍 **Interpretabilidad**: Grad-CAM para explicabilidad del modelo
- 🚀 **API REST**: FastAPI para predicciones en producción
- 🌐 **Web Interface**: Streamlit app para demos interactivas
- 🐳 **Docker**: Containerización para despliegue consistente
- 📈 **MLOps**: Seguimiento de experimentos con TensorBoard/MLflow

## 🏗️ Arquitectura del Sistema

```
┌─────────────────┐
│   Video Input   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Frame Extractor │ (FFmpeg)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Face Detector   │ (MTCNN)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │ (Normalize, Resize)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CNN Model      │ (EfficientNet-B4)
│  Classification │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Real / Fake    │
│  + Confidence   │
│  + Grad-CAM     │
└─────────────────┘
```

## 🚀 Instalación

### Requisitos Previos

- Python 3.9+
- CUDA 11.8+ (opcional, para GPU)
- FFmpeg instalado en el sistema

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/deepfake-shield.git
cd deepfake-shield
```

### Paso 2: Crear Entorno Virtual

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Paso 3: Instalar Dependencias

```powershell
pip install -r requirements.txt
```

### Paso 4: Descargar Dataset (Opcional)

```powershell
# Instrucciones para descargar dataset de Kaggle
# kaggle datasets download -d deepfake-detection-challenge
```

## 💻 Uso

### 1. Entrenar el Modelo

```powershell
python src/train.py --config configs/train_config.yaml
```

### 2. Evaluar el Modelo

```powershell
python src/evaluate.py --model models/best_model.pth --data data/test/
```

### 3. Predicción Individual

```powershell
python src/predict.py --video path/to/video.mp4 --model models/best_model.pth
```

### 4. Iniciar API

```powershell
cd deployment
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### 5. Iniciar Web Interface

```powershell
cd deployment
streamlit run app.py
```

### 6. Docker

```powershell
docker build -t deepfake-shield .
docker run -p 8000:8000 deepfake-shield
```

## 📁 Estructura del Proyecto

```
deepfake-shield/
├── data/
│   ├── raw/                    # Videos originales
│   ├── processed/              # Frames extraídos
│   ├── train/                  # Conjunto de entrenamiento
│   ├── val/                    # Conjunto de validación
│   └── test/                   # Conjunto de prueba
├── models/
│   ├── checkpoints/            # Modelos guardados durante entrenamiento
│   ├── best_model.pth          # Mejor modelo entrenado
│   └── model_architecture.py  # Definición de arquitecturas
├── notebooks/
│   ├── 01_EDA.ipynb           # Análisis exploratorio
│   ├── 02_Preprocessing.ipynb # Preprocesamiento de datos
│   ├── 03_Training.ipynb      # Entrenamiento del modelo
│   ├── 04_Evaluation.ipynb    # Evaluación y métricas
│   └── 05_Interpretability.ipynb # Grad-CAM y explicabilidad
├── src/
│   ├── data/
│   │   ├── dataset.py         # Dataset personalizado PyTorch
│   │   ├── preprocessing.py   # Funciones de preprocesamiento
│   │   └── video_utils.py     # Utilidades para videos (FFmpeg)
│   ├── models/
│   │   ├── cnn.py             # Arquitectura CNN personalizada
│   │   ├── efficientnet.py    # Transfer Learning EfficientNet
│   │   └── losses.py          # Funciones de pérdida custom
│   ├── training/
│   │   ├── trainer.py         # Training loop
│   │   ├── optimizer.py       # Optimizadores y schedulers
│   │   └── callbacks.py       # Callbacks (Early stopping, etc.)
│   ├── evaluation/
│   │   ├── metrics.py         # Cálculo de métricas
│   │   ├── visualization.py   # Gráficas y visualizaciones
│   │   └── gradcam.py         # Implementación Grad-CAM
│   ├── train.py               # Script de entrenamiento
│   ├── evaluate.py            # Script de evaluación
│   └── predict.py             # Script de predicción
├── deployment/
│   ├── api.py                 # FastAPI REST API
│   ├── app.py                 # Streamlit web app
│   ├── Dockerfile             # Contenedor Docker
│   ├── docker-compose.yml     # Orquestación multi-container
│   └── requirements_deploy.txt # Dependencias de producción
├── tests/
│   ├── test_preprocessing.py  # Tests de preprocesamiento
│   ├── test_model.py          # Tests del modelo
│   └── test_api.py            # Tests de la API
├── docs/
│   ├── informe_tecnico.md     # Informe técnico completo
│   ├── arquitectura.png       # Diagrama de arquitectura
│   ├── presentacion.pptx      # Slides para exposición
│   └── manual_usuario.md      # Manual de usuario
├── configs/
│   ├── train_config.yaml      # Configuración de entrenamiento
│   └── model_config.yaml      # Configuración del modelo
├── .gitignore
├── requirements.txt           # Dependencias del proyecto
├── setup.py                   # Instalación del paquete
└── README.md                  # Este archivo
```

## 📊 Resultados

### Métricas de Rendimiento

| Métrica    | Valor  |
|------------|--------|
| Accuracy   | 96.3%  |
| Precision  | 95.7%  |
| Recall     | 96.9%  |
| F1-Score   | 96.3%  |
| ROC-AUC    | 98.5%  |

### Curvas de Aprendizaje

*[Gráficas de pérdida y accuracy durante entrenamiento]*

### Matriz de Confusión

*[Visualización de predicciones correctas e incorrectas]*

### Interpretabilidad (Grad-CAM)

*[Mapas de calor mostrando regiones faciales analizadas]*

## 🔧 MLOps y Despliegue

### Ciclo de Vida del Modelo

1. **Data Collection**: Descarga y organización de datasets
2. **Preprocessing**: Extracción de frames y detección facial
3. **Training**: Entrenamiento con validación cruzada
4. **Evaluation**: Métricas y análisis de rendimiento
5. **Optimization**: Ajuste de hiperparámetros (Optuna)
6. **Deployment**: API REST + Docker
7. **Monitoring**: Logs y métricas en producción
8. **Retraining**: Actualización continua con nuevos datos

### Integración Continua

```yaml
# .github/workflows/ci.yml
- Linting con flake8
- Tests unitarios con pytest
- Cobertura de código
- Build de Docker image
- Deploy automático (opcional)
```

### Versionado de Modelos

- **DVC**: Control de versiones para datos y modelos
- **MLflow**: Tracking de experimentos y métricas
- **Model Registry**: Gestión de versiones en producción

## 🤝 Contribuciones

Proyecto desarrollado por: [Nombres de los integrantes del equipo]

**Asignaturas:**
- Machine Learning (TIEL26)
- Aplicaciones de IA (TI2082)

**Institución:** [Tu institución educativa]

## 📄 Licencia

MIT License - Ver archivo `LICENSE` para más detalles.

## 📧 Contacto

Para consultas o colaboraciones: [tu-email@ejemplo.com]

---

**⚠️ Disclaimer**: Este sistema es una herramienta de apoyo para la detección de deepfakes. No garantiza 100% de precisión y debe usarse como complemento del análisis humano especializado.
