# Instrucciones de Despliegue Docker

## 🐳 Opción 1: Docker Compose (Recomendado)

### Construcción y ejecución
```bash
# Construir y levantar el contenedor
docker-compose up --build -d

# Ver logs
docker-compose logs -f

# Detener
docker-compose down
```

La aplicación estará disponible en: **http://localhost:5000**

---

## 🐋 Opción 2: Docker Manual

### 1. Construir la imagen
```bash
docker build -t deepfake-shield:latest .
```

### 2. Ejecutar el contenedor
```bash
docker run -d \
  --name deepfake-shield \
  -p 5000:5000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/uploads:/app/uploads \
  deepfake-shield:latest
```

### 3. Ver logs
```bash
docker logs -f deepfake-shield
```

### 4. Detener y eliminar
```bash
docker stop deepfake-shield
docker rm deepfake-shield
```

---

## ☁️ Despliegue en la Nube

### Google Cloud Run
```bash
# 1. Autenticarse
gcloud auth login

# 2. Configurar proyecto
gcloud config set project YOUR_PROJECT_ID

# 3. Build y push a Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/deepfake-shield

# 4. Deploy
gcloud run deploy deepfake-shield \
  --image gcr.io/YOUR_PROJECT_ID/deepfake-shield \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --timeout 300 \
  --allow-unauthenticated
```

### AWS (Elastic Container Service)
```bash
# 1. Crear repositorio ECR
aws ecr create-repository --repository-name deepfake-shield

# 2. Autenticar Docker
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# 3. Tag y push
docker tag deepfake-shield:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/deepfake-shield:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/deepfake-shield:latest

# 4. Crear tarea y servicio en ECS (usar consola web o CLI)
```

---

## 🔧 Configuración de Producción

### Variables de entorno recomendadas
Crear archivo `.env`:
```env
FLASK_ENV=production
MAX_CONTENT_LENGTH=104857600
MODEL_PATH=checkpoints/demo_model.pth
UPLOAD_FOLDER=uploads
```

### Uso en docker-compose.yml
```yaml
services:
  deepfake-shield:
    env_file:
      - .env
```

---

## 📊 Monitoreo

### Verificar estado
```bash
# Healthcheck manual
curl http://localhost:5000/

# Métricas del contenedor
docker stats deepfake-shield
```

### Logs
```bash
# Últimas 100 líneas
docker logs --tail 100 deepfake-shield

# En tiempo real
docker logs -f deepfake-shield
```
