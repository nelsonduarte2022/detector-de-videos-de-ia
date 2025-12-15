import torch.nn as nn

# Ejemplo simple: EfficientNet-B4 con torchvision (puedes ajustar según tu proyecto)
def get_model(model_name='efficientnet_b4', pretrained=True):
    from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
    if model_name == 'efficientnet_b4':
        if pretrained:
            weights = EfficientNet_B4_Weights.DEFAULT
        else:
            weights = None
        model = efficientnet_b4(weights=weights)
        # Ajustar la última capa para 2 clases (deepfake vs real)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 2)
        return model
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")
