import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
from typing import Union


# Fonction pour charger le modèle
@st.cache_resource
def load_model(model_path: str, num_classes: int = 4) -> nn.Module:
    """
    Charge un modèle ResNet18 pré-entraîné à partir du fichier spécifié et 
    l'adapte pour le nombre de classes spécifié.
    
    Args:
        model_path (str): Chemin vers le fichier du modèle enregistré.
        num_classes (int): Nombre de classes de sortie du modèle.
        
    Returns:
        nn.Module: Le modèle PyTorch chargé et prêt à l'inférence.
    """
    model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Mettre le modèle en mode évaluation
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Transforme et redimensionne une image PIL en un tenseur 
    compatible avec PyTorch.
    S'assure que l'image a 3 canaux (RGB).
    
    Args:
        image (Image.Image): L'image à transformer.
        
    Returns:
        torch.Tensor: Tenseur représentant l'image, avec une dimension 
        supplémentaire pour le batch.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Ajouter une dimension batch
    return image_tensor


# Liste des noms de classes
class_names = [
    "adenocarcinoma_left.lower.lobe",
    "large.cell.carcinoma_left.hilum",
    "normal",
    "squamous.cell.carcinoma_left.hilum"
]


def predict(model: nn.Module, image_tensor: torch.Tensor) -> str:
    """
    Prend une image transformée sous forme de tenseur et prédit sa classe à l'aide du modèle.
    Renvoie le nom de la classe prédite.

    Args:
        model (nn.Module): Le modèle de classification PyTorch.
        image_tensor (torch.Tensor): L'image transformée sous forme de tenseur.
        
    Returns:
        str: Le nom de la classe prédite.
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)  # Obtenir la classe
        class_idx = predicted.item()  # Obtenir l'indice de la classe
        return class_names[class_idx]  # Retourne le nom de la classe


# Interface Streamlit
st.title("Classification d'images CT-Scan")

# Chargement du modèle
model_path = "models/ct-scan-classifier.pth"
model = load_model(model_path)

# Option de téléchargement d'image
uploaded_file = st.file_uploader("Choisissez une image à classer",
                                 type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Afficher l'image téléchargée
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée', use_column_width=True)

    # Prétraitement de l'image
    image_tensor = preprocess_image(image)

    # Classification
    if st.button('Classer'):
        label = predict(model, image_tensor)

        # Affichage du résultat
        st.write(f"La classe prédite est : {label}")
