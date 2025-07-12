# deep learning libraries
import torch
import torch.nn as nn
# own modules
from src.utils import get_dropout_random_indexes
import torchvision.models as models



class Dropout(torch.nn.Module):
    """
    This the Dropout class.

    Attr:
        p: probability of the dropout.
        inplace: indicates if the operation is done in-place.
            Defaults to False.
    """

    def __init__(self, p: float, inplace: bool = False) -> None:
        """
        This function is the constructor of the Dropout class.

        Args:
            p: probability of the dropout.
            inplace: if the operation is done in place.
                Defaults to False.
        """
        # TODO
        # Inicializa la clase padre (torch.nn.Module)
        super(Dropout, self).__init__()
        
        # Almacena la probabilidad de dropout
        self.p = p
        
        # Indica si la operación se realiza in-place
        self.inplace = inplace

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forwward pass.

        Args:
            inputs: inputs tensor. Dimensions: [*].

        Returns:
            outputs. Dimensions: [*], same as inputs tensor.
        """
        # TODO
        # En modo de evaluación, retorna directamente los inputs
        if not self.training or self.p <= 0:
            return inputs
        
        # Obtenemos los índices para el dropout
        drop_indexes = get_dropout_random_indexes(inputs.shape, self.p)
        
        # Factor de escala para compensar los elementos eliminados
        scale = 1.0 / (1.0 - self.p)
        
        # Aplicamos el dropout
        if self.inplace:
            # Para inplace, primero aplicamos el escalado a todo el tensor
            inputs.mul_(scale)
            # Luego, ponemos a cero los elementos que deben eliminarse
            inputs[drop_indexes.bool()] = 0
            return inputs
        else:
            # Para non-inplace, creamos una nueva copia
            mask = torch.ones_like(inputs)
            mask[drop_indexes.bool()] = 0
            return inputs * mask * scale


class CNNModel(nn.Module):
    """
    Modelo de transfer learning para Imagenette usando ResNet18 preentrenado.
    
    Atributos:
        base_model: Modelo base preentrenado (ResNet18).
    """
    def __init__(self, num_classes: int = 10, freeze_base: bool = True, unfreeze_layers: int = 0) -> None:
        """
        Constructor del modelo de transfer learning.
        
        Args:
            num_classes: Número de clases de salida (por defecto 10 para Imagenette).
            freeze_base: Si True, congela los pesos del modelo base excepto las capas especificadas.
            unfreeze_layers: Número de capas finales a descongelar (0 = solo FC, 1 = layer4, 2 = layer3+layer4, etc.)
        """
        super().__init__()
        # Cargar el modelo preentrenado ResNet18
        # Usar weights='IMAGENET1K_V1' en lugar de pretrained=True para evitar la advertencia de deprecación
        self.base_model = models.resnet18(weights='IMAGENET1K_V1')
        
        if freeze_base:
            # Congelar todos los parámetros del modelo base por defecto
            for param in self.base_model.parameters():
                param.requires_grad = False
                
            # Descongelar las capas finales según el parámetro
            if unfreeze_layers >= 1:
                for param in self.base_model.layer4.parameters():
                    param.requires_grad = True
                    
            if unfreeze_layers >= 2:
                for param in self.base_model.layer3.parameters():
                    param.requires_grad = True
                    
            if unfreeze_layers >= 3:
                for param in self.base_model.layer2.parameters():
                    param.requires_grad = True
                    
            if unfreeze_layers >= 4:
                for param in self.base_model.layer1.parameters():
                    param.requires_grad = True
        
        # Reemplazar la capa final para adaptarla al número de clases de Imagenette
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            # nn.Dropout(0.5),  # Añadir dropout antes de la capa final para regularización
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo.
        
        Args:
            x: Tensor de entrada con dimensiones [batch, canales, altura, anchura].
            
        Returns:
            Logits con dimensiones [batch, num_classes].
        """
        return self.base_model(x)
