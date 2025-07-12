# deep learning libraries
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ReLU(torch.nn.Module):
    """
    This is the class that represents the ReLU Layer.
    """

    def __init__(self):
        """
        This method is the constructor of the ReLU layer.
        """

        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [*].

        Returns:
            outputs tensor. Dimensions: [*] (same as the input).
        """
        outputs = inputs.clone()
        outputs[outputs <= 0] = 0
        return outputs


class Linear(torch.nn.Module):
    """
    This is the class that represents the Linear Layer.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        This method is the constructor of the Linear layer. Follow the pytorch convention.

        Args:
            input_dim: input dimension.
            output_dim: output dimension.
        """
        super(Linear, self).__init__()

        # Initialize weights using Kaiming Uniform initialization
        self.weights = torch.nn.Parameter(torch.empty(input_dim, output_dim))
        torch.nn.init.kaiming_uniform_(self.weights, mode='fan_in', nonlinearity='relu')

        # Bias initialized to zeros
        self.bias = torch.nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method performs the forward pass of the layer.

        Args:
            inputs: inputs tensor. Dimensions: [batch, input dim].

        Returns:
            outputs tensor. Dimensions: [batch, output dim].
        """

        # Perform the matrix multiplication and add the bias
        outputs = inputs @ self.weights + self.bias
        return outputs


class MyModel(torch.nn.Module):
    """
    This is the class to construct the model. Only layers defined in
    this script can be used.
    """

    def __init__(
        self, input_size: int, output_size: int, hidden_sizes: tuple[int, ...]
    ) -> None:
        """
        This method is the constructor of the model.

        Args:
            input_size: size of the input
            output_size: size of the output
            hidden_sizes: three hidden sizes of the model
        """

        super(MyModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        # Bucle para crear las capas lineales, una detras de otra
        self.layers = torch.nn.ModuleList()
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(Linear(prev_size, hidden_size))  # Capa lineal
            prev_size = hidden_size  # Actualizar el tamaño de la capa previa

        # capa de salida
        self.output_layer = Linear(prev_size, output_size)

        self.relu = ReLU()
        # nos aseguramos de que todos los parámetros estén en el device

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            inputs: input tensor, Dimensions: [batch, channels, height,
                width].

        Returns:
            outputs of the model. Dimensions: [batch, 1].
        """

        # Pasar la entrada a través de las capas lineales y ReLU
        x = inputs.view(
            inputs.size(0), -1
        )  # Reshape para que sea plano y evitar errores de dimensionalidad.
        for layer in self.layers:
            x = layer.forward(x)
            x = self.relu.forward(
                x
            )  # Aplicar ReLU después de cada capa lineal (función de activación)

        # Pasar a través de la capa de salida
        x = self.output_layer.forward(x)
        return x