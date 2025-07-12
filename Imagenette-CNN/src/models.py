# standard libraries
import math
from typing import Any

# 3pps
import torch
import torch.nn.functional as F

import torch.nn as nn


class ReLUFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the ReLU.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward method of the relu.

        Args:
            ctx: Context for saving elements for the backward.
            inputs: Input tensor. Dimensions: [*].

        Returns:
            Outputs tensor. Dimensions: [*], same as inputs.
        """

        # TODO
        mask = (inputs > 0)
        ctx.save_for_backward(mask.detach())
        outputs = inputs * mask
        return outputs

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        This method is the backward of the relu.

        Args:
            ctx: Context for loading elements from the forward.
            grad_output: Outputs gradients. Dimensions: [*].

        Returns:
            Inputs gradients. Dimensions: [*], same as the grad_output.
        """

        # TODO
        mask, = ctx.saved_tensors # el valor que devuelve ctx.saved_tensors es una tupla
        grad_input = mask * grad_output
        return grad_input


class ReLU(torch.nn.Module):
    """
    This is the class that represents the ReLU Layer.
    """

    def __init__(self):
        """
        This method is the constructor of the ReLU layer.
        """

        # call super class constructor
        super().__init__()

        self.fn = ReLUFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: Inputs tensor. Dimensions: [*].

        Returns:
            Outputs tensor. Dimensions: [*] (same as the input).
        """

        return self.fn(inputs)


class LinearFunction(torch.autograd.Function):
    """
    This class implements the forward and backward of the Linear layer.
    """

    @staticmethod
    def forward(
        ctx: Any, inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """
        This method is the forward pass of the Linear layer.

        Args:
            ctx: Contex for saving elements for the backward.
            inputs: Inputs tensor. Dimensions:
                [batch, input dimension].
            weight: weights tensor.
                Dimensions: [output dimension, input dimension].
            bias: Bias tensor. Dimensions: [output dimension].

        Returns:
            Outputs tensor. Dimensions: [batch, output dimension].
        """

        # TODO
        ctx.save_for_backward(inputs, weight, bias)
        return inputs @ weight.T + bias 


    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This method is the backward for the Linear layer.

        Args:
            ctx: Context for loading elements from the forward.
            grad_output: Outputs gradients.
                Dimensions: [batch, output dimension].

        Returns:
            Inputs gradients. Dimensions: [batch, input dimension].
            Weights gradients. Dimensions: [output dimension,
                input dimension].
            Bias gradients. Dimension: [output dimension].
        """

        # TODO
        inputs, weight, bias = ctx.saved_tensors
        # derivada respecto a inputs, derivada respecto a weights, derivada respecto a bias
        # como se traspone w en el forward, hay que trasponer el grad output en su derivada
        # needs_input_grad calcula el gradiente solo si es necesario 
        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.T @ inputs
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias


class Linear(torch.nn.Module):
    """
    This is the class that represents the Linear Layer.

    Attributes:
        weight: Weight torch parameter. Dimensions: [output dimension,
            input dimension].
        bias: Bias torch parameter. Dimensions: [output dimension].
        fn: Autograd function.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        This method is the constructor of the Linear layer.
        The attributes must be named the same as the parameters of the
        linear layer in pytorch. The parameters should be initialized

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_dim, input_dim)
        )
        self.bias: torch.nn.Parameter = torch.nn.Parameter(torch.empty(output_dim))

        # init parameters corectly
        self.reset_parameters()

        # define layer function
        self.fn = LinearFunction.apply

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method if the forward pass of the layer.

        Args:
            inputs: Inputs tensor. Dimensions: [batch, input dim].

        Returns:
            Outputs tensor. Dimensions: [batch, output dim].
        """

        return self.fn(inputs, self.weight, self.bias)

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        # init parameters the correct way
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

        return None


class Conv2dFunction(torch.autograd.Function):
    """
    Class to implement the forward and backward methods of the Conv2d
    layer.
    """

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        padding: int,
        stride: int,
    ) -> torch.Tensor:
        """
        This function is the forward method of the class.

        Args:
            ctx: Context for saving elements for the backward.
            inputs: Inputs for the model. Dimensions: [batch,
                input channels, height, width].
            weight: Weight of the layer.
                Dimensions: [output channels, input channels,
                kernel size, kernel size].
            bias: Bias of the layer. Dimensions: [output channels].
            padding: padding parameter.
            stride: stride parameter.

        Returns:
            Output of the layer. Dimensions:
                [batch, output channels,
                (height + 2*padding - kernel size) / stride + 1,
                (width + 2*padding - kernel size) / stride + 1]
        """

        # TODO
        # guardamos lo que necesitamos para el backward
        ctx.save_for_backward(inputs, weight, bias)
        ctx.padding = padding
        ctx.stride = stride

        # dimensiones inputs: (batch, canales de entrada, altura, ancho)
        batch_size, in_channels, in_height, in_width = inputs.shape
        # dimensiones weights: (canales de salida, canales de entrada, tamaño del kernel, tamaño del kernel)
        out_channels, _, kernel_size, _ = weight.shape

        # unfold de los inputs, dimensiones: (batch, canales entrada*kernel*kernel, h'*w')
        unfolded = F.unfold(inputs, 
                           kernel_size=(kernel_size, kernel_size),
                           padding=padding,
                           stride=stride)
        
        # view de weights para la multiplicación matricial, dimensiones: (canales out, canales in * kernel*kernel)
        weight_reshaped = weight.view(out_channels, -1)

        # (c out, c in*k*k) * (batch, c in*k*k,h'*w') = (batch, c out, h'*w')
        out = weight_reshaped @ unfolded

        if bias is not None:
            out = out + bias.view(1, -1, 1)

        out_height = (in_height + 2 * padding - kernel_size) // stride + 1
        out_width = (in_width + 2 * padding - kernel_size) // stride + 1

        # reshape final
        output = out.view(batch_size, out_channels, out_height, out_width)

        return output



    @staticmethod
    def backward(  # type: ignore
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """
        This is the backward of the layer.

        Args:
            ctx: Context for loading elements needed in the backward.
            grad_output: Outputs gradients. Dimensions:
                [batch, output channels,
                (height + 2*padding - kernel size) / stride + 1,
                (width + 2*padding - kernel size) / stride + 1]

        Returns:
            Inputs gradients. Dimensions: [batch, input channels,
                height, width].
            Weight gradients. Dimensions: [output channels,
                input channels, kernel size, kernel size].
            Bias gradients. Dimensions: [output channels].
            None.
            None.
        """

        # TODO
        # Recuperamos del contexto los tensores guardados durante la pasada forward
        inputs, weight, bias = ctx.saved_tensors
        padding = ctx.padding
        stride = ctx.stride

        # Obtenemos las dimensiones de los inputs y de los pesos
        batch_size, in_channels, in_height, in_width = inputs.shape
        out_channels, _, kernel_size, _ = weight.shape

        # Calculamos el gradiente del bias sumando grad_output a lo largo de las dimensiones batch, height y width
        grad_bias = grad_output.sum((0, 2, 3))

        # Aplicamos unfold a los inputs para extraer todos los parches (ventanas) que se usaron en la convolución forward
        # Esto genera un tensor de dimensiones: [batch, in_channels * kernel_size * kernel_size, L] donde L es el número de parches
        unfolded_input = F.unfold(inputs,
                                kernel_size=(kernel_size, kernel_size),
                                padding=padding,
                                stride=stride)
        
        # Remodelamos grad_output para que tenga la forma [batch, out_channels, L]
        grad_output_reshaped = grad_output.view(batch_size, out_channels, -1)
        
        # Calculamos el gradiente de los pesos mediante una multiplicación matricial batch-wise:
        # Cada muestra contribuye a un gradiente parcial que luego se suman sobre el batch.
        # torch.bmm multiplica matrices por lotes: (batch, out_channels, L) @ (batch, L, in_channels * kernel_size * kernel_size)
        grad_weight = torch.bmm(grad_output_reshaped, 
                                unfolded_input.transpose(1, 2))
        
        # Sumamos los gradientes parciales de cada muestra del batch y remodelamos al tamaño original de weight
        grad_weight = grad_weight.sum(0)
        grad_weight = grad_weight.view(out_channels, in_channels, kernel_size, kernel_size)

        # Remodelamos los pesos para facilitar el cálculo del gradiente respecto a los inputs
        # Cambiamos la forma a [out_channels, in_channels * kernel_size * kernel_size]
        weight_reshaped = weight.view(out_channels, -1)
        
        # Calculamos el gradiente sobre el input unfolded realizando la multiplicación de:
        # (weight_reshaped transpuesto) de forma: [in_channels * kernel_size * kernel_size, out_channels]
        # por grad_output_reshaped: [batch, out_channels, L] dando como resultado: [batch, in_channels * kernel_size * kernel_size, L]
        grad_unfolded = weight_reshaped.t() @ grad_output_reshaped

        # Reconstruimos el gradiente del input original utilizando fold, que es la operación inversa a unfold.
        # Esto nos devuelve un tensor de gradiente con dimensiones: [batch, in_channels, in_height, in_width]
        grad_input = F.fold(grad_unfolded,
                            output_size=(in_height, in_width),
                            kernel_size=(kernel_size, kernel_size),
                            padding=padding,
                            stride=stride)
        
        # Se retorna el gradiente respecto al input, al weight, al bias y None para los parámetros padding y stride
        return grad_input, grad_weight, grad_bias, None, None



class Conv2d(torch.nn.Module):
    """
    This is the class that represents the Linear Layer.

    Attributes:
        weight: Weight pytorch parameter. Dimensions: [output channels,
            input channels, kernel size, kernel size].
        bias: Bias torch parameter. Dimensions: [output channels].
        padding: Padding parameter.
        stride: Stride parameter.
        fn: Autograd function.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
    ) -> None:
        """
        This method is the constructor of the Linear layer. Follow the
        pytorch convention.

        Args:
            input_channels: Input dimension.
            output_channels: Output dimension.
            kernel_size: Kernel size to use in the convolution.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_channels, input_channels, kernel_size, kernel_size)
        )
        self.bias: torch.nn.Parameter = torch.nn.Parameter(torch.empty(output_channels))
        self.padding = padding
        self.stride = stride

        # init parameters corectly
        self.reset_parameters()

        # define layer function
        self.fn = Conv2dFunction.apply

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method if the forward pass of the layer.

        Args:
            inputs: Inputs tensor. Dimensions: [batch, input channels,
                output channels, height, width].

        Returns:
            outputs tensor. Dimensions: [batch, output channels,
                height - kernel size + 1, width - kernel size + 1].
        """

        return self.fn(inputs, self.weight, self.bias, self.padding, self.stride)

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        # init parameters the correct way
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

        return None


class Block(torch.nn.Module):
    """
    Neural net block composed of 3x(conv(kernel=3, padding=1) + ReLU).

    Attributes:
        net: Sequential containing all the layers.
    """

    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        """
        Constructor of the Block class. It is composed of
        3x(conv(kernel=3) + ReLU). Only the second conv
        will have stride. Use a Sequential for encapsulating all the
        layers. Clue: convs may have padding to fit into the correct
        dimensions.

        Args:
            input_channels: Input channels for Block.
            output_channels: Output channels for Block.
            stride: Stride only for the second convolution of the
                Block.
        """

        # TODO
        super().__init__()

        self.net = nn.Sequential(
            # First conv block: input_channels -> output_channels, stride=1
            nn.Conv2d( # no hace falta especificar el tamaño de imagen
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1  # Padding=1 to maintain spatial dimensions
            ),
            nn.ReLU(),

            # Second conv block: output_channels -> output_channels, custom stride
            nn.Conv2d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=stride,
                padding=1  # Padding=1 to only reduce dimensions by stride
            ),
            nn.ReLU(),

            # Third conv block: output_channels -> output_channels, stride=1
            nn.Conv2d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1  # Padding=1 to maintain spatial dimensions
            ),
            nn.ReLU()
        )
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method if the forward pass.

        Args:
            inputs: Inputs batch of tensors.
                Dimensions: [batch, input_channels, height, width].

        Returns:
            Outputs batch of tensors. Dimensions: [batch, output_channels,
                (height - 1)/stride + 1, (width - 1)/stride + 1].
        """

        # TODO
        return self.net(inputs)


class CNNModel(torch.nn.Module):
    """
    Model constructed used Block modules.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...],
        input_channels: int = 3,
        output_channels: int = 10,
        dropout_prob: float = 0.5
    ) -> None:
        """
        Constructor of the class CNNModel.

        Args:
            layers: Output channel dimensions of the Blocks.
            input_channels: Input channels of the model.
        """

        # TODO

        super(CNNModel, self).__init__()
        blocks = []
        current_channels = input_channels
        # Construcción de los bloques: se aplica stride=2 a partir del segundo bloque
        for i, hidden_size in enumerate(hidden_sizes):
            stride = 2 if i % 2 == 0 else 1  # Downsample every other block
            blocks.append(Block(current_channels, hidden_size, stride))
            current_channels = hidden_size

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_sizes[-1], output_channels)

   

   
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method returns a batch of logits. It is the output of the
        neural network.

        Args:
            inputs: Inputs batch of images.
                Dimensions: [batch, channels, height, width].

        Returns:
            Outputs batch of logits. Dimensions: [batch,
                output_channels].
        """

        # TODO

        # Extracción de características
        x = self.features(inputs)
        # Pooling global adaptativo para obtener salida 1x1
        x = self.pool(x) # el tensor que sale tiene dimension (batch, channels, 1,1)
        # Aplanar la salida manteniendo la dimensión de batch
        x = torch.flatten(x, 1) # flattten para alimentar la capa lineal con (batch, channels)
        # Aplicar Dropout
        x = self.dropout(x)
        # Clasificación final
        logits = self.classifier(x)
        return logits

    
