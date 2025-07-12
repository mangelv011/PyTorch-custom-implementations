# deep learning libraries
import torch
import torch.nn.functional as F

# other libraries
from typing import Optional, Any


def unfold_max_pool_2d(
    inputs: torch.Tensor, kernel_size: int, stride: int, padding: int
) -> torch.Tensor:
    """
    This function computes the unfold needed for the MaxPool2d.
    Since the maxpool only computes the max over single channel
    and not over all the channels, we need that the second dimension of
    our unfold tensors are data from only channel. For that, we will
    include the channels into another dimension that will
    not be affected by the consequently operations.

    Args:
        inputs: inputs tensor. Dimensions: [batch, channels, height,
            width].
        kernel_size: size of the kernel to use. In this case the
            kernel will be symmetric, that is why only an integer is
            accepted.
        stride: stride to use in the maxpool operation. As in the case
            of the kernel size, the stride willm be symmetric.
        padding: padding to use in the maxpool operation. As in the
            case of the kernel.

    Returns:
        inputs unfolded. Dimensions: [batch * channels,
            kernel size * kernel size, number of windows].
    """

    # TODO
    batch, channels, _, _ = inputs.shape
    unfolded = F.unfold(input=inputs, kernel_size=kernel_size, stride=stride, padding=padding)
    _ , _ , number_of_blocks = unfolded.shape
    return unfolded.view(batch*channels, kernel_size*kernel_size,number_of_blocks)


def fold_max_pool_2d(
    inputs: torch.Tensor,
    output_size: int,
    batch_size: int,
    kernel_size,
    stride: int,
    padding: int,
) -> torch.Tensor:
    """
    This function computes the fold needed for the MaxPool2d.
    Since the maxpool only comute sthe max over single channel
    and not over all the channels, we need that the second dimension of
    our unfold tensors are data from only channel. To do that, we
    this fold version recovers the channel dimensions before executing 
    the fold operation.

    Args:
        inputs: inputs unfolded. Dimensions: [batch * channels,
            kernel size * kernel size, number of windows].
        output_size: output size for the fold, i.e., the height and
            the width.
        batch_size: batch size
        stride: stride to use in the maxpool operation. As in the case
            of the kernel size, the stride willm be symmetric.
        padding: padding to use in the maxpool operation. As in the
            case of the kernel.

    Returns:
        inputs folded. Dimensions: [batch, channels, height, width].
    """

    # TODO
    
    batch_x_channels, _ , number_of_blocks = inputs.shape
    channels = int(batch_x_channels / batch_size)
    inputs = inputs.view(batch_size, channels *kernel_size*kernel_size, number_of_blocks)
    return F.fold(input=inputs, output_size=output_size,kernel_size=kernel_size,stride=stride, padding=padding)


class MaxPool2dFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the MaxPool2d.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> torch.Tensor:
        """
        This is the forward method of the MaxPool2d.

        Args:
            ctx: context for saving elements for the backward.
            inputs: inputs for the model. Dimensions: [batch,
                channels, height, width].

        Returns:
            output of the layer. Dimensions:
                [batch, channels,
                (height + 2*padding - kernel size) / stride + 1,
                (width + 2*padding - kernel size) / stride + 1]
        """

        # TODO
        batch_size, num_channels, input_height, input_width = inputs.shape
        output_height = (input_height + 2*padding - kernel_size)//stride + 1
        output_width = output_height  # Assuming square input
        
        unfolded_inputs = unfold_max_pool_2d(inputs, kernel_size=kernel_size, stride=stride, padding=padding)
        
        # Compute max over unfolded inputs
        unfolded_outputs, max_indices = unfolded_inputs.max(dim=1)
        outputs = unfolded_outputs.view(batch_size, num_channels, output_height, output_width)
        
        ctx.save_for_backward(max_indices, unfolded_inputs, inputs)
        ctx.kernel_size = kernel_size
        ctx.padding = padding
        ctx.stride = stride

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        """
        This method is the backward of the MaxPool2d.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions:
                [batch, channels,
                (height + 2*padding - kernel size) / stride + 1,
                (width + 2*padding - kernel size) / stride + 1]

        Returns:
            inputs gradients dimensions: [batch, channels,
                height, width].
            None value.
            None value.
            None value.
        """

        # TODO
        # 1. Recuperamos los valores guardados en el forward
        max_indices, unfolded_inputs, inputs = ctx.saved_tensors
        kernel_size, padding, stride = ctx.kernel_size, ctx.padding, ctx.stride
        
        # 2. Obtenemos las dimensiones
        batch_size, num_channels = inputs.shape[:2]
        
        # 3. Aplanamos los gradientes de salida
        grad_outputs_flat = grad_outputs.reshape(batch_size*num_channels, -1)
        
        # 4. Creamos un tensor de ceros y colocamos los gradientes solo en las posiciones máximas
        grad_inputs_unfolded = torch.zeros_like(unfolded_inputs)
        grad_inputs_unfolded.scatter_(
            dim=1,                          # La dimensión sobre la que actuamos (k×k)
            index=max_indices.unsqueeze(1), # Índices de los valores máximos
            src=grad_outputs_flat.unsqueeze(1)  # Valores de gradiente
        )
        
        # 5. Volvemos al formato de entrada original mediante fold
        grad_inputs = fold_max_pool_2d(
            inputs=grad_inputs_unfolded,
            output_size=inputs.shape[2],
            batch_size=batch_size,
            kernel_size=kernel_size,
            padding=padding, 
            stride=stride
        )
        
        # Solo el primer argumento necesita gradiente
        return grad_inputs, None, None, None


class MaxPool2d(torch.nn.Module):
    """
    This is the class that represents the MaxPool2d Layer.
    """

    kernel_size: int
    stride: int

    def __init__(
        self, kernel_size: int, stride: Optional[int], padding: int = 0
    ) -> None:
        """
        This method is the constructor of the MaxPool2d layer.
        """

        # call super class constructor
        super().__init__()

        # set attributes value
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding

        # save function
        self.fn = MaxPool2dFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, channels,
                output channels, height, width].

        Returns:
            outputs tensor. Dimensions: [batch, channels,
                height - kernel size + 1, width - kernel size + 1].
        """

        return self.fn(inputs, self.kernel_size, self.stride, self.padding)
