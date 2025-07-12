# standard libraries
import math
from typing import Any

# 3pps
import torch
import torch.nn.functional as F

import torch.nn as nn




## ReLU
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
        ctx.mask = mask
        return mask * inputs
     

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
        mask = ctx.mask 
        grad_input = torch.ones_like(grad_output)
        return grad_input * mask

  

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
    

## SELU

class SELUFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the SELU.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, alpha: float, scale: float) -> torch.Tensor:
        """
        This is the forward method of the SELU.

        Args:
            ctx: Context for saving elements for the backward.
            inputs: Input tensor. Dimensions: [*].
            alpha: Alpha parameter. Dimensions: [*].
            scale: Scale parameter. Dimensions: [*].

        Returns:
            Outputs tensor. Dimensions: [*], same as inputs.
        """

        # TODO
        outputs = torch.empty_like(inputs)
        ctx.scale = scale
        ctx.alpha = alpha 
        ctx.inputs = inputs
        outputs[inputs >= 0] = scale * inputs[inputs >= 0]
        outputs[inputs < 0 ] = scale * (alpha*(torch.exp(inputs[inputs < 0 ])-1))
        return outputs
           

    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        This method is the backward of the SELU.
        Defines how to compute gradient in backpropagation

        Args:
            ctx: Context for loading elements from the forward.
            grad_output: Outputs gradients. Dimensions: [*].

        Returns:
            Inputs gradients. Dimensions: [*], same as the grad_output.
            None.
            None.
        """
        # TODO
        alpha = ctx.alpha 
        scale = ctx.scale 
        inputs = ctx.inputs 
        der_inputs = torch.empty_like(inputs)
        der_inputs[ inputs >= 0 ] = scale 
        der_inputs[ inputs < 0 ] = scale * alpha * torch.exp(inputs[inputs < 0])
        return grad_output* der_inputs, None, None
    


class SELU(torch.nn.Module):
    """
    Class for the implementation of the forward and backward pass of
    the SELU.
    """

    def __init__(self, alpha: float = 1.67326, scale: float = 1.0507):
        super().__init__()
        self.alpha = alpha
        self.scale = scale

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return SELUFunction.apply(inputs, self.alpha, self.scale)
    


import torch

class MaxoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        weights_first: torch.Tensor,
        bias_first: torch.Tensor,
        weights_second: torch.Tensor,
        bias_second: torch.Tensor,
    ) -> torch.Tensor:
        # Calcula ambas salidas
        outputs_first  = inputs @ weights_first.T  + bias_first    # [B, D_out]
        outputs_second = inputs @ weights_second.T + bias_second   # [B, D_out]

        # Máscara booleana: True donde second > first
        mask = outputs_second > outputs_first                     # [B, D_out]

        # Salida final: copia first, y en los True de mask mete second
        outputs = outputs_first.clone()
        outputs[mask] = outputs_second[mask]

        # Guardamos para el backward
        ctx.inputs        = inputs
        ctx.weights_first = weights_first
        ctx.weights_second= weights_second
        ctx.bias_first    = bias_first
        ctx.bias_second   = bias_second
        ctx.mask          = mask
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs         = ctx.inputs            # [B, D_in]
        W1             = ctx.weights_first    # [D_out, D_in]
        W2             = ctx.weights_second   # [D_out, D_in]
        b1             = ctx.bias_first       # [D_out]
        b2             = ctx.bias_second      # [D_out]
        mask           = ctx.mask             # [B, D_out]

        B, D_out = grad_output.shape
        _, D_in  = inputs.shape

        # Creamos dos copias de grad_output y las enmascaramos
        grad1 = grad_output.clone()   # para la rama first
        grad2 = grad_output.clone()   # para la rama second

        # En grad1 sólo dejamos los where mask==False; en grad2 donde mask==True
        grad1[mask] = 0
        grad2[~mask] = 0

        # 1) Gradiente w.r.t. inputs:
        #   grad_inputs = grad1 @ W1  +  grad2 @ W2
        grad_inputs = grad1 @ W1  +  grad2 @ W2  # [B, D_in]

        # 2) Gradiente w.r.t. weights:
        #   grad_w_first  = grad1.T  @ inputs   -> [D_out, D_in]
        #   grad_w_second = grad2.T  @ inputs
        grad_w_first  = grad1.T @ inputs       # [D_out, D_in]
        grad_w_second = grad2.T @ inputs       # [D_out, D_in]

        # 3) Gradiente w.r.t. biases:
        #   grad_b_first  = sum over batch de grad1
        #   grad_b_second = sum over batch de grad2
        grad_b_first  = grad1.sum(dim=0)       # [D_out]
        grad_b_second = grad2.sum(dim=0)       # [D_out]

        # Devolvemos en el mismo orden que en forward:
        # grad_inputs, grad_w_first, grad_b_first, grad_w_second, grad_b_second
        return grad_inputs, grad_w_first, grad_b_first, grad_w_second, grad_b_second



class Maxout(torch.nn.Module):
    """
    This is the class that represents the Maxout Layer.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        This method is the constructor of the Maxout layer.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weights_first: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_dim, input_dim)
        )
        self.bias_first: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_dim)
        )
        self.weights_second: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_dim, input_dim)
        )
        self.bias_second: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_dim)
        )

        # init parameters corectly
        self.reset_parameters()

        self.fn = MaxoutFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, input dim].

        Returns:
            outputs tensor. Dimensions: [batch, output dim].
        """

        return self.fn(
            inputs,
            self.weights_first,
            self.bias_first,
            self.weights_second,
            self.bias_second,
        )

    @torch.no_grad()
    def set_parameters(
        self,
        weights_first: torch.Tensor,
        bias_first: torch.Tensor,
        weights_second: torch.Tensor,
        bias_second: torch.Tensor,
    ) -> None:
        """
        This function is to set the parameters of the model.

        Args:
            weights_first: weights for the first branch.
            bias_first: bias for the first branch.
            weights_second: weights for the second branch.
            bias_second: bias for the second branch.
        """

        # set attributes
        self.weights_first = torch.nn.Parameter(weights_first)
        self.bias_first = torch.nn.Parameter(bias_first)
        self.weights_second = torch.nn.Parameter(weights_second)
        self.bias_second = torch.nn.Parameter(bias_second)

        return None

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        # init parameters the correct way
        torch.nn.init.kaiming_uniform_(self.weights_first, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weights_second, a=math.sqrt(5))
        if self.bias_first is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights_first)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias_first, -bound, bound)
            torch.nn.init.uniform_(self.bias_second, -bound, bound)

        return None



class SoftshrinkFuncion(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the Softshrink.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        lambd: float,
    ) -> torch.Tensor:
        """
        This is the forward method of the Softshrink.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch, *].

        Returns:
            outputs tensor. Dimensions: [batch, *].
        """

        # TODO
        ctx.inputs = inputs
        ctx.lambd = lambd
        outputs = torch.zeros_like(inputs)
        outputs[inputs > lambd] = inputs[inputs > lambd] - lambd
        outputs[inputs < -lambd] = inputs[inputs < -lambd] + lambd
        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        """
        This method is the backward of the Softshrink.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions: [batch, *].

        Returns:
            inputs gradients. Dimensions: [batch, *].
        """

        # TODO
        inputs = ctx.inputs 
        lambd = ctx.lambd
        grad_input = torch.zeros_like(inputs)
        grad_input[inputs > lambd] = 1
        grad_input[inputs < -lambd] = 1
        return grad_outputs * grad_input, None


class Softshrink(torch.nn.Module):
    """
    This is the class that represents the Softshrink Layer.
    """

    padding_idx: int

    def __init__(self, lambd: float = 0.5) -> None:
        """
        This method is the constructor of the Softshrink layer.
        """

        # call super class constructor
        super().__init__()

        # init parameters corectly
        self.lambd = lambd

        self.fn = SoftshrinkFuncion.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, *].

        Returns:
            outputs tensor. Dimensions: [batch, *].
        """

        return self.fn(inputs, self.lambd)
