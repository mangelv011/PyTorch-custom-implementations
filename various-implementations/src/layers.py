# standard libraries
import math
from typing import Any

# 3pps
import torch
import torch.nn.functional as F

import torch.nn as nn

from src.utils import get_dropout_random_indexes

## Linear
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
        ctx.inputs = inputs
        ctx.weight = weight
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
        inputs = ctx.inputs
        weight = ctx.weight 
        return grad_output @ weight, grad_output.T @ inputs, grad_output.sum(dim=0)
       


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
    


## Dropout
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
        super().__init__()
        self.p = p
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
        if not self.training or self.p <= 0:
            return inputs 
        shape = inputs.shape
        indexes = get_dropout_random_indexes(shape, self.p)
        scale_factor = 1 / (1-self.p)
        
        if self.inplace:
            inputs[indexes.bool()] = 0 
            inputs*=scale_factor
            return inputs
        
        outputs = inputs.clone()
        outputs[indexes.bool()] = 0 
        return outputs * scale_factor



## RNN 
class RNNFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the RNN.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        inputs: torch.Tensor,
        h0: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This is the forward method of the RNN.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch, sequence,
                input size].
            h0: first hidden state. Dimensions: [1, batch,
                hidden size].
            weight_ih: weight for the inputs.
                Dimensions: [hidden size, input size].
            weight_hh: weight for the inputs.
                Dimensions: [hidden size, hidden size].
            bias_ih: bias for the inputs.
                Dimensions: [hidden size].
            bias_hh: bias for the inputs.
                Dimensions: [hidden size].


        Returns:
            outputs tensor. Dimensions: [batch, sequence,
                hidden size].
            final hidden state for each element in the batch.
                Dimensions: [1, batch, hidden size].
        """

        batch, seq, in_size = inputs.shape
        hidd_size = bias_hh.shape[0]

        outputs = torch.empty(batch, seq, hidd_size, dtype=inputs.dtype)

        h_t = h0.view(batch, hidd_size) # [batch, hidden size]
        
        # Store all hidden states for backward pass
        hidden_states = [h_t]  # Start with h0

        for t in range(seq):
            x_t = inputs[:,t,:] # [batch, in_size]
            h_t = torch.relu(x_t @ weight_ih.T + bias_ih + h_t @ weight_hh.T + bias_hh) # [batch, hidd_size]
            outputs[:,t,:] = h_t # [batch, hidd_size]
            hidden_states.append(h_t.clone())  # Store each hidden state

        hfinal = h_t.view(1, batch, hidd_size) # [1, batch, hidden size]
        
        # Save tensors needed for backward
        ctx.save_for_backward(inputs, h0, weight_ih, weight_hh, bias_ih, bias_hh)
        # Store hidden states for backward computation
        ctx.hidden_states = hidden_states
        
        return outputs, hfinal
        


    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor, grad_hn: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        This method is the backward of the RNN.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions: [*].

        Returns:
            inputs gradients. Dimensions: [batch, sequence,
                input size].
            h0 gradients state. Dimensions: [1, batch,
                hidden size].
            weight_ih gradient. Dimensions: [hidden size,
                input size].
            weight_hh gradients. Dimensions: [hidden size,
                hidden size].
            bias_ih gradients. Dimensions: [hidden size].
            bias_hh gradients. Dimensions: [hidden size].
        """
        # Recuperar los datos guardados del contexto
        inputs, h0, weight_ih, weight_hh, bias_ih, bias_hh = ctx.saved_tensors
        hidden_states = ctx.hidden_states

        # Obtener dimensiones
        batch, seq_len, in_size = inputs.shape
        hidden_size = bias_hh.shape[0]

        # Inicializar gradientes
        grad_inputs = torch.zeros_like(inputs)
        grad_weight_ih = torch.zeros_like(weight_ih)
        grad_weight_hh = torch.zeros_like(weight_hh)
        grad_bias_ih = torch.zeros_like(bias_ih)
        grad_bias_hh = torch.zeros_like(bias_hh)

        # Inicializar gradiente del último estado oculto
        grad_h = grad_hn.squeeze(0)

        # Backpropagation a través del tiempo (BPTT)
        for t in reversed(range(seq_len)):
            # Sumar gradiente del output en tiempo t
            grad_h = grad_h + grad_output[:, t, :]
            
            # Estado oculto previo y entrada actual
            h_prev = hidden_states[t]
            x_t = inputs[:, t, :]
            
            # Calcular la entrada a la función ReLU
            z = x_t @ weight_ih.T + bias_ih + h_prev @ weight_hh.T + bias_hh
            relu_grad = (z > 0)
            
            # Gradiente después de ReLU
            grad_after_relu = grad_h * relu_grad
            
            # Actualizar gradientes
            grad_inputs[:, t, :] = grad_after_relu @ weight_ih
            grad_weight_ih += grad_after_relu.T @ x_t
            grad_weight_hh += grad_after_relu.T @ h_prev
            grad_bias_ih += grad_after_relu.sum(dim=0)
            grad_bias_hh += grad_after_relu.sum(dim=0)
            
            # Gradiente para el estado oculto anterior
            grad_h = grad_after_relu @ weight_hh

        # Formato final para h0
        grad_h0 = grad_h.unsqueeze(0)

        return (
            grad_inputs,
            grad_h0,
            grad_weight_ih,
            grad_weight_hh,
            grad_bias_ih,
            grad_bias_hh,
        )



class RNN(torch.nn.Module):
    """
    This is the class that represents the RNN Layer.
    """

    def __init__(self, input_dim: int, hidden_size: int):
        """
        This method is the constructor of the RNN layer.
        """
        # call super class constructor
        super().__init__()

        # define attributes
        self.hidden_size = hidden_size
        self.weight_ih: torch.Tensor = torch.nn.Parameter(
            torch.empty(hidden_size, input_dim)
        )
        self.weight_hh: torch.Tensor = torch.nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.bias_ih: torch.Tensor = torch.nn.Parameter(torch.empty(hidden_size))
        self.bias_hh: torch.Tensor = torch.nn.Parameter(torch.empty(hidden_size))

        # init parameters corectly
        self.reset_parameters()

        self.fn = RNNFunction.apply

    def forward(self, inputs: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, sequence,
                input size].
            h0: initial hidden state.

        Returns:
            outputs tensor. Dimensions: [batch, sequence,
                hidden size].
            final hidden state for each element in the batch.
                Dimensions: [1, batch, hidden size].
        """
        return self.fn(
            inputs, h0, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh
        )

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

        return None
    


## LSTM
class LSTMFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of the LSTM.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        inputs: torch.Tensor,
        h0: torch.Tensor,
        c0: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the LSTM.

        Args:
            ctx: context for saving elements for backward.
            inputs: input tensor. [batch, sequence, input size]
            h0: initial hidden state. [1, batch, hidden size]
            c0: initial cell state. [1, batch, hidden size]
            weight_ih: weights for input. [4 * hidden size, input size]
            weight_hh: weights for hidden state. [4 * hidden size, hidden size]
            bias_ih: bias for input. [4 * hidden size]
            bias_hh: bias for hidden state. [4 * hidden size]

        Returns:
            outputs: [batch, sequence, hidden size]
            hn: final hidden state. [1, batch, hidden size]
            cn: final cell state. [1, batch, hidden size]
        """

        # TODO
        batch, seq, input_size = inputs.shape
        _,_, hidden_size = h0.shape

        wii, wif, wig, wio = weight_ih.view(4, hidden_size, input_size) # [hidden_size, input_size]
        whi, whf, whg, who = weight_hh.view(4, hidden_size, hidden_size)
        bii, bif, big, bio = bias_ih.view(4, hidden_size)
        bhi, bhf, bhg, bho = bias_hh.view(4, hidden_size)

        ht = h0.unsqueeze(0) # [batch, hidden_size]
        ct = c0.unsqueeze(0)
        outputs = torch.zeros(batch, seq, hidden_size, dtype=torch.double)

        for t in range(seq):
            x_t = inputs[:,t,:] # [batch, input_size]

            i_t = torch.sigmoid(x_t @ wii.T + bii + ht @ whi.T + bhi) # [batch, hidden_size]
            f_t = torch.sigmoid(x_t @ wif.T + bif + ht @ whf.T + bhf)
            g_t = torch.tanh(x_t @ wig.T + big + ht @ whg.T + bhg)
            o_t = torch.sigmoid(x_t @ wio.T + bio + ht @ who.T + bho)
            ct = f_t * ct + i_t * g_t
            ht = o_t * torch.tanh(ct)

            outputs[:,t,:] = ht

        hn = ht.squeeze(0)
        cn = ct.squeeze(0)

        return outputs, hn, cn
      


class LSTM(torch.nn.Module):
    """
    Custom LSTM layer.
    """

    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.weight_ih = torch.nn.Parameter(torch.empty(4 * hidden_size, input_dim))
        self.weight_hh = torch.nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih = torch.nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.empty(4 * hidden_size))

        self.reset_parameters()
        self.fn = LSTMFunction.apply

    def forward(self, inputs: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.fn(inputs, h0, c0, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)



## GRU
class GRUFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of the GRU.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        inputs: torch.Tensor,
        h0: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GRU.

        Args:
            ctx: context for saving elements for backward.
            inputs: input tensor. [batch, sequence, input size]
            h0: initial hidden state. [1, batch, hidden size]
            weight_ih: weights for input. [3 * hidden size, input size]
            weight_hh: weights for hidden state. [3 * hidden size, hidden size]
            bias_ih: bias for input. [3 * hidden size]
            bias_hh: bias for hidden state. [3 * hidden size]

        Returns:
            outputs: [batch, sequence, hidden size]
            hn: final hidden state. [1, batch, hidden size]
        """
        # TODO: implement the forward pass
        batch, seq, input_size = inputs.shape
        _, _, hidden_size = h0.shape

        wir, wiz, win = weight_ih.view(3,hidden_size, input_size)
        whr, whz, whn = weight_hh.view(3, hidden_size, hidden_size)
        bir, biz, bin = bias_ih.view(3, hidden_size)
        bhr, bhz, bhn = bias_hh.view(3, hidden_size)

        outputs = torch.zeros(batch, seq, hidden_size, dtype=torch.double)

        ht = h0.squeeze(0)

        for t in range(seq):
            x_t = inputs[:,t,:]

            rt = torch.sigmoid(x_t @ wir.T + bir + ht @ whr.T + bhr)
            zt = torch.sigmoid(x_t @ wiz.T + biz + ht @ whz.T + bhz)
            nt = torch.tanh(x_t @ win.T + bin + rt * (ht @ whn.T + bhn))
            ht = (1-zt) * nt + zt * ht

            outputs[:,t,:] = ht

        hn = ht.unsqueeze(0)

        return outputs, hn
            

       

class GRU(torch.nn.Module):
    """
    Custom GRU layer.
    """

    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.weight_ih = torch.nn.Parameter(torch.empty(3 * hidden_size, input_dim))
        self.weight_hh = torch.nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = torch.nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.empty(3 * hidden_size))

        self.reset_parameters()
        self.fn = GRUFunction.apply

    def forward(self, inputs: torch.Tensor, h0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fn(inputs, h0, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)