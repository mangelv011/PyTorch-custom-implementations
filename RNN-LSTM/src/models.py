# deep learning libraries
import torch

# other libraries
import math
from typing import Any


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
        # TODO
        # Guardar las entradas para el backward pass
        ctx.save_for_backward(inputs, h0, weight_ih, weight_hh, bias_ih, bias_hh)

        # Obtener dimensiones
        batch_size, seq_len, _ = inputs.shape
        hidden_size = weight_ih.shape[0]

        # Inicializar tensor para almacenar todos los estados ocultos
        # [batch, sequence, hidden_size]
        outputs = torch.zeros(
            batch_size, seq_len, hidden_size, device=inputs.device, dtype=inputs.dtype
        )

        # Inicializar h_t con el estado inicial h0 (quitar dimensión extra)
        h_t = h0.squeeze(0)

        # Lista para almacenar todos los estados ocultos (para el backward)
        hidden_states = [h_t]

        # Iterar sobre la secuencia
        for t in range(seq_len):
            # Obtener la entrada en el tiempo t
            x_t = inputs[:, t, :]

            # PyTorch implementa RNN como:
            # h_t = ReLU(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
            gate_input = torch.mm(x_t, weight_ih.t()) + bias_ih
            gate_hidden = torch.mm(h_t, weight_hh.t()) + bias_hh
            h_t = torch.relu(gate_input + gate_hidden)

            # Guardar el estado oculto para esta paso de tiempo
            outputs[:, t, :] = h_t
            hidden_states.append(h_t)

        # El último estado oculto para cada elemento del batch
        # PyTorch devuelve esto como [1, batch, hidden_size]
        h_n = h_t.unsqueeze(0)

        # Guardar los estados ocultos para backward
        ctx.hidden_states = hidden_states

        return outputs, h_n

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
        # TODO
        # Recuperar los datos guardados del contexto
        inputs, h0, weight_ih, weight_hh, bias_ih, bias_hh = ctx.saved_tensors
        hidden_states = ctx.hidden_states

        # Obtener dimensiones
        _, seq_len, _ = inputs.shape

        # Inicializar gradientes
        grad_inputs = torch.zeros_like(inputs)
        grad_h0 = torch.zeros_like(h0.squeeze(0))
        grad_weight_ih = torch.zeros_like(weight_ih)
        grad_weight_hh = torch.zeros_like(weight_hh)
        grad_bias_ih = torch.zeros_like(bias_ih)
        grad_bias_hh = torch.zeros_like(bias_hh)

        # Inicializar gradiente del estado oculto en el último paso de tiempo
        # Combinar los gradientes del output y del estado final
        grad_h = grad_hn.squeeze(0)

        # Backpropagation a través del tiempo (BPTT)
        for t in reversed(range(seq_len)):
            # Añadir gradiente del output en el tiempo t
            grad_h = grad_h + grad_output[:, t, :]

            # Obtener los estados ocultos relevantes
            h_prev = hidden_states[t]
            _ = hidden_states[t + 1]

            # Obtener las entradas en el tiempo t
            x_t = inputs[:, t, :]

            # Calcular el ReLU gradiente: 1 si el valor era > 0, 0 en caso contrario
            gate_input = torch.mm(x_t, weight_ih.t()) + bias_ih
            gate_hidden = torch.mm(h_prev, weight_hh.t()) + bias_hh
            relu_grad = (gate_input + gate_hidden) > 0

            # Aplicar el gradiente de ReLU
            grad_h = grad_h * relu_grad.float()

            # Calcular gradientes de pesos y bias
            grad_weight_ih += torch.mm(grad_h.t(), x_t)
            grad_weight_hh += torch.mm(grad_h.t(), h_prev)
            grad_bias_ih += grad_h.sum(dim=0)
            grad_bias_hh += grad_h.sum(dim=0)

            # Calcular gradientes para la entrada y estado oculto anterior
            grad_inputs[:, t, :] = torch.mm(grad_h, weight_ih)
            grad_h_prev = torch.mm(grad_h, weight_hh)

            # Actualizar gradiente para el paso de tiempo anterior
            grad_h = grad_h_prev

        # El gradiente de h0 es el gradiente acumulado del primer paso
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


class MyModel(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        """
        This method is the constructor of the class.

        Args:
            hidden_size: hidden size of the RNN layers
        """
        # TODO
        super(MyModel, self).__init__()

        # Define input features (24 hours data per day)
        input_size = 24
        # Output is next day's 24 hourly prices
        output_size = 24

        # First GRU layer (faster than LSTM and often performs better for time series)
        self.gru1 = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size * 2,
            num_layers=2,
            batch_first=True,
            dropout=0.25,
        )

        # Attention mechanism for focusing on important patterns
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1),
        )

        # Skip connection
        self.skip_connection = torch.nn.Linear(input_size, hidden_size)

        # Layer normalization for better stability
        self.layer_norm = torch.nn.LayerNorm(hidden_size * 2)

        # Prediction network with residual connection
        self.fc1 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)

        # Activation functions
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)

        # Convert to double precision
        self.double()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            inputs: inputs tensor. Dimensions: [batch, number of past days, 24].

        Returns:
            output tensor. Dimensions: [batch, 24].
        """
        # TODO
        batch_size, seq_len, _ = inputs.shape

        # Get the last day's input for skip connection
        last_day = inputs[:, -1, :]

        # Pass through GRU layers
        gru_out, _ = self.gru1(inputs)  # [batch, seq_len, hidden_size*2]

        # Apply layer normalization
        gru_out = self.layer_norm(gru_out)

        # Apply attention mechanism
        attention_weights = self.attention(gru_out).squeeze(-1)
        attention_weights = torch.softmax(attention_weights, dim=1).unsqueeze(-1)
        context_vector = torch.sum(gru_out * attention_weights, dim=1)

        # Process through prediction network with residual connections
        x = self.fc1(context_vector)
        x = self.relu(x)
        x = self.dropout(x)

        residual = x
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x + residual  # Residual connection

        # Add skip connection from the last day
        skip = self.skip_connection(last_day)

        # Final layer with skip connection addition
        predictions = self.fc3(x + skip)

        return predictions
