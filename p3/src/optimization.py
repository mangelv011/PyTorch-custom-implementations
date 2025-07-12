# deep learning libraries
import torch

# other libraries
from typing import Iterator, Dict, Any, DefaultDict


class SGD(torch.optim.Optimizer):
    """
    This class is a custom implementation of the SGD algorithm.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]  # para almacenar información de los parámetros

    def __init__(
        self, params: Iterator[torch.nn.Parameter], lr=1e-3, weight_decay: float = 0.0
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # define defaults
        defaults: Dict[Any, Any] = dict(lr=lr, weight_decay=weight_decay)

        # call super class constructor
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """
        # Para cada grupo de parámetros
        for group in self.param_groups:
            # obtenemos el lr y el weight_decay
            lr = group['lr']
            weight_decay = group['weight_decay']
            
            # Para cada parámetro en el grupo
            for param in group['params']:
                if param.grad is None:
                    # Nos saltamos los parámetros que no tienen gradiente
                    continue
                
                grad = param.grad.data.clone()  # Hacemos un clone para evitar problemas con in-place operations
                
                # Aplicamos weight decay si se especifica
                # weight_decay θ = θ - lr * (∇L(θ) + λ*θ)
                if weight_decay != 0:
                    
                    decay_term = torch.zeros_like(param.data)
                    flat_decay = decay_term.view(-1)
                    flat_param = param.data.view(-1)
                    
                    for i in range(len(flat_decay)):
                        flat_decay[i] = flat_param[i] * weight_decay
                    
                    flat_grad = grad.view(-1)
                    for i in range(len(flat_grad)):
                        flat_grad[i] = flat_grad[i] + flat_decay[i]
                
                
                update = torch.zeros_like(grad)
                flat_update = update.view(-1)
                flat_grad = grad.view(-1)
                
                for i in range(len(flat_update)):
                    flat_update[i] = flat_grad[i] * lr
                
                new_param = torch.zeros_like(param.data)
                flat_new_param = new_param.view(-1)
                flat_param = param.data.view(-1)
                
                for i in range(len(flat_new_param)):
                    flat_new_param[i] = flat_param[i] - flat_update[i]
                
            
                param.data.copy_(new_param)


class SGDMomentum(torch.optim.Optimizer):
    """
    This class is a custom implementation of the SGD algorithm with
    momentum.

    Attr:
        param_groups: list with the dict of the parameters.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """
        # TODO
        # Definimos los valores por defecto para este optimizador
        defaults: Dict[Any, Any] = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        
        # Llamamos al constructor de la clase padre
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Attr:
            param_groups: list with the dict of the parameters.
            state: dict with the state for each parameter.
        """
        # TODO
        # Para cada grupo de parámetros
        for group in self.param_groups:
            # Obtenemos los hiperparámetros
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            
            # Para cada parámetro en el grupo
            for param in group['params']:
                if param.grad is None:
                    # Nos saltamos los parámetros que no tienen gradiente
                    continue
                
                # Obtenemos el gradiente
                grad = param.grad.data.clone()
                
                # Aplicamos weight decay si es necesario: g += λ*θ
                if weight_decay != 0:
                    # Implementamos weight decay manualmente
                    decay_term = torch.zeros_like(param.data)
                    flat_decay = decay_term.view(-1)
                    flat_param = param.data.view(-1)
                    
                    for i in range(len(flat_decay)):
                        flat_decay[i] = flat_param[i] * weight_decay
                    
                    flat_grad = grad.view(-1)
                    for i in range(len(flat_grad)):
                        flat_grad[i] = flat_grad[i] + flat_decay[i]
                
                # Obtenemos el estado para este parámetro
                state = self.state[param]
                
                # Inicializamos el buffer de momentum si es la primera vez
                if 'momentum_buffer' not in state:
                    # Creamos un buffer de momentum vacío
                    momentum_buffer = torch.zeros_like(param.data)
                    state['momentum_buffer'] = momentum_buffer
                
                # Obtenemos el buffer de momentum actual
                momentum_buffer = state['momentum_buffer']
                
                # Actualizamos el buffer de momentum: v = μ*v + g
                # Primero multiplicamos por el factor de momentum
                flat_buffer = momentum_buffer.view(-1)
                for i in range(len(flat_buffer)):
                    flat_buffer[i] = flat_buffer[i] * momentum
                
                # Luego agregamos el gradiente
                flat_grad = grad.view(-1)
                for i in range(len(flat_buffer)):
                    flat_buffer[i] = flat_buffer[i] + flat_grad[i]
                
                # Ahora usamos el buffer de momentum para actualizar los parámetros: θ = θ - lr*v
                update = torch.zeros_like(momentum_buffer)
                flat_update = update.view(-1)
                flat_buffer = momentum_buffer.view(-1)
                
                for i in range(len(flat_update)):
                    flat_update[i] = flat_buffer[i] * lr
                
                # Creamos un nuevo parámetro
                new_param = torch.zeros_like(param.data)
                flat_new_param = new_param.view(-1)
                flat_param = param.data.view(-1)
                
                for i in range(len(flat_new_param)):
                    flat_new_param[i] = flat_param[i] - flat_update[i]
                
                # Actualizamos los parámetros
                param.data.copy_(new_param)


class SGDNesterov(torch.optim.Optimizer):
    """
    This class is a custom implementation of the SGD algorithm with
    momentum.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """
        # TODO
        # Definimos los valores por defecto para este optimizador
        defaults: Dict[Any, Any] = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)

        # Llamamos al constructor de la clase padre
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """
        # TODO
        # Para cada grupo de parámetros
        for group in self.param_groups:
            # Obtenemos los hiperparámetros del grupo
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            # Para cada parámetro en el grupo
            for param in group["params"]:
                if param.grad is None:
                    # Nos saltamos los parámetros que no tienen gradiente
                    continue
                
                # Obtenemos el gradiente
                grad = param.grad.data.clone()
                
                # Aplicamos weight decay (regularización L2) si es necesario
                if weight_decay != 0:
                    # Añadimos λ*θ al gradiente: g_t = g_t + λ*θ_t
                    for i in range(grad.numel()):
                        grad.view(-1)[i] += weight_decay * param.data.view(-1)[i]
                
                # Obtenemos el estado para este parámetro
                param_state = self.state[param]
                
                # Inicializamos el buffer de momentum si es la primera vez
                if "momentum_buffer" not in param_state:
                    # Creamos un buffer de momentum inicial igual al gradiente actual
                    param_state["momentum_buffer"] = grad.clone()
                else:
                    # Actualizamos el buffer de momentum: v_t = μ*v_{t-1} + g_t
                    for i in range(grad.numel()):
                        param_state["momentum_buffer"].view(-1)[i] = (
                            param_state["momentum_buffer"].view(-1)[i] * momentum + grad.view(-1)[i]
                        )

                # Aplicamos la corrección de Nesterov: g_t = g_t + μ*v_t
                # Esto "mira hacia adelante" en la dirección del momentum
                for i in range(grad.numel()):
                    grad.view(-1)[i] += momentum * param_state["momentum_buffer"].view(-1)[i]
                
                # Actualizamos los parámetros: θ_t = θ_{t-1} - lr*g_t
                for i in range(param.data.numel()):
                    param.data.view(-1)[i] -= lr * grad.view(-1)[i]


class Adam(torch.optim.Optimizer):
    """
    This class is a custom implementation of the Adam algorithm.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
            betas: coeficientes para los promedios móviles de primer y segundo momento.
            eps: término para mejorar la estabilidad numérica.
            weight_decay: factor de regularización L2.
        """
        # TODO
        # Definimos los valores por defecto
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        
        # Llamamos al constructor de la clase padre
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """
        # TODO
        # Para cada grupo de parámetros
        for group in self.param_groups:
            # Obtenemos los hiperparámetros
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            # Para cada parámetro en el grupo
            for param in group['params']:
                if param.grad is None:
                    continue
                
                # Obtenemos el gradiente
                grad = param.grad.data.clone()
                
                # Aplicamos weight decay si es necesario
                if weight_decay != 0:
                    for i in range(grad.numel()):
                        grad.view(-1)[i] += weight_decay * param.data.view(-1)[i]
                
                # Obtenemos el estado para este parámetro
                state = self.state[param]
                
                # Inicializamos el estado si es la primera vez
                if len(state) == 0:
                    state['step'] = 0
                    # Inicializar estimación del primer momento (media)
                    state['exp_avg'] = torch.zeros_like(param.data)
                    # Inicializar estimación del segundo momento (varianza)
                    state['exp_avg_sq'] = torch.zeros_like(param.data)
                
                # Obtenemos el estado actual
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                # Incrementamos el contador de pasos
                state['step'] += 1
                step = state['step']
                
                # Actualizamos las estimaciones de primer momento (m_t = β₁*m_{t-1} + (1-β₁)*g_t)
                for i in range(exp_avg.numel()):
                    exp_avg.view(-1)[i] = beta1 * exp_avg.view(-1)[i] + (1 - beta1) * grad.view(-1)[i]
                
                # Actualizamos las estimaciones de segundo momento (v_t = β₂*v_{t-1} + (1-β₂)*g_t²)
                for i in range(exp_avg_sq.numel()):
                    exp_avg_sq.view(-1)[i] = beta2 * exp_avg_sq.view(-1)[i] + (1 - beta2) * (grad.view(-1)[i] ** 2)
                
                # Calculamos las correcciones de bias
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Calculamos el paso adaptativo
                step_size = lr / bias_correction1
                
                # Aplicamos la actualización: θ = θ - α * m̂_t / (√v̂_t + ε)
                for i in range(param.data.numel()):
                    # Corrección de bias para m_t: m̂_t = m_t / (1-β₁ᵗ)
                    m_corrected = exp_avg.view(-1)[i]
                    
                    # Corrección de bias para v_t: v̂_t = v_t / (1-β₂ᵗ)
                    v_corrected = exp_avg_sq.view(-1)[i] / bias_correction2
                    
                    # Actualización: θ = θ - α * m̂_t / (√v̂_t + ε)
                    param.data.view(-1)[i] -= step_size * m_corrected / ((v_corrected ** 0.5) + eps)
