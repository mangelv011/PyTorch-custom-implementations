# deep learning libraries
import torch

# other libraries
from typing import Iterator, Dict, Any, DefaultDict

## SGD
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
        # TODO
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']

            for param in group['params']:
                grad = param.grad
                if grad is None:
                    continue

                if weight_decay != 0:
                    grad += weight_decay*param.data
                
                param.data -= lr*grad



## SGDMomentum
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
            momentum: momentum coefficient. Defaults to 0.9.
            weight_decay: weight decay (L2 penalty). Defaults to 0.0.
        """
        # Define defaults
        defaults: Dict[Any, Any] = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        
        # Call super class constructor
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
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            for param in group['params']:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]
                if weight_decay != 0:
                    grad += weight_decay*param.data
                if momentum != 0:
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(param.data)
                    state['momentum_buffer'] *= momentum 
                    state['momentum_buffer'] += grad
                    grad = state['momentum_buffer']
                param.data -= lr*grad



## SGDNesterov 
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
        defaults: Dict[Any, Any] = dict(params=params, lr=lr, momentum=momentum, weight_decay=weight_decay)

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
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for param in group['params']:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]
                if weight_decay != 0:
                    grad += weight_decay * param.data
                if momentum != 0:

                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(param.data)

                    state['momentum_buffer'] *= momentum
                    state['momentum_buffer'] += grad

                    grad += momentum*state['momentum_buffer']
                
                param.data -= lr*grad



## Adam
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
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for param in group['params']:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]

                if weight_decay != 0:
                    grad += weight_decay * param.data

                if len(state) == 0:
                    state['step'] = 0 
                    state['first_moment'] = torch.zeros_like(param.data)
                    state['second_moment'] = torch.zeros_like(param.data)

                state['step'] += 1 

                state['first_moment'] *= beta1
                state['first_moment'] += (1-beta1) * grad

                state['second_moment'] *= beta2
                state['second_moment'] += (1-beta2) * torch.square(grad)

                bias_correction_1 = state['first_moment'] / (1- (beta1**state['step']))
                bias_correction_2 = state['second_moment'] / (1- (beta2**state['step']))


                param.data -= (lr * bias_correction_1) / (torch.sqrt(bias_correction_2) + eps)




class NAdam(torch.optim.Optimizer):
    """
    This class is a custom implementation of the NAdam algorithm.

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
        momentum_decay: float = 0.004,
    ) -> None:
        """
        This is the constructor for NAdam.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
            betas: betas for Adam. Defaults to (0.9, 0.999).
            eps: epsilon for approximation. Defaults to 1e-8.
            weight_decay: weight decay. Defaults to 0.0.
            momentum_decay: momentum decay. Defaults to 0.004.
        """

        # TODO
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum_decay=momentum_decay)
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
        
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            momentum_decay = group['momentum_decay']
            for param in group['params']:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]

                if weight_decay != 0:
                    grad += weight_decay * param.data

                if len(state) == 0: 
                    state['step'] = 0 
                    state['first_moment'] = torch.zeros_like(param.data)
                    state['second_moment'] = torch.zeros_like(param.data)
                    state['ut_list'] = []
                
                state['step'] += 1
                t = state['step']

                ut = b1 * (1-(0.5*(0.96**(t*momentum_decay))))
                ut_next = b1 * (1-(0.5*(0.96**((t+1)*momentum_decay))))

                if t == 1:
                    state['ut_list'].append(ut)
                    state['ut_list'].append(ut_next)
                else:
                    state['ut_list'].append(ut_next)


                state['first_moment'] *= b1
                state['first_moment'] += (1-b1)*grad

                state['second_moment'] *= b2
                state['second_moment'] += (1-b2)*(grad**2)

                prod1 = 1
                for i in range(t+1):
                    prod1 *= state['ut_list'][i]

                prod2 = 1
                for i in range(t):
                    prod2 *= state['ut_list'][i]

                bias_correction_1 = ((ut_next*state['first_moment'])/(1-prod1)) + (((1-ut)*grad)/(1-prod2))
                bias_correction_2 = state['second_moment'] / (1-(b2**t))

                param.data -= (lr * bias_correction_1) / (torch.sqrt(bias_correction_2) + eps)
