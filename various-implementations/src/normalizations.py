# deep learning libraries
import torch


## GroupNorm
class GroupNorm(torch.nn.Module):
    """
    This class implements the GroupNorm layer of torch.

    Attributes:
        num_groups: Number of groups.
        num_channels: Number of channels.
        eps: epsilon to avoid division by 0.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5) -> None:
        """
        This method is the constructor of GroupNorm class.

        Args:
            num_groups: Number of groups.
            num_channels: Number of channels.
            eps: epsilon to avoid division by 0. Defaults to 1e-5.

        Returns:
            None.
        """

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the layer.

        Args:
            inputs: Inputs tensor. Dimensions: [batch, channels,
                height, width].

        Returns:
            Outputs tensor. Dimensions: [batch, channels, height,
                width].
        """

        # TODO
        batch, channels, height, width = inputs.shape
        inputs = inputs.view(batch, self.num_groups, -1)
        mean = torch.mean(inputs, dim=-1, keepdim=True)
        std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
        outputs = (inputs - mean) / std
        outputs = outputs.view(batch, channels, height, width)
        return outputs


## BatchNorm
class BatchNorm(torch.nn.Module):
    """
    This class implements the BatchNorm layer of torch.

    Attributes:
        num_groups: Number of groups.
        num_channels: Number of channels.
        eps: epsilon to avoid division by 0.
    """

    def __init__(
        self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True
    ) -> None:
        """
        This method is the constructor of GroupNorm class.

        Args:
            num_groups: Number of groups.
            num_channels: Number of channels.
            eps: epsilon to avoid division by 0. Defaults to 1e-5.
            affine: Indicator to perform affine transformation. Defaults to True.
        """
        super().__init__()

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(num_channels))
            self.bias = torch.nn.Parameter(torch.empty(num_channels))
            self.reset_parameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the layer.

        Args:
            inputs: Inputs tensor. Dimensions: [batch, channels,
                height, width].

        Returns:
            Outputs tensor. Dimensions: [batch, channels, height,
                width].
        """
        # Save original shape
        _, channels, _, _ = inputs.shape

        mean = inputs.mean(dim=(0, 2, 3), keepdim=True)  # [1, C, 1, 1]
        var = inputs.var(dim=(0, 2, 3), keepdim=True, unbiased=False)

        outputs = (inputs - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            outputs = outputs * self.weight.view(1, channels, 1, 1) + self.bias.view(
                1, channels, 1, 1
            )

        return outputs

    def reset_parameters(self) -> None:
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)


## LayerNorm
class LayerNorm(torch.nn.Module):
    """
    This class implements the LayerNorm layer of torch.

    Attributes:
        num_features: Number of features.
        eps: epsilon to avoid division by 0.
    """

    def __init__(
        self, num_features: int, eps: float = 1e-5, affine: bool = True
    ) -> None:
        """
        This method is the constructor of LayerNorm class.

        Args:
            num_features: Number of features.
            eps: epsilon to avoid division by 0. Defaults to 1e-5.
            affine: Indicator to perform affine transformation. Defaults to True.

        Returns:
            None.
        """
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(num_features))
            self.bias = torch.nn.Parameter(torch.empty(num_features))
            self.reset_parameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the layer.

        Args:
            inputs: Inputs tensor. Dimensions: [batch, channels,
                height, width].

        Returns:
            Outputs tensor. Dimensions: [batch, channels, height,
                width].
        """
        # TODO
        _, channels, _, _ = inputs.shape
        mean = inputs.mean(dim=(1, 2, 3), keepdim=True)  # B, 1, 1, 1
        var = inputs.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        outputs = (inputs - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            outputs = outputs * self.weight.view(1, channels, 1, 1) + self.bias.view(
                1, channels, 1, 1
            )

        return outputs

    def reset_parameters(self) -> None:
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)


## InstanceNorm
class InstanceNorm(torch.nn.Module):
    """
    This class implements the InstanceNorm layer of torch.

    Attributes:
        num_features: Number of features.
        eps: epsilon to avoid division by 0.
        affine: Indicator to perform affine transformation.
    """

    def __init__(
        self, num_features: int, eps: float = 1e-5, affine: bool = True
    ) -> None:
        """
        This method is the constructor of InstanceNorm class.

        Args:
            num_features: Number of features.
            eps: epsilon to avoid division by 0. Defaults to 1e-5.
            affine: Indicator to perform affine transformation. Defaults to True.

        Returns:
            None.
        """
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(num_features))
            self.bias = torch.nn.Parameter(torch.empty(num_features))
            self.reset_parameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the layer.

        Args:
            inputs: Inputs tensor. Dimensions: [batch, channels, height, width].

        Returns:
            Outputs tensor. Dimensions: [batch, channels, height, width].
        """
        # TODO
        _, channels, _, _ = inputs.shape
        mean = inputs.mean(dim=(2, 3), keepdim=True)  # [batch, channels, 1,1]
        var = inputs.var(dim=(2, 3), keepdim=True, unbiased=False)

        outputs = (inputs - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            outputs = outputs * self.weight.view(1, channels, 1, 1) + self.bias.view(
                1, channels, 1, 1
            )

        return outputs

    def reset_parameters(self) -> None:
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)
