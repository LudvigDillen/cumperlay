import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import lmp
from lmp.util.typing import *


def _neight2channels_like_kernel(kernel: torch.Tensor) -> torch.Tensor:
    h, w = kernel.size()
    kernel = torch.eye(h * w, dtype=kernel.dtype, device=kernel.device)
    return kernel.view(h * w, 1, h, w)


def erosion(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4
) -> torch.Tensor:
    r"""Return the eroded image applying the same kernel in each channel.

    .. image:: _static/img/erosion.png

    The kernel must have 2 dimensions.

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied. Its shape is :math:`(k_x, k_y)`.
            For full structural elements use torch.ones_like(structural_element).
        structuring_element (torch.Tensor, optional): Structuring element used for the grayscale dilation.
            It may be a non-flat structuring element.
        origin: Origin of the structuring element. Default: ``None`` and uses the center of
            the structuring element as origin (rounding towards zero).
        border_type: It determines how the image borders are handled, where ``border_value`` is the value
            when ``border_type`` is equal to ``constant``. Default: ``geodesic`` which ignores the values that are
            outside the image when applying the operation.
        border_value: Value to fill past edges of input if border_type is ``constant``.
        max_val: The value of the infinite elements in the kernel.

    Returns:
        Eroded image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(5, 5)
        >>> output = erosion(tensor, kernel)
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) != 4:
        raise ValueError(
            f"Input size must have 4 dimensions. Got {tensor.dim()}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(
            f"Kernel type is not a torch.Tensor. Got {type(kernel)}")

    if len(kernel.shape) != 2:
        raise ValueError(
            f"Kernel size must have 2 dimensions. Got {kernel.dim()}")

    # origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # pad
    pad_e: List[int] = [origin[1], se_w -
                        origin[1] - 1, origin[0], se_h - origin[0] - 1]
    if border_type == "geodesic":
        border_value = max_val
        border_type = "constant"
    output: torch.Tensor = F.pad(
        tensor, pad_e, mode=border_type, value=border_value)

    # computation
    if structuring_element is None:
        neighborhood = torch.zeros_like(kernel)
        neighborhood[kernel == 0] = -max_val
    else:
        neighborhood = structuring_element.clone()
        neighborhood[kernel == 0] = -max_val

    B, C, H, W = tensor.size()
    Hpad, Wpad = output.shape[-2:]
    reshape_kernel = _neight2channels_like_kernel(kernel)
    output, _ = F.conv2d(
        output.view(B * C, 1, Hpad, Wpad), reshape_kernel, padding=0, bias=-neighborhood.view(-1)
    ).min(dim=1)
    
    output = output.view(B, C, H, W)

    return output


class Erosion(nn.Module):
    kernel: Tensor
    structuring_element: Optional[Tensor]
    
    def __init__(
        self,
        kernel_size: int = 3,
        kernel: Optional[torch.Tensor] = None,
        structuring_element: Optional[torch.Tensor] = None,
        origin: Optional[List[int]] = None,
        border_type: str = "geodesic",
        border_value: float = 0.0,
        max_val: float = 1e4
    ) -> None:
        super().__init__()

        if kernel is None:
            kernel = torch.ones((kernel_size, kernel_size))
        self.register_buffer("kernel", kernel)
        self.register_buffer("structuring_element", structuring_element)

        self.origin = origin
        self.border_type = border_type
        self.border_value = border_value
        self.max_val = max_val

    def forward(self, x: Tensor) -> Tensor:
        return erosion(
            x,
            self.kernel,
            self.structuring_element,
            origin=self.origin,
            border_type=self.border_type,
            border_value=self.border_value,
            max_val=self.max_val,
        )
