import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import torch.autograd as autograd
from torch.autograd.function import FunctionCtx


# Surrogate gradient estimators based on https://snntorch.readthedocs.io/en/latest/_modules/snntorch/surrogate.html


class SurrogateATanFunc(autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of shifted arc-tan function.

        .. math::

                S&≈\\frac{1}{π}\\text{arctan}(πU \\frac{α}{2}) \\\\
                \\frac{∂S}{∂U}&=\\frac{1}{π}\\frac{1}{(1+(πU\\frac{α}{2})^2)}


    α defaults to 2, and can be modified by calling \
        ``surrogate.atan(alpha=2)``.

    Adapted from:

    *W. Fang, Z. Yu, Y. Chen, T. Masquelier, T. Huang,
    Y. Tian (2021) Incorporating Learnable Membrane Time Constants
    to Enhance Learning of Spiking Neural Networks. Proc. IEEE/CVF
    Int. Conf. Computer Vision (ICCV), pp. 2661-2671.*"""

    @staticmethod
    def forward(ctx: FunctionCtx, input_: Tensor, alpha: float) -> Tensor:
        ctx.save_for_backward(input_)
        ctx.alpha = alpha
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> tuple[Tensor, None]:
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            ctx.alpha
            / 2
            / (1 + (torch.pi / 2 * ctx.alpha * input_).pow_(2))
            * grad_input
        )
        return grad, None


def surrogate_atan(alpha=2.0):
    """ArcTan surrogate gradient enclosed with a parameterized slope."""
    alpha = alpha

    def inner(x):
        return SurrogateATanFunc.apply(x, alpha)

    return inner


class SurrogateStraightThroughEstimator(autograd.Function):
    """
    Straight Through Estimator.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of fast sigmoid function.

        .. math::

                \\frac{∂S}{∂U}=1


    """

    @staticmethod
    def forward(ctx: FunctionCtx, input_: Tensor) -> Tensor:
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        grad_input = grad_output.clone()
        return grad_input


def surrogate_straight_through_estimator():
    """Straight Through Estimator surrogate gradient enclosed
    with a parameterized slope."""

    def inner(x):
        return SurrogateStraightThroughEstimator.apply(x)

    return inner


class SurrogateFastSigmoid(autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of fast sigmoid function.

        .. math::

                S&≈\\frac{U}{1 + k|U|} \\\\
                \\frac{∂S}{∂U}&=\\frac{1}{(1+k|U|)^2}

    :math:`k` defaults to 25, and can be modified by calling \
        ``surrogate.fast_sigmoid(slope=25)``.

    Adapted from:

    *F. Zenke, S. Ganguli (2018) SuperSpike: Supervised Learning in
    Multilayer Spiking Neural Networks. Neural Computation, pp. 1514-1541.*"""

    @staticmethod
    def forward(ctx: FunctionCtx, input_: Tensor, slope: float) -> Tensor:
        ctx.save_for_backward(input_)
        ctx.slope = slope
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> tuple[Tensor, None]:
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (ctx.slope * torch.abs(input_) + 1.0) ** 2
        return grad, None


def surrogate_fast_sigmoid(slope=25):
    """FastSigmoid surrogate gradient enclosed with a parameterized slope."""
    slope = slope

    def inner(x):
        return SurrogateFastSigmoid.apply(x, slope)

    return inner


def get_surrogate_grad_threshold_fn(name, **kwargs):
    if name == "atan":
        fn = surrogate_atan
    elif name == "straight_through":
        fn = surrogate_straight_through_estimator
    elif name == "fast_sigmoid":
        fn = surrogate_fast_sigmoid
    else:
        raise ValueError(f"Unknown surrogate grad threshold function {name}")
    return fn(**kwargs)
