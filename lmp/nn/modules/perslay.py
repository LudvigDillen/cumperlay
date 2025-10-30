# Based on the tensorflow implementation from Gudhi
#  found in https://github.com/GUDHI/gudhi-devel/blob/master/src/python/gudhi/tensorflow/perslay.py

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *


class GridPerslayWeight(nn.Module):
    """
    This is a class for computing a differentiable weight function for persistence diagram points. This function is defined from an array that contains its values on a D-dimensional grid.
    """

    grid: Float[Tensor, "N N"]
    grid_bounds: Float[Tensor, "2 2"]

    def __init__(self, grid: Float[Tensor, "*N"], grid_bounds: Float[Tensor, "2 2"]):
        """
        Constructor for the GridPerslayWeight class.

        Parameters:
            grid (NXN 2-dimensional torch tensor): grid of values.
            grid_bnds (2 x 2 torch tensor): boundaries of the grid, of the form [[min_x, max_x], [min_y, max_y]].
        """
        super().__init__()

        self.grid_shape = grid.shape

        assert grid_bounds.shape[0] == 2 and grid_bounds.shape[1] == 2
        assert len(self.grid_shape) == 2 and self.grid_shape[0] == self.grid_shape[1]

        self.register_parameter("grid", nn.Parameter(grid, requires_grad=True))
        self.register_buffer("grid_bounds", grid_bounds)

    def forward(self, diagrams: Float[Tensor, "N F 2"], masks: Float[Tensor, "N F"]) -> Float[Tensor, "N F"]:
        """
        Apply GridPerslayWeight on a tensor containing a list of persistence diagrams.

        Parameters:
            diagrams (N x F x 2): tensor containing N persistence diagrams. 

        Returns:
            weight (N F): tensor containing the weights of the points in the n persistence diagrams.
        """
        indices = []
        for dim in range(2):
            bnds = self.grid_bounds[dim]
            m, M = bnds[0], bnds[1]
            coords = diagrams[:, :, dim:dim+1]
            ids = self.grid_shape[dim] * (coords - m) / (M - m)
            indices.append(ids.to(torch.long))
        weight = self.grid[indices]
        return weight


class GaussianMixturePerslayWeight(nn.Module):
    """
    This is a class for computing a differentiable weight function for persistence diagram points. This function is defined from a mixture of Gaussian functions.
    """

    means: Float[Tensor, "2 N"]
    variances: Float[Tensor, "2 N"]

    def __init__(self, gaussians: Float[Tensor, "4 N"]):
        """
        Constructor for the GaussianMixturePerslayWeight class.

        Parameters:
            gaussians (4 x N torch tensor): parameters of the N Gaussian functions, of the form transpose([[mu_x^1, mu_y^1, sigma_x^1, sigma_y^1], ..., [mu_x^n, mu_y^n, sigma_x^n, sigma_y^n]]). 
        """
        super().__init__()
        D, N = gaussians.shape
        assert D == 4

        with torch.no_grad():
            means = gaussians[:2, :].clone().reshape(1, 1, 2, N)
            variances = gaussians[2:, :].clone().reshape(1, 1, 2, N)

        self.register_parameter(
            "means", nn.Parameter(means, requires_grad=True))
        self.register_parameter("variances", nn.Parameter(
            variances, requires_grad=True))

    def forward(self, diagrams: Float[Tensor, "N F 2"], masks: Float[Tensor, "N F"]) -> Float[Tensor, "N F"]:
        """
        Apply GaussianMixturePerslayWeight on a tensor containing a list of persistence diagrams.

        Parameters:
            diagrams (N x F x 2): tensor containing N persistence diagrams. 

        Returns:
            weight (N F): tensor containing the weights of the points in the n persistence diagrams.
        """
        diagrams = diagrams.unsqueeze(-1)
        dists = torch.square(diagrams - self.means) * \
            (1 / torch.square(self.variances))
        weight = torch.sum(torch.exp(torch.sum(-dists, dim=2)), dim=2)
        return weight


class PowerPerslayWeight(nn.Module):
    """
    This is a class for computing a differentiable weight function for persistence diagram points. This function is defined as a constant multiplied by the distance to the diagonal of the persistence diagram point raised to some power.
    """

    constant: Union[Float[Tensor, ""], float]

    def __init__(self, constant: float, power: float, learnable_constant: bool = True, normalize=False, device=None, dtype=None):
        """
        Constructor for the PowerPerslayWeight class.

        Parameters:
            constant (float): constant value.
            power (float): power applied to the distance to the diagonal. 
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if learnable_constant:
            self.register_parameter("constant", nn.Parameter(
                torch.scalar_tensor(constant, **factory_kwargs))) # type: ignore
        else:
            self.constant = constant
        self.power = power

        self.normalize = normalize
        self.eps = 1e-6

    def forward(self, diagrams: Float[Tensor, "N F 2"], masks: Float[Tensor, "N F"]) -> Float[Tensor, "N F"]:
        """
        Apply PowerPerslayWeight on a tensor containing a list of persistence diagrams.

        Parameters:
            diagrams (N x F x 2): tensor containing N persistence diagrams. 

        Returns:
            weight (N F): tensor containing the weights of the points in the n persistence diagrams.
        """
        weight = self.constant * \
            torch.pow(torch.abs(diagrams[:, :, 1] -
                      diagrams[:, :, 0]) + self.eps, self.power)
        
        if self.normalize:
            weight = weight / (torch.sum(weight, dim=1, keepdim=True) + self.eps) # Normalize over F dimension
        return weight


class GaussianPerslayPhi(nn.Module):
    """
    This is a class for computing a transformation function for persistence diagram points. This function turns persistence diagram points into 2D Gaussian functions centered on the points, that are then evaluated on a regular 2D grid.
    """
    variance: Float[Tensor, ""]

    image_bnds: Float[Tensor, "2 2"]

    mu: Float[Tensor, "2 H W"]

    def __init__(self, image_size: list[int], image_bnds: Float[Tensor, "2 2"], variance: float, device=None, dtype=None):
        """
        Constructor for the GaussianPerslayPhi class.

        Parameters:
            image_size (2 int array): number of grid elements on each grid axis, of the form [n_x, n_y].
            image_bnds (2 x 2 torch tensor): boundaries of the grid, of the form [[min_x, max_x], [min_y, max_y]].
            variance (float): variance of the Gaussian functions. 
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        assert len(image_size) == 2
        self.D = len(image_size)

        D1, D2 = image_bnds.shape
        assert D1 == self.D and D2 == 2

        self.register_parameter("variance", nn.Parameter(
            torch.scalar_tensor(variance, **factory_kwargs)))

        self.image_size = image_size
        self.register_buffer("image_bnds", image_bnds)
        self._calculate_buffers()

    @torch.no_grad
    def _calculate_buffers(self):
        coords = []
        for i in range(self.D):
            bnd_m, bnd_M = self.image_bnds[i, 0].item(
            ), self.image_bnds[i, 1].item()
            step = (bnd_M - bnd_m) / self.image_size[i]
            coords.append(torch.arange(bnd_m, bnd_M, step))

        M = torch.meshgrid(*coords, indexing="xy")  # tf.meshgrid default is xy
        mu = torch.cat([x.unsqueeze(0) for x in M], dim=0)
        self.register_buffer("mu", mu)

    def forward(self, diagrams: Float[Tensor, "N F 2"], masks: Float[Tensor, "N F"]) -> tuple[Float[Tensor, "N F H W 1"], torch.Size]:
        """
        Apply GaussianPerslayPhi on a tensor containing a list of persistence diagrams.

        Parameters:
            diagrams (N x F x 2): tensor containing N persistence diagrams. 

        Returns:
            output (n x F x H x W): tensor containing the evaluations on the 2D grid of the 2D Gaussian functions corresponding to the persistence diagram points, in the form of a 2D image with 1 channel that can be processed with, e.g., convolutional layers. 
            output_shape (torch.Size): shape of the output tensor.
        """
        diag_1, diag_2 = diagrams[:, :, 0:1], diagrams[:, :, 1:2]
        diagrams_d = torch.cat((
            diag_1,
            diag_2 - diag_1,
        ), axis=2)
        for _ in range(self.D):
            diagrams_d = diagrams_d.unsqueeze(-1)
        sq_var = torch.square(self.variance)
        dists = torch.square(diagrams_d - self.mu) / (2 * sq_var)
        gauss = torch.exp(torch.sum(-dists, dim=2)) / (2 * torch.pi * sq_var)
        output = gauss # .unsqueeze(-1)
        output_shape = self.mu.shape[1:] # + tuple([1])
        return output, output_shape


class TentPerslayPhi(nn.Module):
    """
    This is a class for computing a transformation function for persistence diagram points. This function turns persistence diagram points into 1D tent functions (linearly increasing on the first half of the bar corresponding to the point from zero to half of the bar length, linearly decreasing on the second half and zero elsewhere) centered on the points, that are then evaluated on a regular 1D grid.
    """

    samples: Float[Tensor, "S"]

    def __init__(self, samples: Float[Tensor, "S"]):
        """
        Constructor for the TentPerslayPhi class.

        Parameters:
            samples (S float tensor): grid elements on which to evaluate the tent functions, of the form [x_1, ..., x_s].
        """
        super().__init__()
        self.register_parameter("samples", nn.Parameter(
            samples, requires_grad=True))

    def forward(self, diagrams: Float[Tensor, "N F 2"], masks: Float[Tensor, "N F"]) -> tuple[Float[Tensor, "N F S"], torch.Size]:
        """
        Apply TentPerslayPhi on a tensor containing a list of persistence diagrams.

        Parameters:
            diagrams (N x F x 2): tensor containing N persistence diagrams. 

        Returns:
            output (N x F x S): tensor containing the evaluations on the 1D grid of the 1D tent functions corresponding to the persistence diagram points.
            output_shape (torch.Size): shape of the output tensor.
        """
        samples = self.samples.unsqueeze(0).unsqueeze(0)
        xs, ys = diagrams[:, :, 0:1], diagrams[:, :, 1:2]
        output = F.relu(.5 * (ys - xs) -
                        torch.abs(samples - .5 * (ys + xs)))
        output_shape = self.samples.shape
        return output, output_shape


class FlatPerslayPhi(nn.Module):
    """
    This is a class for computing a transformation function for persistence diagram points. This function turns persistence diagram points into 1D constant functions (that evaluate to half of the bar length on the bar corresponding to the point and zero elsewhere), that are then evaluated on a regular 1D grid.
    """

    samples: Float[Tensor, "S"]
    theta: Float[Tensor, ""]

    def __init__(self, samples: Float[Tensor, "S"], theta: float, device=None, dtype=None):
        """
        Constructor for the FlatPerslayPhi class.

        Parameters:
            samples (S float tensor): grid elements on which to evaluate the constant functions, of the form [x_1, ..., x_s].
            theta (float): sigmoid parameter used to approximate the constant function with a differentiable sigmoid function. The bigger the theta, the closer to a constant function the output will be. 

       """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.register_parameter("samples", nn.Parameter(
            samples, requires_grad=True))
        self.register_parameter("theta", nn.Parameter(
            torch.scalar_tensor(theta, **factory_kwargs)))

    def forward(self, diagrams: Float[Tensor, "N F 2"], masks: Float[Tensor, "N F"]) -> tuple[Float[Tensor, "N F S"], torch.Size]:
        """
        Apply FlatPerslayPhi on a tensor containing a list of persistence diagrams.

        Parameters:
            diagrams (N x F x 2): tensor containing N persistence diagrams. 

        Returns:
            output (N x F x S): tensor containing the evaluations on the 1D grid of the 1D constant functions corresponding to the persistence diagram points.
            output_shape (torch.Size): shape of the output tensor.
        """
        xs, ys = diagrams[:, :, 0:1], diagrams[:, :, 1:2]

        samples = self.samples.unsqueeze(0).unsqueeze(0)
        inter = .5 * (ys - xs) - torch.abs(samples - .5 * (ys + xs))
        # output = 1. / (1. + torch.exp(-self.theta * inter))
        output = torch.sigmoid(self.theta * inter)
        output_shape = self.samples.shape
        return output, output_shape


ModuleCallable = Union[nn.Module, Callable]


class Perslay(nn.Module):
    """
    This is a layer for vectorizing persistence diagrams in a differentiable way within a neural network. This function implements the PersLay equation, see `the corresponding article <http://proceedings.mlr.press/v108/carriere20a.html>`_.
    """

    def __init__(self, weight: ModuleCallable, phi: ModuleCallable, perm_op: Union[Callable, str]):
        """
        Constructor for the Perslay class.

        Parameters:
            weight (nn.Module or function): weight function for the persistence diagram points. Can be either :class:`~lmp.nn.perslay.GridPerslayWeight`, :class:`~lmp.nn.perslay.GaussianMixturePerslayWeight`, :class:`~lmp.nn.perslay.PowerPerslayWeight`, or a custom Pytorch function that takes persistence diagrams as argument (represented as an (N x F x 2) tensor, where N is the number of diagrams).
            phi (nn.Module or function): transformation function for the persistence diagram points. Can be either :class:`~lmp.nn.perslay.GaussianPerslayPhi`, :class:`~lmp.nn.perslay.TentPerslayPhi`, :class:`~lmp.nn.perslay.FlatPerslayPhi`, or a custom Pytorch module that takes persistence diagrams as argument (represented as an (N x F x 2) tensor, where N is the number of diagrams).
            perm_op (nn.Module or function): permutation invariant function, such as `torch.sum`, `torch.mean`, `torch.max`, `torch.min`, or a custom Pytorch function that takes two arguments: a tensor and an dimension on which to apply the permutation invariant operation. If perm_op is the string "topk" (where k is a number), this function will be computed as `torch.topk` with parameter `int(k)`.
        """
        super().__init__()
        self.weight = weight
        self.phi = phi
        self.perm_op = perm_op

    def forward(self, diagrams: Float[Tensor, "N F 2"], masks: Float[Tensor, "N F"]):
        """
        Apply Perslay on a tensor containing a list of persistence diagrams.

        Parameters:
            diagrams (N x F x 2): tensor containing N persistence diagrams.

        Returns:
            vector (N x output_shape): tensor containing the vectorizations of the persistence diagrams.
        """
        vector: Tensor
        dim: torch.Size
        weight: Tensor

        vector, dim = self.phi(diagrams, masks)
        weight = self.weight(diagrams, masks) * masks
        for _ in range(len(dim)):
            weight = weight.unsqueeze(-1)
        vector = vector * weight

        permop = self.perm_op
        if type(permop) == str and permop[:3] == 'top':
            k = int(permop[3:])
            # vector = vector.nan_to_num(nan=-1e10)
            # vector = vector + ((masks - 1.0) * 1e10).view(*masks.shape, *[1 for _ in range(len(dim))]) 
            # vector = torch.topk(vector.permute(0, 2, 1), k=k).values
            vector = torch.topk(vector, dim=-2, k=k).values
            # vector = tf.math.top_k(tf.transpose(
            #     vector, perm=[0, 2, 1]), k=k).values
            # vector = vector.view(-1, k * dim[0])
            vector = vector.flatten(-2, -1)
        else:
            vector = permop(vector, dim=1)

        return vector
