import torch


class FastSigmoid(torch.autograd.Function):
    """Fast-sigmoid surrogated gradient
    Apply the fast-sigmoid gradient as a surrogated gradient for the heavyside step function.

    .. math::
       \\frac{\\partial S}{\\partial V} = \\frac{1}{(\\lambda \left|v\\right| + 1.0)^2}

    Where :math:`\\lambda` is a scale factor with default value 10.
    """

    scale = 10  # Scale value applied to fast sigmoid

    @staticmethod
    def pseudo_derivative(v):
        """Compute the gradient of the fast-sigmoid function.

        Args:
            V(float): Neuron voltage to which threshold is applied to.

        Returns:
            float: The fast-sigmoid gradient of V.

        """
        return 1.0 / (FastSigmoid.scale * torch.abs(v) + 1.0) ** 2

    @staticmethod
    def forward(ctx, V):
        """"""
        ctx.save_for_backward(V)
        return (V >= 0).type(V.dtype)

    @staticmethod
    def backward(ctx, dy):
        """"""
        (V,) = ctx.saved_tensors

        dE_dz = dy
        dz_dv_scaled = FastSigmoid.pseudo_derivative(V)
        dE_dv_scaled = dE_dz * dz_dv_scaled

        return dE_dv_scaled


fast_sigmoid = FastSigmoid.apply


class Step(torch.autograd.Function):
    """Step function surrogated gradient
    Use the step function as a surrogated gradient of itself
    """

    @staticmethod
    def pseudo_derivative(V):
        """Compute the step function surrogate gradient.

        Args:
            V(float): Neuron voltage to which threshold is applied to.

        Returns:
            float: The surrogate triangular gradient of V.

        """
        return (V >= 0).type(V.dtype)

    @staticmethod
    def forward(ctx, V):
        """"""
        ctx.save_for_backward(V)
        return (V >= 0).type(V.dtype)

    @staticmethod
    def backward(ctx, dy):
        """"""
        (V,) = ctx.saved_tensors

        dE_dz = dy
        dz_dv_scaled = Step.pseudo_derivative(V)
        dE_dv_scaled = dE_dz * dz_dv_scaled

        return dE_dv_scaled


step = Step.apply


class Triangular(torch.autograd.Function):
    """Triangular surrogated gradient
    Apply the triangular function as a surrogated gradient for the heavyside step function.

    .. math::
       \\frac{\\partial S}{\\partial V} = \\lambda max(1 - \\left|V\\right|, 0)

    Where :math:`\\lambda` is a scale factor with default value 0.3.
    """

    scale = 0.3  # Scale value applied to fast sigmoid

    @staticmethod
    def pseudo_derivative(v):
        """Compute the triangular surrogate gradient.

        Args:
            V(float): Neuron voltage to which threshold is applied to.

        Returns:
            float: The surrogate triangular gradient of V.

        """
        return torch.maximum(1 - torch.abs(v), torch.tensor(0)) * Triangular.scale

    @staticmethod
    def forward(ctx, V):
        """"""
        ctx.save_for_backward(V)
        return (V >= 0).type(V.dtype)

    @staticmethod
    def backward(ctx, dy):
        """"""
        (V,) = ctx.saved_tensors

        dE_dz = dy
        dz_dv_scaled = Triangular.pseudo_derivative(V)
        dE_dv_scaled = dE_dz * dz_dv_scaled

        return dE_dv_scaled


triangular = Triangular.apply


class STE(torch.autograd.Function):
    """
    Straight Through Estimator
    """

    @staticmethod
    def pseudo_derivative(v):
        """Compute the STE surrogate gradient.

        Args:
            V(float): Neuron voltage to which threshold is applied to.

        Returns:
            float: The surrogate STE gradient of V.

        """
        return torch.ones_like(v)

    @staticmethod
    def forward(ctx, V):
        """"""
        ctx.save_for_backward(V)
        return (V >= 0).type(V.dtype)

    @staticmethod
    def backward(ctx, dy):
        """"""
        (V,) = ctx.saved_tensors

        dE_dz = dy
        dz_dv_scaled = STE.pseudo_derivative(V)
        dE_dv_scaled = dE_dz * dz_dv_scaled

        return dE_dv_scaled
