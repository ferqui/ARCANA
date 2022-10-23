import torch

class SpikeFunction(torch.autograd.Function):

    @staticmethod
    def pseudo_derivative(V):
        raise NotImplementedError(SpikeFunction.__name__) 

    @staticmethod
    def forward(ctx, V):
        ctx.save_for_backward(V)
        return (V>=0).type(V.dtype)    

    @staticmethod
    def backward(ctx, dy):
        (V,) = ctx.saved_tensors

        dE_dz = dy
        dz_dv_scaled = SpikeFunction.pseudo_derivative(V)
        dE_dv_scaled = dE_dz * dz_dv_scaled

        return dE_dv_scaled

class FastSigmoid(SpikeFunction):

    scale = 10 # Scale value applied to fast sigmoid

    @staticmethod
    def pseudo_derivative(v):
        """
        Return the fast-sigmoid surrogated gradient

        :param V: Neuron voltage to which threshold is applied to.
        :type V: float
        :return: The surrogated gradient of V.
        :rtype: float

        """
        #return torch.maximum(1 - torch.abs(v), torch.tensor(0)) * SpikeFunction.scale
        return 1. / (FastSigmoid.scale * torch.abs(v) + 1.0) ** 2

fast_sigmoid = FastSigmoid.apply

class Triangular(SpikeFunction):

    scale = 0.3 # Scale value applied to fast sigmoid

    @staticmethod
    def pseudo_derivative(v):
        """
        Return the fast-sigmoid surrogated gradient

        :param V: Neuron voltage to which threshold is applied to.
        :type V: float
        :return: The surrogated gradient of V.
        :rtype: float

        """
        return torch.maximum(1 - torch.abs(v), torch.tensor(0)) * Triangular.scale

triangular = Triangular.apply