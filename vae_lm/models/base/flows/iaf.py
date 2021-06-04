from .made import MADE
from .maf import MAFlow
from .flow import Flow


@Flow.register("iaf")
class IAFlow(MAFlow):
    """
    An implementation of Inverse Autoregressive Flow for Density Estimation.
    IAF is just like MAF but we swap forward and backward.

    Parameters
    ----------
    made : `MADE`, required
        Instance of MADE (Masked autoencoder for Density estimation).
    parity : `bool`, required
        Simply flipping the transformation on last dimension.
        Works good when we stack IAF.
    """

    def __init__(self, made: MADE, parity: bool) -> None:
        super().__init__(made, parity)
        self.forward, self.backward = self.backward, self.forward
