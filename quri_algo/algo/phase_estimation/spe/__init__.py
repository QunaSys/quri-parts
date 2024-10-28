# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from .gaussian import GaussianFittingGSEE, GaussianFittingPhaseEstimation
from .lt22 import (
    LT22GSEE,
    LT22PhaseEstimation,
    SingleSignalLT22GSEE,
    SingleSignalLT22PhaseEstimation,
)


@dataclass
class StepFunctionParam:
    r"""The parameters for building the step function convolution function.

    Args:
        d: The maximal value of the Fourier mode k.
        delta: The parameter that controls the precision:

            .. math::
                |F_{d, \delta}(x) - H(x)| \leq \epsilon \quad \forall x \in [-\pi + \delta, - \delta] \cup [\delta, \pi - \delta]

        n_sample:
            Number of samples to construct the step function.
    """

    d: int
    delta: float
    n_sample: int


@dataclass
class GaussianParam:
    r"""Hyperparameters of the Gaussian distribution function.

    .. math::
        G_{\sigma}(x) = \int_{-T}^{T} e^{-\frac{1}{2}(\sigma^2 \pi^2 k^2)} e^{2\pi i k x} dk

    Args:
        T: The boundary of the Fourier integration region.
        N: Number of points to disceretize the integration interval.
        sigma: Standard deviation of the Gaussian distribution.
        n_sample: Number of samples to sample the Fourier coefficient
            :math:`e^{-\frac{1}{2}(\sigma^2 \pi^2 k^2)}`
    """

    T: float
    N: int
    sigma: float
    n_sample: int


__all__ = [
    "LT22GSEE",
    "LT22PhaseEstimation",
    "GaussianFittingGSEE",
    "GaussianParam",
    "GaussianFittingPhaseEstimation",
    "SingleSignalLT22GSEE",
    "SingleSignalLT22PhaseEstimation",
    "StepFunctionParam",
]
