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
from typing import Generic, NamedTuple, Sequence

from quri_algo.core.estimator import OperatorPowerEstimatorBase, StateT
from quri_algo.problem.interface import ProblemT

from .fourier.coefficient import SPEFourierCoefficient


class SPESample(NamedTuple):
    r"""Holds the classically sampled Fourier coefficient
    :math:`\widetilde{F}(k)` along with the Fourier coefficient of the spectral
    density function :math:`\widetilde{\varrho}(k)`

    Args:
        classical_sample: The Fourier coefficient :math:`\widetilde{F}(k)`.
        op_exp_val:
            The spectral density function's Fourier coefficient
            :math:`\widetilde{\varrho}`(k). This should be the same as
            :\langle U^k \rangle:
    """

    classical_sample: SPEFourierCoefficient
    op_exp_val: complex


@dataclass
class ExpectationValueCollector(Generic[ProblemT, StateT]):
    r"""Computes the expectation value :math:`\langle \psi|U^{k}|\psi\rangle`
    for a given sequence of :class:`SPEFourierCoefficient`. This sets up the
    convolution function's Fourier coefficient.

    ..math::
        \langle \psi|U^{k}|\psi\rangle \widetilde{F}(k)
    """

    operator_power_estimator: OperatorPowerEstimatorBase[ProblemT, StateT]
    state: StateT

    def __call__(
        self, classical_samples: Sequence[SPEFourierCoefficient]
    ) -> Sequence[SPESample]:
        """
        classical_samples: list of \tilde{F}(k)
        """
        sample: list[SPESample] = []
        for c_sample in classical_samples:
            exp_val = self.operator_power_estimator(
                self.state, c_sample.k, c_sample.n_distribute
            )
            spe_sample = SPESample(c_sample, exp_val.value)
            sample.append(spe_sample)
        return sample
