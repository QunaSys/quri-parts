# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "sAS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import networkx as nx
import pytest

from quri_parts.qiskit.backend import device_connectivity_graph


class MockConfiguration:
    def __init__(self, coupling_map: list[tuple[int, int]], num_qubits: int) -> None:
        self.coupling_map = coupling_map
        self.num_qubits = num_qubits


class MockBackend:
    def __init__(self, configuration: MockConfiguration) -> None:
        self.config = configuration

    def configuration(self) -> MockConfiguration:
        return self.config


class TestHTDiagonalization:
    def test_adjacency(self) -> None:
        coupling_map = [(0, 1), (1, 0)]
        num_qubits = 2

        config = MockConfiguration(coupling_map, num_qubits)
        backend = MockBackend(config)

        received_graph = device_connectivity_graph(backend)

        edge_list = [(0, 1)]
        lines = [f"{a} {b}" for (a, b) in edge_list]

        constructed_graph = nx.parse_edgelist(lines, nodetype=int)

        assert nx.is_isomorphic(received_graph, constructed_graph)

    def test_no_coupling_map(self) -> None:
        coupling_map = [(0, 1), (1, 0)]
        num_qubits = 2

        class NoMap(MockConfiguration):
            def __init__(
                self, coupling_map: list[tuple[int, int]], num_qubits: int
            ) -> None:
                # not setting the map.
                self.num_qubits = num_qubits

        config = NoMap(coupling_map, num_qubits)
        backend = MockBackend(config)

        with pytest.raises(ValueError, match="does not have a coupling map."):
            device_connectivity_graph(backend)
