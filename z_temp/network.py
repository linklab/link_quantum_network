from netsquid.components import FibreDelayModel, QuantumMemory
from netsquid.examples.teleportation import ClassicalConnection
from netsquid.nodes import Connection, Node

from netsquid.components.qchannel import QuantumChannel
from netsquid.protocols import NodeProtocol
from netsquid.qubits import StateSampler, ketstates
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.models import FixedDelayModel, DepolarNoiseModel
import netsquid.qubits.ketstates as ks

import netsquid as ns


class EntanglingConnection(Connection):
    def __init__(self, length, source_frequency):
        super().__init__(name="EntanglingConnection")
        timing_model = FixedDelayModel(delay=(1e9 / source_frequency))
        self.qsource = QSource(
            name="qsource", state_sampler=StateSampler([ks.b00], [1.0]),
            num_ports=2, timing_model=None, status=SourceStatus.INTERNAL
        )
        self.add_subcomponent(self.qsource)
        qchannel_c2a = QuantumChannel(
            name="qchannel_C2A", length=length / 2, models={"delay_model": FibreDelayModel()}
        )
        qchannel_c2b = QuantumChannel(
            name="qchannel_C2B", length=length / 2, models={"delay_model": FibreDelayModel()}
        )
        # Add channels and forward quantum channel output to external port output:
        self.add_subcomponent(component=qchannel_c2a, forward_output=[("A", "recv")])
        self.add_subcomponent(component=qchannel_c2b, forward_output=[("B", "recv")])

        # Connect qsource output to quantum channel input:
        self.qsource.ports["qout0"].connect(qchannel_c2a.ports["send"])
        self.qsource.ports["qout1"].connect(qchannel_c2b.ports["send"])


def example_network_setup(node_distance=4e-3, depolar_rate=1e7):
    # Setup nodes Alice and Bob with quantum memories:
    noise_model = DepolarNoiseModel(depolar_rate=depolar_rate)
    alice = Node(
        name="Alice", port_names=['qin_charlie', 'cout_bob'],
        qmemory=QuantumMemory(
            name="AliceMemory", num_positions=2, memory_noise_models=[noise_model] * 2
        )
    )
    alice.ports['qin_charlie'].forward_input(alice.qmemory.ports['qin1'])
    bob = Node(
        name="Bob", port_names=['qin_charlie', 'cin_alice'],
        qmemory=QuantumMemory(
            name="BobMemory", num_positions=1, memory_noise_models=[noise_model]
        )
    )
    bob.ports['qin_charlie'].forward_input(bob.qmemory.ports['qin0'])

    # Setup classical connection between nodes:
    c_conn = ClassicalConnection(length=node_distance)
    alice.ports['cout_bob'].connect(c_conn.ports['A'])
    bob.ports['cin_alice'].connect(c_conn.ports['B'])

    # Setup entangling connection between nodes:
    q_conn = EntanglingConnection(length=node_distance, source_frequency=2e7)
    alice.ports['qin_charlie'].connect(q_conn.ports['A'])
    bob.ports['qin_charlie'].connect(q_conn.ports['B'])
    return alice, bob, q_conn.qsource, q_conn, c_conn


class EntanglementGenerationProtocol(NodeProtocol):
    def __init__(self, source, node, partner):
        super().__init__(node)
        self.source = source
        self.partner = partner

    def run(self):
        yield self.await_port_input(self.node.ports["qout0"])
        yield self.await_port_input(self.partner.ports["qout1"])

        q1 = self.node.ports["qout0"].rx_input().items[0]
        q2 = self.partner.ports["qout1"].rx_input().items[0]
        self.node.qmemory.put(q1, positions=0)
        self.partner.qmemory.put(q2, positions=0)

        fidelity = ns.qubits.fidelity([self.node.qmemory.peek(0), self.partner.qmemory.peek(0)], ketstates.b00)
        success = fidelity > 0.9  # Threshold for considering successful entanglement
        print(f"Entangled fidelity (after 5 ns wait) = {fidelity:.3f}: SUCESS: {success}")


def main():
    ns.set_qstate_formalism(ns.QFormalism.DM)
    alice, bob, source, q_conn, c_conn = example_network_setup()

    protocol = EntanglementGenerationProtocol(source, alice, bob)
    protocol.start()
    stats = ns.sim_run(15)
    print(stats)


if __name__ == "__main__":
    main()