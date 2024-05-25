import netsquid as ns
from netsquid.components import QSource, SourceStatus, QuantumChannel
from netsquid.protocols import NodeProtocol
from netsquid.qubits import StateSampler, ketstates
from netsquid.nodes import Node, Connection
from netsquid.components.models.delaymodels import FibreDelayModel, FixedDelayModel
from netsquid.qubits import ketstates as ks
from netsquid.qubits import qubitapi as qapi


class EntanglingConnection(Connection):
    """A connection that generates entanglement.

    Consists of a midpoint holding a quantum source that connects to
    outgoing quantum channels.

    Parameters
    ----------
    length : float
        End to end length of the connection [km].
    source_frequency : float
        Frequency with which midpoint entanglement source generates entanglement [Hz].
    name : str, optional
        Name of this connection.

    """

    def __init__(self, length, name="EntanglingConnection"):
        super().__init__(name=name)
        self.qsource = QSource(f"qsource_{name}", StateSampler([ks.b00], [1.0]), num_ports=2,
                          timing_model=FixedDelayModel(delay=20),
                          status=SourceStatus.INTERNAL)
        self.add_subcomponent(self.qsource, name="qsource")
        qchannel_c2a = QuantumChannel("qchannel_C2A", length=length / 2,
                                      models={"delay_model": FibreDelayModel()})
        qchannel_c2b = QuantumChannel("qchannel_C2B", length=length / 2,
                                      models={"delay_model": FibreDelayModel()})
        # Add channels and forward quantum channel output to external port output:
        self.add_subcomponent(qchannel_c2a, forward_output=[("A", "recv")])
        self.add_subcomponent(qchannel_c2b, forward_output=[("B", "recv")])
        # Connect qsource output to quantum channel input:
        self.qsource.ports["qout0"].connect(qchannel_c2a.ports["send"])
        self.qsource.ports["qout1"].connect(qchannel_c2b.ports["send"])


class ConditionalEntanglementProtocol(NodeProtocol):
    def __init__(self, node, partner, threshold_fidelity):
        super().__init__(node)
        self.partner = partner
        self.threshold_fidelity = threshold_fidelity

    def run(self):
        while True:
            # Wait for entangled qubits to be generated
            yield self.await_port_input(self.node.ports["qmemory"])

            # Check if the qubits are entangled
            q1 = self.node.qmemory.pop(0)
            q2 = self.partner.qmemory.pop(0)
            fidelity = qapi.fidelity([q1, q2], ketstates.b00)
            success = fidelity > self.threshold_fidelity

            if success:
                self.send_signal(signal_label=Signals.SUCCESS)
            else:
                self.send_signal(signal_label=Signals.FAILURE)

# Instantiate and run the protocol
protocol_a = ConditionalEntanglementProtocol(node_a, node_b, threshold_fidelity)
protocol_b = ConditionalEntanglementProtocol(node_b, node_a, threshold_fidelity)

protocol_a.start()
protocol_b.start()

# Example of generating an entangled pair
ev = EntanglingConnection(length=1.0)
qubits = ev.ent_source.generate()
q1, q2 = qubits.items
print(f"Qubit at Alice: {q1}")
print(f"Qubit at Bob: {q2}")

# Run a simple simulation
ns.sim_run(duration=10)