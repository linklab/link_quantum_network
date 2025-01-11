import netsquid as ns


from netsquid.nodes import Node
from netsquid.qubits import ketstates as ks
from netsquid.components import QuantumMemory
from netsquid.components.qchannel import QuantumChannel
from netsquid.protocols.protocol import Protocol
from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.components.models import FixedDelayModel, FibreLossModel
from netsquid.components.models.qerrormodels import DepolarNoiseModel

class EntanglementProtocol(Protocol):
    def __init__(self, node, channel, position):
        self.node = node
        self.channel = channel
        self.position = position

    def run(self):
        qubit_0, qubit_1 = ns.qubits.create_qubits(2)
        ns.qubits.operate(qubit_0, ns.H)
        ns.qubits.operate([qubit_0, qubit_1], ns.CNOT)

        self.node.qmemory.put([qubit_0], positions=[self.position])
        self.channel.send(qubit_1)

class BellMeasurementProtocol(NodeProtocol):
    def run(self):
        qubit_0 = self.node.qmemory.peek(positions=[0])
        qubit_1 = self.node.qmemory.peek(positions=[1])
        ns.qubits.operate(qubit_0[0], ns.H)
        ns.qubits.operate([qubit_0[0], qubit_1[0]], ns.CNOT)

def example_network_setup(length=5):
    node_a = Node("node_A", port_names=['qin_ab'])
    node_b = Node("node_B")
    node_c = Node("node_C", port_names=['qin_bc'])

    depolar_noise = DepolarNoiseModel(depolar_rate=500)
    qmemory_a = QuantumMemory("memory_A", num_positions=1, memory_noise_models=[depolar_noise])
    qmemory_b = QuantumMemory("memory_B", num_positions=2, memory_noise_models=[depolar_noise]*2)
    qmemory_c = QuantumMemory("memory_C", num_positions=1, memory_noise_models=[depolar_noise])

    node_a.add_subcomponent(qmemory_a, name="memoryA")
    node_b.add_subcomponent(qmemory_b, name="memoryB")
    node_c.add_subcomponent(qmemory_c, name="memoryC")

    qchannel_ab = QuantumChannel(
        name="qchannel_ba",
        length=length,
        models={
            "quantum_loss_model": FibreLossModel(p_loss_init=0.2, p_loss_length=0.25)
        }
    )
    qchannel_bc = QuantumChannel(
        name="qchannel_bc",
        length=length,
        models={
            "quantum_loss_model": FibreLossModel(p_loss_init=0.2, p_loss_length=0.25)
        }
    )
    qchannel_ab.ports['recv'].connect(node_a.ports['qin_ab'])
    qchannel_bc.ports['recv'].connect(node_c.ports['qin_bc'])

    node_a.ports['qin_ab'].forward_input(node_a.qmemory.ports['qin0'])
    node_c.ports['qin_bc'].forward_input(node_c.qmemory.ports['qin0'])

    return node_a, node_b, node_c, qchannel_ab, qchannel_bc

if __name__ == "__main__":
    ns.set_qstate_formalism(ns.QFormalism.DM)
    node_a, node_b, node_c, qchannel_ab, qchannel_bc = example_network_setup(length=1)
    protocol_ab = EntanglementProtocol(node_b, qchannel_ab, position=0)
    protocol_bc = EntanglementProtocol(node_b, qchannel_bc, position=1)
    protocol_swap = BellMeasurementProtocol(node_b)

    for i in range(3):
        protocol_ab.reset()
        protocol_bc.reset()
        ns.sim_run(duration=1e9 / 100000 * 25 * 50)

        q0, = node_a.qmemory.peek(positions=[0])
        q1, = node_b.qmemory.peek(positions=[0])
        q2, = node_b.qmemory.peek(positions=[1])
        q3, = node_c.qmemory.peek(positions=[0])

        print(q0)
        print(q1)
        print(q2)
        print(q3)

        print("q0.qstate == q1.qstate: ", q0.qstate == q1.qstate)
        print("q2.qstate == q3.qstate: ", q2.qstate == q3.qstate)
        print("q0.qstate == q2.qstate: ", q0.qstate == q2.qstate)
        print("q0.qstate == q3.qstate: ", q0.qstate == q3.qstate)
        print("q1.qstate == q3.qstate: ", q1.qstate == q3.qstate)

        if q0 is not None and q1 is not None:
            print("Entanglement generation success (A-B)", end=" ")
            print("Fidelity: {}".format(ns.qubits.fidelity([q0, q1], ks.b00)))
        else:
            print("Entanglement generation fail (A-B)")

        if q2 is not None and q3 is not None:
            print("Entanglement generation success (B-C)", end=" ")
            print("Fidelity: {}".format(ns.qubits.fidelity([q2, q3], ks.b00)))
        else:
            print("Entanglement generation fail (B-C)")

    # protocol_swap.start()
    # ns.sim_run(duration=1000)
    #
    # print("SWAPPING")
    # print("q0.qstate == q1.qstate: ", q0.qstate == q1.qstate)
    # print("q2.qstate == q3.qstate: ", q2.qstate == q3.qstate)
    # print("q0.qstate == q2.qstate: ", q0.qstate == q2.qstate)
    # print("q0.qstate == q3.qstate: ", q0.qstate == q3.qstate)
    # print("q1.qstate == q3.qstate: ", q1.qstate == q3.qstate)
    #
    # if q0 is not None and q1 is not None:
    #     print("Entanglement generation success (A-B)", end=" ")
    #     print("Fidelity: {}".format(ns.qubits.fidelity([q0, q1], ks.b00)))
    # else:
    #     print("Entanglement generation fail (A-B)")
    #
    # if q2 is not None and q3 is not None:
    #     print("Entanglement generation success (B-C)", end=" ")
    #     print("Fidelity: {}".format(ns.qubits.fidelity([q2, q3], ks.b00)))
    # else:
    #     print("Entanglement generation fail (B-C)")