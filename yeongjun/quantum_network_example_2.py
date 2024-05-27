import netsquid as ns

from netsquid.nodes.network import Network
from netsquid.nodes import Node
from netsquid.components import QuantumMemory
from netsquid.qubits import ketstates as ks
from netsquid.components.qchannel import QuantumChannel
from netsquid.nodes.connections import Connection
from netsquid.components.models import FixedDelayModel, FibreDelayModel, FibreLossModel
from netsquid.qubits import StateSampler
from netsquid.components.qsource import QSource, SourceStatus


class EntanglingConnection(Connection):
    def __init__(self, p, length, source_frequency):
        super().__init__(name="EntanglingConnection")
        timing_model = FixedDelayModel(delay=(1e9 / source_frequency))
        qsource = QSource(f"QS{p}", StateSampler([ks.b00], [1.0]), num_ports=2,
                          timing_model=timing_model,
                          status=SourceStatus.INTERNAL)
        self.add_subcomponent(qsource)
        qchannel_r = QuantumChannel("qchannel_r", length=length / 2,
                                      models={"loss_model": FibreLossModel(p_loss_init=0.2,  p_loss_length=0.25)})
        qchannel_l = QuantumChannel("qchannel_l", length=length / 2,
                                      models={"loss_model": FibreLossModel(p_loss_init=0.2, p_loss_length=0.25)})
        # Add channels and forward quantum channel output to external port output:
        self.add_subcomponent(qchannel_r, forward_output=[("A", "recv")])
        self.add_subcomponent(qchannel_l, forward_output=[("B", "recv")])
        # Connect qsource output to quantum channel input:
        qsource.ports["qout0"].connect(qchannel_r.ports["send"])
        qsource.ports["qout1"].connect(qchannel_l.ports["send"])

def swapping(node):
    q0, = node.qmemory.peek(positions=[0])
    q1, = node.qmemory.peek(positions=[1])
    ns.qubits.operate(q0, ns.H)
    ns.qubits.operate([q0, q1], ns.CNOT)

def example_network_setup():
    network = Network("Entangle_swapping")
    node_a = Node("node_A", port_names=['qin_ab'])
    node_b = Node("node_B", port_names=['qin_ab', 'qin_bc'])
    node_c = Node("node_C", port_names=['qin_bc'])

    qmemory_a = QuantumMemory("memory_A", num_positions=2)
    qmemory_b = QuantumMemory("memory_B", num_positions=2)
    qmemory_c = QuantumMemory("memory_C", num_positions=2)

    node_a.add_subcomponent(qmemory_a, name="memoryA")
    node_b.add_subcomponent(qmemory_b, name="memoryB")
    node_c.add_subcomponent(qmemory_c, name="memoryC")

    node_a.ports['qin_ab'].forward_input(node_a.qmemory.ports['qin0'])
    node_b.ports['qin_ab'].forward_input(node_b.qmemory.ports['qin0'])
    node_b.ports['qin_bc'].forward_input(node_b.qmemory.ports['qin1'])
    node_c.ports['qin_bc'].forward_input(node_c.qmemory.ports['qin0'])

    q_conn_ab = EntanglingConnection(p = 'R', length=4e-3, source_frequency=2e7)
    q_conn_bc = EntanglingConnection(p = 'L', length=4e-3, source_frequency=2e7)

    node_a.ports['qin_ab'].connect(q_conn_ab.ports['A'])
    node_b.ports['qin_ab'].connect(q_conn_ab.ports['B'])
    node_b.ports['qin_bc'].connect(q_conn_bc.ports['A'])
    node_c.ports['qin_bc'].connect(q_conn_bc.ports['B'])

    return node_a, node_b, node_c, q_conn_ab, q_conn_bc

ns.set_qstate_formalism(ns.QFormalism.DM)
node_a, node_b, node_c, *_ = example_network_setup()

stats = ns.sim_run(15)

q0, = node_a.qmemory.peek(positions=[0])
q1, = node_b.qmemory.peek(positions=[0])
q2, = node_b.qmemory.peek(positions=[1])
q3, = node_c.qmemory.peek(positions=[0])

print("q0 == q1: ", q0.qstate == q1.qstate)
print("q0 == q2: ", q0.qstate == q2.qstate)
print("q0 == q3: ", q0.qstate == q3.qstate)
print("q1 == q2: ", q1.qstate == q2.qstate)
print("q1 == q3: ", q1.qstate == q3.qstate)
print("q2 == q3: ", q2.qstate == q3.qstate)

swapping(node_b)
print("SWAPPING")
print("q0 == q1: ", q0.qstate == q1.qstate)
print("q0 == q2: ", q0.qstate == q2.qstate)
print("q0 == q3: ", q0.qstate == q3.qstate)
print("q1 == q2: ", q1.qstate == q2.qstate)
print("q1 == q3: ", q1.qstate == q3.qstate)
print("q2 == q3: ", q2.qstate == q3.qstate)