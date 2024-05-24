import netsquid as ns
import pydynaa
from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.protocols.protocol import Signals
from netsquid.components.instructions import INSTR_SWAP, INSTR_MEASURE_BELL, INSTR_X, INSTR_Z
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor, QuantumProgram
from netsquid.qubits import ketstates as ks
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.models.delaymodels import FixedDelayModel
from netsquid.components.qchannel import QuantumChannel
from netsquid.nodes.network import Network
from netsquid.components import Message
from netsquid.util.datacollector import DataCollector
from pydynaa import EventExpression
from netsquid.examples.teleportation import EntanglingConnection, ClassicalConnection
from netsquid.examples.repeater_chain import FibreDepolarizeModel

# Parameter
entanglement_probability = 1.0


class EntangleNodes(NodeProtocol):
    def __init__(self, node, role, start_expression=None, input_mem_pos=0, num_pairs=1, name=None):
        if role.lower() not in ["source", "receiver"]:
            raise ValueError
        self._is_source = role.lower() == "source"
        name = name if name else "EntangleNode({}, {})".format(node.name, role)
        super().__init__(node=node, name=name)
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))
        self.start_expression = start_expression
        self._num_pairs = num_pairs
        self._mem_positions = None
        # Claim input memory position:
        if self.node.qmemory is None:
            raise ValueError("Node {} does not have a quantum memory assigned.".format(self.node))
        self._input_mem_pos = input_mem_pos
        self._qmem_input_port = self.node.qmemory.ports["qin{}".format(self._input_mem_pos)]
        self.node.qmemory.mem_positions[self._input_mem_pos].in_use = True

    def start(self):
        self.entangled_pairs = 0  # counter
        self._mem_positions = [self._input_mem_pos]
        # Claim extra memory positions to use (if any):
        extra_memory = self._num_pairs - 1
        if extra_memory > 0:
            unused_positions = self.node.qmemory.unused_positions
            if extra_memory > len(unused_positions):
                raise RuntimeError("Not enough unused memory positions available: need {}, have {}"
                                   .format(self._num_pairs - 1, len(unused_positions)))
            for i in unused_positions[:extra_memory]:
                self._mem_positions.append(i)
                self.node.qmemory.mem_positions[i].in_use = True
        # Call parent start method
        return super().start()

    def stop(self):
        # Unclaim used memory positions:
        if self._mem_positions:
            for i in self._mem_positions[1:]:
                self.node.qmemory.mem_positions[i].in_use = False
            self._mem_positions = None
        # Call parent stop method
        super().stop()

    def run(self):
        while True:
            if self.start_expression is not None:
                yield self.start_expression
            elif self._is_source and self.entangled_pairs >= self._num_pairs:
                # If no start expression specified then limit generation to one round
                break
            for mem_pos in self._mem_positions[::-1]:
                # Iterate in reverse so that input_mem_pos is handled last
                if self._is_source:
                    self.node.subcomponents[self._qsource_name].trigger()
                yield self.await_port_input(self._qmem_input_port)
                if mem_pos != self._input_mem_pos:
                    self.node.qmemory.execute_instruction(
                        INSTR_SWAP, [self._input_mem_pos, mem_pos])
                    if self.node.qmemory.busy:
                        yield self.await_program(self.node.qmemory)
                self.entangled_pairs += 1
                self.send_signal(Signals.SUCCESS, mem_pos)

    @property
    def is_connected(self):
        if not super().is_connected:
            return False
        if self.node.qmemory is None:
            return False
        if self._mem_positions is None and len(self.node.qmemory.unused_positions) < self._num_pairs - 1:
            return False
        if self._mem_positions is not None and len(self._mem_positions) != self._num_pairs:
            return False
        if self._is_source:
            for name, subcomp in self.node.subcomponents.items():
                if isinstance(subcomp, QSource):
                    self._qsource_name = name
                    break
            else:
                return False
        return True


class SwapProtocol(NodeProtocol):
    def __init__(self, node, name):
        super().__init__(node, name)
        self._qmem_input_port_l = self.node.qmemory.ports["qin1"]
        self._qmem_input_port_r = self.node.qmemory.ports["qin0"]
        self._program = QuantumProgram(num_qubits=2)
        q1, q2 = self._program.get_qubit_indices(num_qubits=2)
        self._program.apply(INSTR_MEASURE_BELL, [q1, q2], output_key="m", inplace=False)

    def run(self):
        while True:
            yield (self.await_port_input(self._qmem_input_port_l) &
                   self.await_port_input(self._qmem_input_port_r))
            # Perform Bell measurement
            yield self.node.qmemory.execute_program(self._program, qubit_mapping=[1, 0])
            m, = self._program.output["m"]
            # Send result to right node on end
            self.node.ports["ccon_R"].tx_output(Message(m))


class SwapCorrectProgram(QuantumProgram):
    """Quantum processor program that applies all swap corrections."""
    default_num_qubits = 1

    def set_corrections(self, x_corr, z_corr):
        self.x_corr = x_corr % 2
        self.z_corr = z_corr % 2

    def program(self):
        q1, = self.get_qubit_indices(1)
        if self.x_corr == 1:
            self.apply(INSTR_X, q1)
        if self.z_corr == 1:
            self.apply(INSTR_Z, q1)
        yield self.run()


class CorrectProtocol(NodeProtocol):
    def __init__(self, node, num_nodes):
        super().__init__(node, "CorrectProtocol")
        self.num_nodes = num_nodes
        self._x_corr = 0
        self._z_corr = 0
        self._program = SwapCorrectProgram()
        self._counter = 0

    def run(self):
        while True:
            yield self.await_port_input(self.node.ports["ccon_L"])
            message = self.node.ports["ccon_L"].rx_input()
            if message is None or len(message.items) != 1:
                continue
            m = message.items[0]
            if m == ks.BellIndex.B01 or m == ks.BellIndex.B11:
                self._x_corr += 1
            if m == ks.BellIndex.B10 or m == ks.BellIndex.B11:
                self._z_corr += 1
            self._counter += 1
            if self._counter == self.num_nodes - 2:
                if self._x_corr or self._z_corr:
                    self._program.set_corrections(self._x_corr, self._z_corr)
                    yield self.node.qmemory.execute_program(self._program, qubit_mapping=[1])
                self.send_signal(Signals.SUCCESS)
                self._x_corr = 0
                self._z_corr = 0
                self._counter = 0

def create_chain(network, node_left, node_right, node_distance, source_frequency):
    # node_distance : float, optional
    # Distance between nodes, larger than 0. Default 20 [km].
    # Create quantum connection
    qconn = EntanglingConnection(name=f"qconn_{node_left.ID}-{node_right.ID}", length=node_distance,
                                 source_frequency=source_frequency)
    for channel_name in ['qchannel_C2A', 'qchannel_C2B']:
        qconn.subcomponents[channel_name].models['quantum_noise_model'] = \
            FibreDepolarizeModel()
    port_l_name, port_r_name = network.add_connection(
        node_left, node_right, connection=qconn, label="quantum")
    # Forward qconn directly to quantum memories for right and left inputs:
    node_left.ports[port_l_name].forward_input(node_left.qmemory.ports["qin0"])  # R input
    node_right.ports[port_r_name].forward_input(
        node_right.qmemory.ports["qin1"])  # L input
    # Create classical connection
    cconn = ClassicalConnection(name=f"cconn_{node_left.ID}-{node_right.ID}", length=node_distance)
    port_l_name, port_r_name = network.add_connection(
        node_left, node_right, connection=cconn, label="classical",
        port_name_node1="ccon_R", port_name_node2="ccon_L")
    # Forward cconn to right most node
    if "ccon_L" in node_left.ports:
        node_left.ports["ccon_L"].bind_input_handler(
            lambda message, _node=node_left: _node.ports["ccon_R"].tx_output(message))

def example_network_setup(prep_delay=5, qchannel_delay=100, num_mem_positions=3):
    # Setup nodes:
    network = Network("Entangle_nodes")
    node_a, node_b, node_c = network.add_nodes(["node_A", "node_B", "node_C"])
    node_a.add_subcomponent(QuantumProcessor(
        "QuantumMemoryATest", num_mem_positions, fallback_to_nonphysical=True))
    node_b.add_subcomponent(QuantumProcessor(
        "QuantumMemoryBTest", num_mem_positions, fallback_to_nonphysical=True))
    node_c.add_subcomponent(QuantumProcessor(
        "QuantumMemoryCTest", num_mem_positions, fallback_to_nonphysical=True))
    node_a.add_subcomponent(
        QSource("QSourceTestA", state_sampler=StateSampler(
            [ks.b00, None], [entanglement_probability, 1 - entanglement_probability]
        ), num_ports=2, status=SourceStatus.EXTERNAL,
                models={"emission_delay_model": FixedDelayModel(delay=prep_delay)}))
    node_b.add_subcomponent(
        QSource("QSourceTestB", state_sampler=StateSampler(
            [ks.b00, None], [entanglement_probability, 1 - entanglement_probability]
        ), num_ports=2, status=SourceStatus.EXTERNAL,
                models={"emission_delay_model": FixedDelayModel(delay=prep_delay)}))
    # Create and connect quantum channels:
    qchannel_ab = QuantumChannel("QuantumChannelTestAB", delay=qchannel_delay)
    qchannel_bc = QuantumChannel("QuantumChannelTestBC", delay=qchannel_delay)
    port_name_a, port_name_b = network.add_connection(
        node_a, node_b, channel_to=qchannel_ab, label="quantum")
    port_name_b, port_name_c = network.add_connection(
        node_b, node_c, channel_to=qchannel_bc, label="quantum")
    # Setup Node A ports:
    node_a.subcomponents["QSourceTestA"].ports["qout0"].forward_output(
        node_a.ports[port_name_a])
    node_a.subcomponents["QSourceTestA"].ports["qout1"].connect(
        node_a.qmemory.ports["qin0"])
    # Setup Node B ports:
    node_b.subcomponents["QSourceTestB"].ports["qout0"].forward_output(
        node_b.ports[port_name_b])
    node_b.subcomponents["QSourceTestB"].ports["qout1"].connect(
        node_b.qmemory.ports["qin0"])
    # Setup Node C ports:
    node_c.ports[port_name_c].forward_input(node_c.qmemory.ports["qin0"])

    create_chain(network, node_a, node_b, node_distance=1, source_frequency=1e9)
    create_chain(network, node_b, node_c, node_distance=1, source_frequency=1e9)

    return network

if __name__ == "__main__":
    network = example_network_setup()

    protocol_a = EntangleNodes(node=network.get_node("node_A"), role="source")
    protocol_b1 = EntangleNodes(node=network.get_node("node_B"), role="receiver")
    protocol_b2 = EntangleNodes(node=network.get_node("node_B"), role="source")
    protocol_c = EntangleNodes(node=network.get_node("node_C"), role="receiver")

    # Setup entanglement swapping protocol on node B
    swap_protocol = SwapProtocol(node=network.get_node("node_B"), name="Swap_B")

    # Setup correction protocol on node C
    correct_protocol = CorrectProtocol(node=network.get_node("node_C"), num_nodes=3)

    protocol_a.start()
    protocol_b1.start()

    protocol_b2.start()
    protocol_c.start()

    swap_protocol.start()
    correct_protocol.start()

    ns.sim_run(200)

    q1, = network.get_node("node_A").qmemory.peek(0)
    q2, = network.get_node("node_B").qmemory.peek(0)
    q3, = network.get_node("node_C").qmemory.peek(0)

    if q1 is not None and q2 is not None:
        print("Entanglement generation success (A-B)", end=" ")
        print("Fidelity: {}".format(ns.qubits.fidelity([q1, q2], ks.b00)))
    else:
        print("Entanglement generation fail (A-B)")

    if q2 is not None and q3 is not None:
        print("Entanglement generation success (B-C)", end=" ")
        print("Fidelity: {}".format(ns.qubits.fidelity([q2, q3], ks.b00)))
    else:
        print("Entanglement generation fail (B-C)")

    if q1 is not None and q3 is not None:
        print("Entanglement generation success (A-C)", end=" ")
        print("Fidelity: {}".format(ns.qubits.fidelity([q1, q3], ks.b00)))
    else:
        print("Entanglement generation fail (A-C)")
