from netsquid.nodes import Network, Node
from netsquid.components import QuantumProcessor, QuantumChannel, ClassicalChannel
from netsquid.components.instructions import INSTR_INIT, INSTR_EMIT
from netsquid.nodes.connections import DirectConnection
from netsquid.protocols import NodeProtocol, Protocol
from netsquid.components.qprogram import QuantumProgram
from netsquid.util.simtools import sim_reset, sim_run
import netsquid as ns

class HeraldedConnection(DirectConnection):
    def __init__(self, name, length_to_a, length_to_b, time_window=0):
        super().__init__(name=name)
        delay_a = length_to_a / 200000 * 1e9  # in ns
        delay_b = length_to_b / 200000 * 1e9  # in ns
        channel_a = QuantumChannel("QChannel_A", delay=delay_a)
        channel_b = QuantumChannel("QChannel_B", delay=delay_b)
        self.add_subcomponent(channel_a, forward_output=[("A", "recv")])
        self.add_subcomponent(channel_b, forward_output=[("B", "recv")])
        self.ports['A'].forward_input(channel_a.ports['send'])
        self.ports['B'].forward_input(channel_b.ports['send'])

def create_example_network_with_intermediates(num_qubits=3):
    network = Network("SimpleLinkNetwork")
    alice = Node("Alice", qmemory=QuantumProcessor("qmem_Alice", num_positions=num_qubits))
    bob = Node("Bob", qmemory=QuantumProcessor("qmem_Bob", num_positions=num_qubits))
    charlie = Node("Charlie", qmemory=QuantumProcessor("qmem_Charlie", num_positions=num_qubits))

    network.add_nodes([alice, bob, charlie])

    distance = 2  # km
    conn1 = HeraldedConnection("HeraldedConnection1", length_to_a=distance / 2, length_to_b=distance / 2)
    conn2 = HeraldedConnection("HeraldedConnection2", length_to_a=distance / 2, length_to_b=distance / 2)

    network.add_connection(alice, charlie, connection=conn1, label='quantum1')
    network.add_connection(charlie, bob, connection=conn2, label='quantum2')

    classical_conn1 = DirectConnection("ClassicalConnection1", channel_AtoB=ClassicalChannel("classical_channel_1", length=distance))
    classical_conn2 = DirectConnection("ClassicalConnection2", channel_AtoB=ClassicalChannel("classical_channel_2", length=distance))

    network.add_connection(alice, charlie, connection=classical_conn1, label='classical1')
    network.add_connection(charlie, bob, connection=classical_conn2, label='classical2')

    return network

class MidpointHeraldingProtocol(NodeProtocol):
    def __init__(self, node, time_step, q_port_name):
        super().__init__(node=node)
        self.time_step = time_step
        self.q_port_name = q_port_name
        self.add_signal("PHOTONOUTCOME")

    class EmitProgram(QuantumProgram):
        def __init__(self):
            super().__init__(num_qubits=2)

        def program(self):
            q1, q2 = self.get_qubit_indices(2)
            self.apply(INSTR_INIT, q1)
            self.apply(INSTR_EMIT, [q1, q2])
            yield self.run()

    def run(self):
        while True:
            yield self.await_timer(self.time_step)
            prog = self.EmitProgram()
            self.node.qmemory.execute_program(prog)
            q_port = self.node.ports[self.q_port_name]
            yield self.await_port_input(q_port)
            message = q_port.rx_input()
            if message.meta.get("header") == 'photonoutcome':
                outcome = message.items[0]
                self.send_signal("PHOTONOUTCOME", outcome)

def setup_protocol_for_alice_charlie(network):
    alice, charlie = network.get_node("Alice"), network.get_node("Charlie")
    q_port_ac = network.get_connected_ports("Alice", "Charlie", label="quantum1")[0]

    alice_protocol = MidpointHeraldingProtocol(alice, time_step=1000, q_port_name=q_port_ac)
    charlie_protocol = MidpointHeraldingProtocol(charlie, time_step=1000, q_port_name=q_port_ac)

    return alice_protocol, charlie_protocol

def setup_protocol_for_charlie_bob(network):
    charlie, bob = network.get_node("Charlie"), network.get_node("Bob")
    q_port_cb = network.get_connected_ports("Charlie", "Bob", label="quantum2")[0]

    charlie_protocol = MidpointHeraldingProtocol(charlie, time_step=1000, q_port_name=q_port_cb)
    bob_protocol = MidpointHeraldingProtocol(bob, time_step=1000, q_port_name=q_port_cb)

    return charlie_protocol, bob_protocol

class EntanglementSwappingProtocol(Protocol):
    def __init__(self, name, alice_charlie_protocols, charlie_bob_protocols):
        super().__init__(name=name)
        self.alice_protocol, self.charlie_protocol_ac = alice_charlie_protocols
        self.charlie_protocol_cb, self.bob_protocol = charlie_bob_protocols

    def run(self):
        self.alice_protocol.start()
        self.charlie_protocol_ac.start()
        self.charlie_protocol_cb.start()
        self.bob_protocol.start()

        while True:
            outcome_ac = yield self.await_signal(self.alice_protocol, "PHOTONOUTCOME")
            outcome_cb = yield self.await_signal(self.charlie_protocol_cb, "PHOTONOUTCOME")

            if outcome_ac == 1 and outcome_cb == 1:
                self.send_signal("ENTANGLEMENT_SUCCESS", "Alice-Bob")
                break

def setup_swapping_protocol(network):
    alice_charlie_protocols = setup_protocol_for_alice_charlie(network)
    charlie_bob_protocols = setup_protocol_for_charlie_bob(network)
    return EntanglementSwappingProtocol("EntanglementSwappingProtocol", alice_charlie_protocols, charlie_bob_protocols)

def run_entanglement_protocols():
    sim_reset()
    ns.set_random_state(42)
    ns.set_qstate_formalism(ns.QFormalism.DM)
    network = create_example_network_with_intermediates()

    # Setup protocols
    alice_charlie_protocols = setup_protocol_for_alice_charlie(network)
    charlie_bob_protocols = setup_protocol_for_charlie_bob(network)
    swapping_protocol = setup_swapping_protocol(network)

    # Start protocols
    for protocol in alice_charlie_protocols:
        protocol.start()
    for protocol in charlie_bob_protocols:
        protocol.start()
    swapping_protocol.start()

    sim_run()

    # Verification
    alice = network.get_node("Alice")
    bob = network.get_node("Bob")
    charlie = network.get_node("Charlie")

    print("Verification of entanglement:")
    for node, name in zip([alice, charlie, bob], ["Alice", "Charlie", "Bob"]):
        qubits = node.qmemory.peek(0)[0].qstate.qubits
        print(f"{name} qubits: {qubits}")

run_entanglement_protocols()