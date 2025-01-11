import netsquid as ns
from netsquid.components import QuantumChannel, QSource, QuantumProcessor, QuantumMemory, SourceStatus, FibreDelayModel
from netsquid.components.qprocessor import PhysicalInstruction
from netsquid.components.models.qerrormodels import DepolarNoiseModel, FibreLossModel
from netsquid.qubits import StateSampler, ketstates, qubitapi as qapi
from netsquid.nodes import Node, Network, Connection
from netsquid.protocols import NodeProtocol
from netsquid.components.instructions import INSTR_MEASURE
import numpy as np

# Parameters
alpha = 0.2  # Attenuation coefficient (per km)
L = 10  # Length of the quantum channel (km)
P_source = 0.9  # Example success probability of the entanglement source
threshold_fidelity = 0.9  # Threshold for successful entanglement

# Create a depolarizing noise model to simulate the channel noise
depolar_rate = alpha * L
noise_model = DepolarNoiseModel(depolar_rate=depolar_rate)
loss_model = FibreLossModel(p_loss_init=0.1, p_loss_length=0.2)

# Define a source that generates entangled pairs
ent_source = QSource(
    "EntSource",
    state_sampler=StateSampler([ketstates.b00], [1.0]),
    num_ports=2,
    timing_model=None,
    tatus=SourceStatus.INTERNAL
)

# Create quantum channels with noise and loss models
qchannel_c2a = QuantumChannel(
    "qchannel_C2A", length=L/2, models={"quantum_loss": loss_model, "delay_model": FibreDelayModel()}
)
qchannel_c2b = QuantumChannel(
    "qchannel_C2B", length=L/2, models={"quantum_loss": loss_model, "delay_model": FibreDelayModel()}
)


# Define a quantum processor for nodes
def create_processor():
    memory_noise_model = DepolarNoiseModel(depolar_rate=0.01)  # Small depolarization for memory
    processor = QuantumProcessor(
        "processor", num_positions=1,
        mem_noise_models=[memory_noise_model]
    )
    return processor


# Create network nodes
node_a = Node("Alice", qmemory=create_processor())
node_b = Node("Bob", qmemory=create_processor())

# Create a network
network = Network("QuantumNetwork")
network.add_nodes([node_a, node_b])


# Create an entangling connection
class MyEntanglingConnection(Connection):
    def __init__(self):
        super().__init__(name="EntanglingConnection")
        self.add_subcomponent(ent_source, name="ent_source")

        # Add channels and forward quantum channel output to external port output:
        self.add_subcomponent(qchannel_c2a, forward_output=[("A", "recv")])
        self.add_subcomponent(qchannel_c2b, forward_output=[("B", "recv")])

        # Connect ent_source output to quantum channel input:
        ent_source.ports["qout0"].connect(qchannel_c2a.ports["send"])
        ent_source.ports["qout1"].connect(qchannel_c2b.ports["send"])


# Add the entangling connection between nodes
ent_connection = MyEntanglingConnection()
network.add_connection(
    node_a, node_b, connection=ent_connection, label="quantum",
    port_name_node1="qin_charlie", port_name_node2="qin_charlie"
)



def network_setup(node_distance=4e-3, depolar_rate=1e7, dephase_rate=0.2):
    # Setup nodes Alice and Bob with quantum processor:
    alice = Node("Alice", qmemory=create_processor(depolar_rate, dephase_rate))
    bob = Node("Bob", qmemory=create_processor(depolar_rate, dephase_rate))
    # Create a network
    network = Network("Teleportation_network")
    network.add_nodes([alice, bob])
    # Setup classical connection between nodes:
    c_conn = ClassicalConnection(length=node_distance)
    network.add_connection(alice, bob, connection=c_conn, label="classical",
                           port_name_node1="cout_bob", port_name_node2="cin_alice")
    # Setup entangling connection between nodes:
    source_frequency = 4e4 / node_distance
    q_conn = EntanglingConnection(
        length=node_distance, source_frequency=source_frequency)
    port_ac, port_bc = network.add_connection(
        alice, bob, connection=q_conn, label="quantum",
        port_name_node1="qin_charlie", port_name_node2="qin_charlie")
    alice.ports[port_ac].forward_input(alice.qmemory.ports['qin1'])
    bob.ports[port_bc].forward_input(bob.qmemory.ports['qin0'])
    return network


# Define a protocol for conditional entanglement generation
class ConditionalEntanglementProtocol(NodeProtocol):
    def __init__(self, node, partner, threshold_fidelity):
        super().__init__(node)
        self.partner = partner
        self.threshold_fidelity = threshold_fidelity

    def run(self):
        while True:
            print(self.node.ports.keys(), "!!!")
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

# Run the simulation
ns.sim_run(duration=1000)

# Collect results
success_count = protocol_a.num_successful_swaps + protocol_b.num_successful_swaps
total_trials = protocol_a.num_total_swaps + protocol_b.num_total_swaps

# Calculate and print the empirical success rate
empirical_success_rate = success_count / total_trials if total_trials > 0 else 0
print(f"Empirical entanglement generation success rate: {empirical_success_rate:.4f}")
