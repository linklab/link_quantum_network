import netsquid as ns
from netsquid.components import QSource, QuantumChannel, QuantumMemory, SourceStatus
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.qubits import StateSampler, ketstates
from netsquid.nodes import Node, Network
from netsquid.protocols import NodeProtocol, Signals
import numpy as np

# Parameters
alpha = 0.2  # Attenuation coefficient (per km)
L = 10  # Length of the quantum channel (km)
cutoff_time = 100  # Maximum allowable time for an entangled state (in ns)

# Create a depolarizing noise model to simulate the channel noise
depolar_rate = alpha * L
noise_model = DepolarNoiseModel(depolar_rate=depolar_rate)

# Define a source that generates entangled pairs
state_sampler = StateSampler([ketstates.b00], [1.0])
ent_source = QSource("EntangledSource",
                     state_sampler=state_sampler,
                     num_ports=2,
                     timing_model=None,
                     status=SourceStatus.INTERNAL)

# Create quantum channels with the noise model
qc = QuantumChannel("QuantumChannel", length=L, models={"quantum_noise": noise_model})

# Define a quantum processor for nodes
def create_processor():
    memory_noise_model = DepolarNoiseModel(depolar_rate=0.01)  # Small depolarization for memory
    processor = QuantumMemory("Memory", num_positions=1, memory_noise_models=[memory_noise_model])
    return processor

# Create network nodes
node_a = Node("Alice", qmemory=create_processor())
node_b = Node("Bob", qmemory=create_processor())

# Create a network
network = Network("QuantumNetwork")
network.add_nodes([node_a, node_b])

# Connect the nodes with quantum and classical channels
network.add_connection(node_a, node_b, channel=qc, label="quantum")

# Define a protocol for entanglement generation with cutoff time
class EntanglementGenerationProtocol(NodeProtocol):
    def __init__(self, node, source, partner, cutoff_time):
        super().__init__(node)
        self.source = source
        self.partner = partner
        self.cutoff_time = cutoff_time

    def run(self):
        while True:
            # Generate entangled qubits
            qubits = self.source.generate()
            q1, q2 = qubits.items
            self.node.qmemory.put(q1, positions=0)
            self.partner.qmemory.put(q2, positions=0)
            start_time = ns.sim_time()

            # Wait until cutoff time
            yield self.await_timer(self.cutoff_time)

            # Check if the qubits are still entangled
            if ns.sim_time() - start_time < self.cutoff_time:
                fidelity = ns.qubits.fidelity([self.node.qmemory.peek(0), self.partner.qmemory.peek(0)], ketstates.b00)
                success = fidelity > 0.9  # Threshold for considering successful entanglement
                if success:
                    self.send_signal(signal_label=Signals.SUCCESS)
                else:
                    self.send_signal(signal_label=Signals.FAILURE)
            else:
                self.node.qmemory.pop(positions=0)
                self.partner.qmemory.pop(positions=0)
                self.send_signal(signal_label=Signals.FAILURE)

# Instantiate and run the protocol
protocol = EntanglementGenerationProtocol(node_a, ent_source, node_b, cutoff_time=cutoff_time)
protocol.start()

# Run the simulation
ns.sim_run(duration=1000)

# Calculate the empirical success rate
success_events = protocol.num_successful_swaps
total_events = protocol.num_total_swaps
empirical_success_rate = success_events / total_events if total_events > 0 else 0
print(f"Empirical entanglement generation success rate: {empirical_success_rate:.4f}")
