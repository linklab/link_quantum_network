import netsquid as ns

from netsquid.protocols import NodeProtocol, Signals
from netsquid.qubits import qubitapi as qapi


class InitStateProtocol(NodeProtocol):
    def run(self):
        qubit, = qapi.create_qubits(1)
        mem_pos = self.node.qmemory.unused_positions[0]
        self.node.qmemory.put(qubit, mem_pos)
        self.node.qmemory.operate(ns.H, mem_pos)
        self.node.qmemory.operate(ns.S, mem_pos)
        self.send_signal(signal_label=Signals.SUCCESS, result=mem_pos)


class BellMeasurementProtocol(NodeProtocol):
    def __init__(self, node, qubit_protocol):
        super().__init__(node)
        self.add_subprotocol(qubit_protocol, 'qprotocol')

    def run(self):
        qubit_initialised = False
        entanglement_ready = False
        while True:
            evexpr_signal = self.await_signal(
                sender=self.subprotocols['qprotocol'],
                signal_label=Signals.SUCCESS
            )
            evexpr_port = self.await_port_input(self.node.ports["qin_charlie"])
            expression = yield evexpr_signal | evexpr_port
            if expression.first_term.value:
                 # First expression was triggered
                qubit_initialised = True
            else:
                # Second expression was triggered
                entanglement_ready = True
            if qubit_initialised and entanglement_ready:
                # Perform Bell measurement:
                self.node.qmemory.operate(ns.CNOT, [0, 1])
                self.node.qmemory.operate(ns.H, 0)
                m, _ = self.node.qmemory.measure([0, 1])
                # Send measurement results to Bob:
                self.node.ports["cout_bob"].tx_output(m)
                self.send_signal(Signals.SUCCESS)
                print(f"{ns.sim_time():.1f}: Alice received entangled qubit, "
                      f"measured qubits & sending corrections")
                break

    def start(self):
        super().start()
        self.start_subprotocols()


class CorrectionProtocol(NodeProtocol):

    def __init__(self, node):
        super().__init__(node)

    def run(self):
        port_alice = self.node.ports["cin_alice"]
        port_charlie = self.node.ports["qin_charlie"]
        entanglement_ready = False
        meas_results = None
        while True:
            evexpr_port_a = self.await_port_input(port_alice)
            evexpr_port_c = self.await_port_input(port_charlie)
            expression = yield evexpr_port_a | evexpr_port_c
            if expression.first_term.value:
                meas_results = port_alice.rx_input().items
            else:
                entanglement_ready = True
            if meas_results is not None and entanglement_ready:
                if meas_results[0]:
                    self.node.qmemory.operate(ns.Z, 0)
                if meas_results[1]:
                    self.node.qmemory.operate(ns.X, 0)
                self.send_signal(Signals.SUCCESS, 0)
                fidelity = ns.qubits.fidelity(self.node.qmemory.peek(0)[0], ns.y0, squared=True)
                print(f"{ns.sim_time():.1f}: Bob received entangled qubit and "
                      f"corrections! Fidelity = {fidelity:.3f}")
                break

def main():
    from netsquid.examples.teleportation import example_network_setup
    ns.sim_reset()
    ns.set_qstate_formalism(ns.QFormalism.DM)
    ns.set_random_state(seed=42)
    network = example_network_setup()
    alice = network.get_node("Alice")
    bob = network.get_node("Bob")
    random_state_protocol = InitStateProtocol(alice)
    bell_measure_protocol = BellMeasurementProtocol(alice, random_state_protocol)
    correction_protocol = CorrectionProtocol(bob)
    bell_measure_protocol.start()
    correction_protocol.start()
    stats = ns.sim_run(100)

    print(stats)


if __name__ == "__main__":
    main()