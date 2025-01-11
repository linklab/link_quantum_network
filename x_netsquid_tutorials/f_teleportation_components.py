from netsquid import pydynaa
import netsquid as ns
from netsquid.components import QuantumMemory, Port, DepolarNoiseModel, QuantumChannel, FibreDelayModel


class Alice(pydynaa.Entity):
    def __init__(self, teleport_state, cchannel_send_port):
        self.teleport_state = teleport_state
        self.cchannel_send_port = cchannel_send_port
        self.qmemory = QuantumMemory(name="AliceMemory", num_positions=2)
        self._wait(
            handler=pydynaa.EventHandler(self._handle_input_qubit),
            entity=self.qmemory.ports["qin1"], event_type=Port.evtype_input
        )
        self.qmemory.ports["qin1"].notify_all_input = True

    def _handle_input_qubit(self, event):
        # Callback function that does teleportation and schedules a corrections ready event
        q0, = ns.qubits.create_qubits(num_qubits=1, no_state=True)
        ns.qubits.assign_qstate(qubits=[q0], qrepr=self.teleport_state)
        self.qmemory.put(qubits=[q0], positions=[0])
        self.qmemory.operate(operator=ns.CNOT, positions=[0, 1])
        self.qmemory.operate(operator=ns.H, positions=[0])
        m0, m1 = self.qmemory.measure(positions=[0, 1], observable=ns.Z, discard=True)[0]
        self.cchannel_send_port.tx_input([m0, m1])
        print(f"{ns.sim_time():.1f}: Alice received entangled qubit, "
              f"measured qubits & sending corrections")


class Bob(pydynaa.Entity):
    depolar_rate = 1e7  # depolarization rate of waiting qubits [Hz]

    def __init__(self, cchannel_recv_port):
        noise_model = DepolarNoiseModel(depolar_rate=self.depolar_rate)
        self.qmemory = QuantumMemory(name="BobMemory", num_positions=1, memory_noise_models=[noise_model])
        cchannel_recv_port.bind_output_handler(self._handle_corrections)

    def _handle_corrections(self, message):
        # Callback function that handles messages from both Alice and Charlie
        m0, m1 = message.items
        if m1:
            self.qmemory.operate(ns.X, positions=[0])
        if m0:
            self.qmemory.operate(ns.Z, positions=[0])
        qubit = self.qmemory.pop(positions=[0])
        fidelity = ns.qubits.fidelity(qubit, ns.y0, squared=True)
        print(f"{ns.sim_time():.1f}: Bob received entangled qubit and corrections!"
              f" Fidelity = {fidelity:.3f}")


def setup_network(alice, bob, qsource, length=4e-3):
    from netsquid.components.models.delaymodels import FibreDelayModel
    qchannel_c2a = QuantumChannel(
        name="Charlie->Alice", length=length / 2, models={"delay_model": FibreDelayModel()}
    )
    qchannel_c2b = QuantumChannel(
        name="Charlie->Bob", length=length / 2, models={"delay_model": FibreDelayModel()}
    )
    qsource.ports['qout0'].connect(qchannel_c2a.ports['send'])
    qsource.ports['qout1'].connect(qchannel_c2b.ports['send'])
    alice.qmemory.ports['qin1'].connect(qchannel_c2a.ports['recv'])
    bob.qmemory.ports['qin0'].connect(qchannel_c2b.ports['recv'])


def main():
    from netsquid.components import ClassicalChannel
    cchannel = ClassicalChannel(
        name="CChannel", length=4e-3, models={"delay_model": FibreDelayModel()}
    )
    alice = Alice(teleport_state=ns.y0, cchannel_send_port=cchannel.ports["send"])
    bob = Bob(cchannel_recv_port=cchannel.ports["recv"])

    from netsquid.qubits.state_sampler import StateSampler
    import netsquid.qubits.ketstates as ks
    state_sampler = StateSampler([ks.b00], [1.0])

    from netsquid.components.qsource import QSource, SourceStatus
    from netsquid.components.models.delaymodels import FixedDelayModel
    charlie_source = QSource(
        name="Charlie", state_sampler=state_sampler, frequency=100, num_ports=2,
        timing_model=FixedDelayModel(delay=50), status=SourceStatus.INTERNAL
    ) # the internal clock with a delay of 50 ns (frequency of 20 GHz).

    setup_network(alice, bob, charlie_source)

    stats = ns.sim_run(end_time=100)

    print(stats)


if __name__ == "__main__":
    main()