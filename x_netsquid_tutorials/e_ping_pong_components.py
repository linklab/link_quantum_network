from netsquid import pydynaa
from netsquid.components import QuantumMemory, QuantumChannel, FibreDelayModel
from netsquid.components.component import Port
import netsquid as ns

class PingEntity(pydynaa.Entity):
    length = 2e-3  # channel length [km]: 2e-3 km == 2m

    def __init__(self):
        # Create a memory and a quantum channel:
        self.qmemory = QuantumMemory(name="PingMemory", num_positions=1)
        self.qchannel = QuantumChannel(
            name="PingChannel", length=PingEntity.length,
            models={"delay_model": FibreDelayModel(c=2e5)}
        )
        # link output from qmemory (pop) to input of ping channel:
        self.qmemory.ports["qout"].connect(self.qchannel.ports["send"])

        # Setup callback function to handle input on quantum memory port "qin0":
        self._wait(
            handler=pydynaa.EventHandler(self._handle_input_qubit),
            entity=self.qmemory.ports["qin0"], event_type=Port.evtype_input
        )
        self.qmemory.ports["qin0"].notify_all_input = True

    def start(self, qubit):
        # Start the game by having ping player send the first qubit (ping)
        self.qchannel.send(qubit)

    def wait_for_pong(self, other_entity):
        # Setup this entity to pass incoming qubits to its quantum memory
        self.qmemory.ports["qin0"].connect(other_entity.qchannel.ports["recv"])

    def _handle_input_qubit(self, event):
        # Callback function called by the pong handler when pong event is triggered
        [m], [prob] = self.qmemory.measure(positions=[0], observable=ns.Z)
        labels_z = ("|0>", "|1>")
        print(f"{ns.sim_time():.1f}: Pong event! PingEntity measured "
              f"{labels_z[m]} with probability {prob:.2f}")
        self.qmemory.pop(positions=[0])


class PongEntity(pydynaa.Entity):
    length = 2e-3  # channel length [km]

    def __init__(self):
        # Create a memory and a quantum channel:
        self.qmemory = QuantumMemory(name="PongMemory", num_positions=1)
        self.qchannel = QuantumChannel(
            name="PingChannel", length=self.length,
            models={"delay_model": FibreDelayModel()}
        )
        # link output from qmemory (pop) to input of ping channel:
        self.qmemory.ports["qout"].connect(self.qchannel.ports["send"])

        # Setup callback function to handle input on quantum memory:
        self._wait(
            handler=pydynaa.EventHandler(self._handle_input_qubit),
            entity=self.qmemory.ports["qin0"], event_type=Port.evtype_input
        )
        self.qmemory.ports["qin0"].notify_all_input = True

    def wait_for_ping(self, other_entity):
        # Setup this entity to pass incoming qubits to its quantum memory
        self.qmemory.ports["qin0"].connect(other_entity.qchannel.ports["recv"])

    def _handle_input_qubit(self, event):
        # Callback function called by the pong handler when pong event is triggered
        [m], [prob] = self.qmemory.measure(positions=[0], observable=ns.X)
        labels_x = ("|+>", "|->")
        print(f"{ns.sim_time():.1f}: Ping event! PongEntity measured "
              f"{labels_x[m]} with probability {prob:.2f}")
        self.qmemory.pop(positions=[0])


def main():
    # Create entities and register them to each other
    ns.sim_reset()
    ping = PingEntity()
    pong = PongEntity()
    ping.wait_for_pong(pong)
    pong.wait_for_ping(ping)

    # Create a qubit and instruct the ping entity to start
    qubit, = ns.qubits.create_qubits(1)
    ping.start(qubit)

    ns.set_random_state(seed=42)
    stats = ns.sim_run(91)


if __name__ == "__main__":
    main()