import netsquid as ns

from netsquid.nodes import Node
from netsquid.components.models import DelayModel
from netsquid.components import QuantumChannel
from netsquid.nodes import DirectConnection
from netsquid.protocols import NodeProtocol


class PingPongDelayModel(DelayModel):
    def __init__(self, speed_of_light_fraction=0.5, standard_deviation=0.05):
        super().__init__()
        # (the speed of light is about 300,000 km/s)
        self.properties["speed"] = speed_of_light_fraction * 3e5
        self.properties["std"] = standard_deviation
        self.required_properties = ['length']  # in km
        

    def generate_delay(self, **kwargs):
        avg_speed = self.properties["speed"]
        std = self.properties["std"]
        # The 'rng' property contains a random number generator
        # We can use that to generate a random speed
        speed = self.properties["rng"].normal(avg_speed, avg_speed * std)
        delay = 1e9 * kwargs['length'] / speed  # in nanoseconds
        return delay


class PingPongProtocol(NodeProtocol):
    def __init__(self, node, observable, qubit=None):
        super().__init__(node)
        self.observable = observable
        self.qubit = qubit
        # Define matching pair of strings for pretty printing of basis states:
        self.basis = ["|0>", "|1>"] if observable == ns.Z else ["|+>", "|->"]
        self.num_events = 0
        self.num_measure = 0

    def run(self):
        if self.qubit is not None:
            # Send (TX) qubit to the other node via port's output:
            self.node.ports["qubitIO"].tx_output(self.qubit)
            self.num_events += 1
            print(f"Time: {ns.sim_time():5.1f}, #Events: {self.num_events}: {self.node.name} sent {self.qubit}")

        while True:
            # Wait (yield) until input has arrived at our port:
            yield self.await_port_input(self.node.ports["qubitIO"])
            # Receive (RX) qubit on the port's input:
            message = self.node.ports["qubitIO"].rx_input()
            self.num_events += 1
            rx_qubit = message.items[0]
            meas, prob = ns.qubits.measure(rx_qubit, observable=self.observable)
            self.num_measure += 1
            print(f"Time: {ns.sim_time():5.1f}, #Events: {self.num_events}: {self.node.name} measured "
                  f"{self.basis[meas]} with probability {prob:.2f}")
            # Send (TX) qubit to the other node via connection:
            self.node.ports["qubitIO"].tx_output(rx_qubit)
            self.num_events += 1
            print(f"Time: {ns.sim_time():5.1f}, #Events: {self.num_events}: {self.node.name} sent {rx_qubit}")


def main():
    node_ping = Node(name="Ping")
    node_pong = Node(name="Pong")

    distance = 2.74 / 1000  # default unit of length in channels is km
    delay_model = PingPongDelayModel()
    channel_1 = QuantumChannel(
        name="qchannel[ping to pong]", length=distance, models={"delay_model": delay_model}
    )
    channel_2 = QuantumChannel(
        name="qchannel[pong to ping]", length=distance, models={"delay_model": delay_model}
    )

    connection = DirectConnection(
        name="conn[ping|pong]", channel_AtoB=channel_1, channel_BtoA=channel_2
    )
    node_ping.connect_to(
        remote_node=node_pong, connection=connection, local_port_name="qubitIO", remote_port_name="qubitIO"
    )

    qubits = ns.qubits.create_qubits(num_qubits=1)
    ping_protocol = PingPongProtocol(node=node_ping, observable=ns.Z, qubit=qubits[0])
    pong_protocol = PingPongProtocol(node=node_pong, observable=ns.X)

    ping_protocol.start()
    pong_protocol.start()
    run_stats = ns.sim_run(duration=310)

    print(ping_protocol.num_measure)
    print(pong_protocol.num_measure)

    print(run_stats)


if __name__ == "__main__":
    main()