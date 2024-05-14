from netsquid.protocols import NodeProtocol
from netsquid.components import QuantumChannel
from netsquid.nodes import Node, DirectConnection
from netsquid.qubits import qubitapi as qapi
import netsquid as ns

class PingProtocol(NodeProtocol):
    def run(self):
        print(f"Starting ping at t={ns.sim_time()}")
        port = self.node.ports["port_to_channel"]
        qubit, = qapi.create_qubits(1)
        port.tx_output(qubit)  # Send qubit to Pong
        while True:
            # Wait for qubit to be received back
            yield self.await_port_input(port)
            qubit = port.rx_input().items[0]
            m, prob = qapi.measure(qubit, ns.Z)
            labels_z = ("|0>", "|1>")
            print(f"{ns.sim_time()}: Pong event! {self.node.name} measured "
                  f"{labels_z[m]} with probability {prob:.2f}")
            port.tx_output(qubit)  # Send qubit to B


class PongProtocol(NodeProtocol):
    def run(self):
        print("Starting pong at t={}".format(ns.sim_time()))
        port = self.node.ports["port_to_channel"]
        while True:
            yield self.await_port_input(port)
            qubit = port.rx_input().items[0]
            m, prob = qapi.measure(qubit, ns.X)
            labels_x = ("|+>", "|->")
            print(f"{ns.sim_time()}: Ping event! {self.node.name} measured "
                  f"{labels_x[m]} with probability {prob:.2f}")
            port.tx_output(qubit)  # send qubit to Ping


def main():
    ns.sim_reset()
    ns.set_random_state(seed=42)
    node_ping = Node(name="Ping", port_names=["port_to_channel"])
    node_pong = Node(name="Pong", port_names=["port_to_channel"])
    connection = DirectConnection(
        name="Connection",
        channel_AtoB=QuantumChannel("Channel_LR", delay=10),
        channel_BtoA=QuantumChannel("Channel_RL", delay=10)
    )
    node_ping.ports["port_to_channel"].connect(connection.ports["A"])
    node_pong.ports["port_to_channel"].connect(connection.ports["B"])
    ping_protocol = PingProtocol(node_ping)
    pong_protocol = PongProtocol(node_pong)

    ping_protocol.start()
    pong_protocol.start()
    stats = ns.sim_run(91)

    print(stats)


if __name__ == "__main__":
    main()