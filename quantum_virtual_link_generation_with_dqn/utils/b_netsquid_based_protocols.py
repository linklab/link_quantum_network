import netsquid as ns
from netsquid.qubits import ketstates as ks
from netsquid.protocols.protocol import Protocol
from netsquid.protocols.nodeprotocols import NodeProtocol


class EntanglementProtocol(Protocol):
    def __init__(self, node, channel, position):
        self.node = node
        self.channel = channel
        self.position = position

    def run(self):
        qubit_0, qubit_1 = ns.qubits.create_qubits(2)
        ns.qubits.operate(qubit_0, ns.H)
        ns.qubits.operate([qubit_0, qubit_1], ns.CNOT)

        self.node.qmemory.put([qubit_0], positions=[self.position])
        self.channel.send(qubit_1)


class BellMeasurementProtocol(NodeProtocol):
    def run(self):
        qubit_0 = self.node.qmemory.peek(positions=[0])
        qubit_1 = self.node.qmemory.peek(positions=[1])
        ns.qubits.operate(qubit_0[0], ns.H)
        ns.qubits.operate([qubit_0[0], qubit_1[0]], ns.CNOT)
