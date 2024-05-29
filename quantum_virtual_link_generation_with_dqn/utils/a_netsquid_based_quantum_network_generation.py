from pprint import pprint
from netsquid.nodes import Node
from netsquid.components import QuantumMemory
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.models import FixedDelayModel, FibreLossModel
from netsquid.components.models.qerrormodels import DepolarNoiseModel


class QuantumNetwork(object):
    def __init__(self, node_config: dict = None, quantum_channel_config: dict = None):
        super(QuantumNetwork, self).__init__()
        if node_config is None:
            node_config = {
                "names": ["node_A", "node_B", "node_C"],
                "port_names": [["qin_ba", ], None, ["qin_bc", ]],
                "port_forwardings": {"qin_ba": "qin0", None: None, "qin_bc": "qin0"},
                "quantum_memory_names": ["memory_A", "memory_B", "memory_C"],
                "quantum_memory_positions": [1, 2, 1],
                "depolar_rates": [500, 500, 500]
            }

        if quantum_channel_config is None:
            quantum_channel_config = {
                "names": ["qchannel_ba", "qchannel_bc"],
                "connecting_node_names": ["node_A", "node_C"],
                "connecting_port_of_node_names": ["qin_ba", "qin_bc"],
                "fiber_lengths": [50, 50],  # km
                "specify_models": [
                    {"quantum_loss_model": FibreLossModel(p_loss_init=0.2, p_loss_length=0.25)},
                    {"quantum_loss_model": FibreLossModel(p_loss_init=0.2, p_loss_length=0.25)}
                ]
            }

        self.node_config = node_config
        self.quantum_channel_config = quantum_channel_config

        # generate nodes into node_lst
        self.node_lst = self.generate_nodes()
        self.qchannel_lst = self.generate_qchannels()

    def generate_nodes(self):
        node_lst = []
        for idx, node_name in enumerate(self.node_config["names"]):
            port_names = self.node_config["port_names"][idx]
            # generate node
            if port_names is not None:
                node = Node(name=node_name, port_names=port_names)
            else:
                node = Node(name=node_name)

            # generate quantum memory
            quantum_memory_name = self.node_config["quantum_memory_names"][idx]
            quantum_memory_position = self.node_config["quantum_memory_positions"][idx]
            depolar_rate = self.node_config["depolar_rates"][idx]
            depolar_noise_model = DepolarNoiseModel(depolar_rate=depolar_rate)
            qmemory = QuantumMemory(quantum_memory_name, num_positions=quantum_memory_position, memory_noise_models=[depolar_noise_model]*quantum_memory_position)

            # add quantum memory into node
            node.add_subcomponent(qmemory, name=quantum_memory_name)

            # forward node port to qmemory port
            if port_names is not None:
                for port_name in port_names:
                    if port_name in self.node_config["port_forwardings"].keys():
                        qmemory_port_name = self.node_config["port_forwardings"][port_name]
                        node.ports[port_name].forward_input(node.qmemory.ports[qmemory_port_name])

            node_lst.append(node)
        return node_lst

    def generate_qchannels(self):
        qchannel_lst = []
        for idx, channel_name in enumerate(self.quantum_channel_config["names"]):

            fiber_length = self.quantum_channel_config["fiber_lengths"][idx]
            specify_models = self.quantum_channel_config["specify_models"][idx]
            qchannel = QuantumChannel(
                name=channel_name,
                length=fiber_length,
                models=specify_models
            )

            # get node by node_name
            connecting_node_name = self.quantum_channel_config["connecting_node_names"][idx]
            connecting_node = self.node_lst[self.node_config["names"].index(connecting_node_name)]

            # connect qchannel with port of node
            connecting_port_of_node_name = self.quantum_channel_config["connecting_port_of_node_names"][idx]
            qchannel.ports['recv'].connect(connecting_node.ports[connecting_port_of_node_name])

            qchannel_lst.append(qchannel)

        return qchannel_lst


if __name__ == "__main__":
    network = QuantumNetwork()
    pprint(network.node_config)
    pprint(network.quantum_channel_config)
    pprint(network.node_lst)
    pprint(network.qchannel_lst)
