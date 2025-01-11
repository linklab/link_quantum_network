import netsquid as ns
from netsquid.components import FibreLossModel, QuantumChannel
from netsquid.components.models.delaymodels import FibreDelayModel
from netsquid.components import QuantumMemory
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.qubits.qubitapi import create_qubits

delay_model = FibreDelayModel()
loss_model = FibreLossModel(p_loss_init=0.83, p_loss_length=0.2)

qchannel = QuantumChannel(
    name="MyQChannel", length=20, models={'quantum_loss_model': loss_model}
)


# if set to the (arbitrary) value of 1 MHz means that after a microsecond there is a 63% probability of depolarization.
# 1 MHz의 임의의 값으로 설정된다면, 1 마이크로초 후에 63%의 확률로 탈분극이 일어남.
depolar_noise = DepolarNoiseModel(depolar_rate=1e6)  # the depolar_rate is in Hz

qmem = QuantumMemory(
    name="MyMemory", num_positions=2,
    memory_noise_models=[depolar_noise, depolar_noise]
)

qubits = create_qubits(1)
qmem.put(qubits)
print(qmem.peek(0))
qmem.pop(positions=0)
print(qmem.peek(0))


