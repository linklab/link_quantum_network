from netsquid.components.qprocessor import QuantumProcessor
import netsquid.components.instructions as instr
import netsquid as ns

qproc = QuantumProcessor(name="ExampleQPU", num_positions=3, fallback_to_nonphysical=True)

qproc.execute_instruction(instruction=instr.INSTR_INIT, qubit_mapping=[0, 1])
qproc.execute_instruction(instruction=instr.INSTR_H, qubit_mapping=[1])
qproc.execute_instruction(instruction=instr.INSTR_CNOT, qubit_mapping=[1, 0])
m1 = qproc.execute_instruction(instruction=instr.INSTR_MEASURE, qubit_mapping=[0])
m2 = qproc.execute_instruction(instruction=instr.INSTR_MEASURE, qubit_mapping=[1])
print(m1, m2)
print(m1 == m2)  # Measurement results are both the same (either both 1 or both 0)
print(m1 is m2)
print(ns.sim_time())

##################################################################################################
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.components.qprocessor import PhysicalInstruction

phys_instructions = [
    PhysicalInstruction(instr.INSTR_INIT, duration=5),
    PhysicalInstruction(instr.INSTR_H, duration=1, parallel=True, topology=[0, 2]),
    PhysicalInstruction(instr.INSTR_CNOT, duration=4, parallel=True, topology=[(0, 1), (2, 1)]),
    PhysicalInstruction(instr.INSTR_X, duration=1, parallel=True, topology=[0, 2]),
    PhysicalInstruction(instr.INSTR_Z, duration=1, parallel=True, topology=[0, 2]),
    PhysicalInstruction(instr.INSTR_S, duration=1, parallel=True, topology=[0, 2]),
    PhysicalInstruction(
        instr.INSTR_MEASURE, duration=7, parallel=False,
        quantum_noise_model=DepolarNoiseModel(depolar_rate=0.01, time_independent=True),
        apply_q_noise_after=False, topology=[1]
    ),
    PhysicalInstruction(
        instr.INSTR_MEASURE, duration=7, parallel=True, topology=[0, 2]
    )
]
noisy_qproc = QuantumProcessor(
    name="NoisyQPU", num_positions=3,
    mem_noise_models=[DepolarNoiseModel(1e7)] * 3,
    phys_instructions=phys_instructions
)

print(ns.sim_time())
noisy_qproc.execute_instruction(instr.INSTR_INIT, [0, 1])
ns.sim_run()
print(ns.sim_time())

print(noisy_qproc)

#################################

from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from netsquid.components.instructions import INSTR_INIT, INSTR_H, INSTR_CNOT, INSTR_MEASURE
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.qubits import qubitapi as qapi
from netsquid.protocols import NodeProtocol
from netsquid.nodes import Node
from netsquid.components import QuantumProgram

# 명령 정의
phys_instructions_2 = [
    PhysicalInstruction(INSTR_INIT, duration=3),
    PhysicalInstruction(INSTR_H, duration=1, qubit_mapping=[0, 2]),
    PhysicalInstruction(INSTR_CNOT, duration=4, qubit_mapping=[1]),
    PhysicalInstruction(INSTR_MEASURE, duration=7, qubit_mapping=[0, 1, 2], error_model=DepolarNoiseModel(depolar_rate=0.01))
]

# 양자 프로세서 생성 및 설정
memory_noise_model = DepolarNoiseModel(depolar_rate=0.01, time_step=1)
qproc = QuantumProcessor(
    name="ExampleQPU",
    num_positions=3,
    memory_noise_models=[memory_noise_model] * 3,
    phys_instructions=phys_instructions_2
)

# 양자 프로그램 정의
class EntanglingProgram(QuantumProgram):
    def program(self):
        self.apply(INSTR_INIT, [0, 1])
        self.apply(INSTR_H, [1])
        self.apply(INSTR_CNOT, [1, 0])
        m1 = self.apply(INSTR_MEASURE, [0])
        m2 = self.apply(INSTR_MEASURE, [1])
        yield self.run(parallel=True)  # 병렬 실행을 활성화
        results = [self.output[m1], self.output[m2]]
        return results


# 노드와 프로토콜 설정
class MyProtocol(NodeProtocol):
    def run(self):
        qmem = self.node.qmemory
        prog = EntanglingProgram()
        result = yield self.node.qmemory.execute_program(prog)
        m1, m2 = prog.output[result]
        print(f"Measurement results: m1={m1}, m2={m2}, m1==m2: {m1 == m2}")

# 노드 생성
node = Node("QuantumNode", qmemory=qproc)

# 프로토콜 실행
protocol = MyProtocol(node)
protocol.start()
stats = ns.sim_run(10)
print(stats)