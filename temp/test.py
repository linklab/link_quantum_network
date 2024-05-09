import netsquid as ns
print(ns.__version__)
qubits = ns.qubits.create_qubits(num_qubits=1)

qubit = qubits[0]

print("1." + "-" * 50)
print(ns.qubits.reduced_dm(qubit))
print("1." + "-" * 50)

ns.qubits.operate(qubits=qubit, operator=ns.X)

print("2." + "-" * 50)
print(ns.qubits.reduced_dm(qubit))
print("2." + "-" * 50)

measurement_result, prob = ns.qubits.measure(qubit=qubit, observable=ns.Z)
if measurement_result == 0:
    state = "|0>"
else:
    state = "|1>"

print("3." + "-" * 50)
print(f"Measured {state} with probability {prob:.1f}")
print(ns.qubits.reduced_dm(qubit))
print("3." + "-" * 50)

measurement_result, prob = ns.qubits.measure(qubit=qubit, observable=ns.X)
if measurement_result == 0:
    state = "|+>"
else:
    state = "|->"

print("4." + "-" * 50)
print(f"Measured {state} with probability {prob:.1f}")
print(ns.qubits.reduced_dm(qubit))
print("4." + "-" * 50)