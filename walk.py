from fktools import *
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

def control_add_one(position_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(position_qubits + 1, name='control_add_one')

    for i in range(position_qubits, 0, -1):
        qc.mcx([j for j in range(i)], i)
    
    return qc

class walk_circuit_simpler:
    def __init__(self, nodes_power, num_steps, use_phase_defect=False, defect_step=None, defect_qubit=None, defect_strength=None):
        self.nodes_power = nodes_power
        self.num_steps = num_steps
        self.use_phase_defect = use_phase_defect
        self.defect_step = defect_step
        self.defect_qubit = defect_qubit
        self.defect_strength = defect_strength

    def build(self) -> QuantumCircuit:
        q_pos = QuantumRegister(self.nodes_power, "q_pos")
        q_coin = QuantumRegister(1, "q_coin")
        c_pos = ClassicalRegister(self.nodes_power, "c_pos")
        regs = [q_pos, q_coin, c_pos]

        qc = QuantumCircuit(*regs)

        qc.x(q_pos[-1])

        for step in range(self.num_steps):
            # coin
            qc.h(q_coin)


            # defect
            if self.use_phase_defect and self.defect_step is not None and self.defect_qubit is not None and self.defect_strength is not None:
                if step == self.defect_step:
                    qc.p(self.defect_strength, self.defect_qubit)


            # shift right for coin |1>
            qc.append(control_add_one(self.nodes_power), [q_coin, *q_pos])
            
            # shift left for coin |0>
            qc.x(q_coin)
            qc.x(q_pos)
            qc.append(control_add_one(self.nodes_power), [q_coin, *q_pos])
            qc.x(q_coin)
            qc.x(q_pos)


        qc.measure(q_pos, c_pos)

        self.qc = qc

        return qc
    
    def run(self, simulator = AerSimulator(), shots = 1024):

        tcirc = transpile(self.qc, simulator)

        result = simulator.run(tcirc, shots=shots).result()
        counts = result.get_counts(0)

        probs = np.zeros(shape=(2**self.nodes_power), dtype=np.float32)

        for key in counts.keys():
            probs[int(key, 2)] = counts[key]
        probs /= np.sum(probs)

        return probs