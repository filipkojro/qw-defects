from fktools import *

from walk import walk_circuit_simpler
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

from tqdm import tqdm

simulator_backend = AerSimulator()
print(f"using {simulator_backend.name} for simulator backend")

noisy_backend = AerSimulator()
print(f"\nusing {noisy_backend.name} for defected backend")

with_defect = False

while True:

    # randomizing input
    nodes_power = 3
    num_steps = np.random.randint(1, 24)

    start_bits = ""
    for i in range(nodes_power): start_bits += np.random.choice(['0', '1'])

    coin_phase = np.random.rand() * 2 * np.pi

    # defect input
    defect_step = np.random.randint(0, num_steps)
    defect_qubit = np.random.randint(0, nodes_power)
    defect_strength = np.random.rand() * 2 * np.pi


    # running simulations
    if with_defect:
        simulator_wcs = walk_circuit_simpler(nodes_power=nodes_power, num_steps=num_steps, start_bits=start_bits, coin_phase=coin_phase, use_phase_defect=True, defect_step=defect_step, defect_qubit=defect_qubit, defect_strength=defect_strength)
        noisy_wcs =     walk_circuit_simpler(nodes_power=nodes_power, num_steps=num_steps, start_bits=start_bits, coin_phase=coin_phase, use_phase_defect=True, defect_step=defect_step, defect_qubit=defect_qubit, defect_strength=defect_strength)
    else:
        simulator_wcs = walk_circuit_simpler(nodes_power=nodes_power, num_steps=num_steps, start_bits=start_bits, coin_phase=coin_phase)
        noisy_wcs =     walk_circuit_simpler(nodes_power=nodes_power, num_steps=num_steps, start_bits=start_bits, coin_phase=coin_phase)
    
    
    simulator_wcs.build()
    simulator_probs = simulator_wcs.run(simulator=simulator_backend)
    
    noisy_wcs.build()
    noisy_probs = noisy_wcs.run(simulator=noisy_backend)

    # adding to dataset
    X = np.load("dataset_onlydefects_perfect_X.npz")['arr_0']
    y = np.load("dataset_onlydefects_perfect_y.npz")['arr_0']


    X = np.vstack((X, noisy_probs))
    y = np.vstack((y, [1] if with_defect else [0]))


    # X = np.array([noisy_probs])
    # y = np.array([[1] if with_defect else [0]])


    np.savez_compressed("dataset_onlydefects_perfect_X", X)
    np.savez_compressed("dataset_onlydefects_perfect_y", y)

    print(X.shape, y.shape)

    with_defect = not with_defect