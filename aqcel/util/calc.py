from qiskit import *
#from qiskit_ibm_provider import IBMProvider
from qiskit.quantum_info.analysis import hellinger_fidelity

def fidelity(qc1, qc2, backend, shots=1024):
#    backend_ideal = IBMProvider().get_backend("ibmq_qasm_simulator")
    qc1_transpiled = transpile(qc1, backend_ideal, optimization_level=3)
    qc2_transpiled = transpile(qc2, backend, optimization_level=3)
    res1 = backend_ideal.run(qc1_transpiled, shots=shots).result()
    res2 = backend_ideal.run(qc2_transpiled, shots=shots).result()
    res3 = backend.run(qc2_transpiled, shots=shots).result()

    print(f'count1 = {res1.get_counts()}')
    print(f'count2 = {res2.get_counts()}')    
    print(f'count3 = {res3.get_counts()}')    
    ret1 = hellinger_fidelity(res1.get_counts(), res2.get_counts())
    print('Fidelity (sim)  = {:.4f}'.format(ret1))
    ret2 = hellinger_fidelity(res1.get_counts(), res3.get_counts())
    print('Fidelity (meas)  = {:.4f}'.format(ret2))
    return [ret1, ret2]

def fidelity_sim(qc1, qc2, backend, shots=1024):
#    backend_ideal = IBMProvider().get_backend("ibmq_qasm_simulator")
    qc1_transpiled = transpile(qc1, backend_ideal, optimization_level=3)
    qc2_transpiled = transpile(qc2, backend, optimization_level=3)
    res1 = backend_ideal.run(qc1_transpiled, shots=shots).result()
    res2 = backend_ideal.run(qc2_transpiled, shots=shots).result()
    ret = hellinger_fidelity(res1.get_counts(), res2.get_counts())
    print('Fidelity (sim)  = {:.4f}'.format(ret))
    return ret
    
def fidelity_meas(qc1, qc2, backend, shots=1024):
#    backend_ideal = IBMProvider().get_backend("ibmq_qasm_simulator")
    qc1_transpiled = transpile(qc1, backend_ideal, optimization_level=3)
    qc2_transpiled = transpile(qc2, backend, optimization_level=3)
    res1 = backend_ideal.run(qc1_transpiled, shots=shots).result()
    res2 = backend.run(qc2_transpiled, shots=shots).result()
    ret = hellinger_fidelity(res1.get_counts(), res2.get_counts())
    print('Fidelity (meas) = {:.4f}'.format(ret))
    return ret
    