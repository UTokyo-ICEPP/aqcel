import copy

from qiskit import *
from qiskit.providers.aer import StatevectorSimulator
from qiskit.visualization import plot_histogram
from qiskit import quantum_info


class simulator():
    
    def __init__(self, qc):
        self.qc = qc
        
    def aer_simulator(self, shots):
    
        simulator = Aer.get_backend('qasm_simulator')
        job = execute(self.qc, simulator, shots=shots)
        result = job.result()
        counts = result.get_counts()

        return counts
        
    def statevector_simulator(self, output, threshold=1e-10, precision='double'):
        
        qc_copy = copy.deepcopy(self.qc)
        qc_copy.remove_final_measurements()
        
        simulator = StatevectorSimulator(precision=precision) # When the size of quantum circuit is large, "single" is better.
        job = execute(qc_copy, simulator)
        result = job.result()
        statevector = result.get_statevector()
        final_result = quantum_info.Statevector(statevector)
        
        if output == 'statevector':
            return final_result
        
        if output == 'probabilities_distribution':
            
            dic = final_result.probabilities_dict()
            for key in list(dic.keys()):
                if dic[key] < threshold: # Cut off a floating point numerical error
                    dic.pop(key)
            
            return dic