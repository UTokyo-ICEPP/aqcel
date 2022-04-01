from qiskit import *

class zne():
    
    def __init__(self, n, qc):
        self.n = n
        self.num_cx = 2*self.n + 1
        self.qc = qc
    
    def fiim_generate_circs(self):

        """Inserts up to n CNOT gate pairs, for every CNOT, into circuit.
        Args:
        n: The maximum number of CNOTs to add for each CNOT in circ.
        circuit: (QuantumCircuit) The quantum circuit to add CNOTs to.
        Returns:
        QuantumCircuit object which is equivalent (in terms of output) the original
        with extra CNOTs inserted.
        """
        
        qc_n_cx = QuantumCircuit(*self.qc.qregs, *self.qc.cregs)

        for gate in self.qc:

            if gate[0].name == 'cx':
                for _ in range(self.num_cx):
                    qc_n_cx.append(gate[0],gate[1],gate[2])

            else:
                qc_n_cx.append(gate[0],gate[1],gate[2])

        return qc_n_cx


    def apply(self, result_1_cx, result_n_cx):
        
        cnot_mitigated_counts = {}
        for key in result_1_cx.keys():
            
            # Unphysical counts (<0) are igonored (defined as 0 count).
            if key in result_n_cx:
                cnot_mitigated_counts[key] = self.num_cx/2 *result_1_cx[key] - 1/2 *result_n_cx[key]
            else:
                cnot_mitigated_counts[key] = self.num_cx/2 *result_1_cx[key]
            
        return cnot_mitigated_counts