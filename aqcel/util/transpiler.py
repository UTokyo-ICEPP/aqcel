from qiskit import *
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager
#from qiskit.transpiler.passes import Unroller

#from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit

class transpiler():
    
    def __init__(self, qc, backend=None, backend_tket=None, oplevel_tket=2, oplevel_transpiler=3):
        self.qc = qc
        self.backend = backend
        self.backend_tket = backend_tket
        self.oplevel_transpiler = oplevel_transpiler
        self.oplevel_tket = oplevel_tket
        
    def transpile(self):
#        if self.backend_tket is not None:
#            qc = transpile(self.qc, basis_gates=['id','x','sx','rz','cx','reset'])
#            qc_tket = qiskit_to_tk(qc)
#            self.backend_tket.get_compiled_circuit(qc_tket, optimisation_level=self.oplevel_tket)
#            qc = tk_to_qiskit(qc_tket)
#        else:
#            qc = self.qc
        
        transpiled_qc = transpile(circuits=qc, backend=self.backend, basis_gates=None, seed_transpiler=1, optimization_level=self.oplevel_transpiler)
        
        return transpiled_qc
