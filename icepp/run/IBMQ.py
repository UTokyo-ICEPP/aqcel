from qiskit import *
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.tools.monitor import job_monitor, backend_monitor, backend_overview # You can check pending jobs of all backends.

"""
from qiskit.providers.ibmq import least_busy

# バックエンド（実機）のうち量子ビット数2個以上のもののリストをプロバイダから取得し、一番空いているものを選ぶ
backend_filter = lambda b: (not b.configuration().simulator) and (b.configuration().n_qubits >= 3) and b.status().operational
backend = least_busy(provider.backends(filters=backend_filter))

print('Jobs will run on', backend.name())
"""

# Update : Qiskit.runtime

class qc_experiment():
    
    def __init__(self, qc_list, backend, shots):
        """
        qc must be transpiled.
        """
        self.qc_list = qc_list
        self.backend = backend
        self.shots = shots
        
        
    def run(self):
        
        job_manager = IBMQJobManager()
        job_set = job_manager.run(self.qc_list, backend=self.backend, shots=self.shots)
        
        print("Job id :", job_set.job_set_id())
        job_monitor(job_set.jobs()[0])

        results = job_set.results()
        
        return results