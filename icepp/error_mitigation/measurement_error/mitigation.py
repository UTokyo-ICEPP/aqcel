from qiskit.ignis.mitigation.measurement import MeasurementFilter
from qiskit_experiments.database_service import DbExperimentDataV1 as DbExperimentData

from .experiment import MeasurementErrorExperiment

class MeasurementErrorMitigation:
    def __init__(self, backend, qubits):
        self.backend = backend
        self.qubits = qubits
        self.filter = None
        
    def run_experiment(self, circuits_per_state=1):
        exp = MeasurementErrorExperiment(self.qubits, circuits_per_state=circuits_per_state)
        exp_data = exp.run(backend=self.backend, shots=self.backend.configuration().max_shots)
        print('Experiment ID:', exp_data.experiment_id)
        exp_data.block_for_results()
        exp_data.save()
        self._load_from_exp_data(exp_data)
        
        return exp_data.experiment_id
        
    def load_matrix(self, experiment_id):
        exp_data = DbExperimentData.load(experiment_id, self.backend.provider().service("experiment"))
        self._load_from_exp_data(exp_data)
        
    def _load_from_exp_data(self, exp_data):
        analysis_result = exp_data.analysis_results()[0]
        self.filter = MeasurementFilter(analysis_result.value, analysis_result.extra)

    def apply(self, counts_list):
        corrected_counts = []
        for counts in counts_list:
            corrected_counts.append(self.filter.apply(counts))
        
        return corrected_counts
