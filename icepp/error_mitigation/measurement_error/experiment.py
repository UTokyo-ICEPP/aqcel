import matplotlib.pyplot as plt
from qiskit import QuantumRegister
from qiskit.result import Result
from qiskit.test.mock import FakeValencia
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit_experiments.framework import BaseExperiment, BaseAnalysis, Options, AnalysisResultData

class MeasurementErrorAnalysis(BaseAnalysis):
    @classmethod
    def _default_options(cls):
        return Options()
    
    def _run_analysis(self, experiment_data, parameter_guess=None, plot=True, ax=None):
        state_labels = []
        for datum in experiment_data.data():
            state_label = datum['metadata']['state_label']
            if state_label in state_labels:
                break
            state_labels.append(state_label)

        meas_fitter = CompleteMeasFitter(None, state_labels, circlabel='mcal')
        
        nstates = len(state_labels)

        for job_id in experiment_data.job_ids:
            full_result = experiment_data.backend.retrieve_job(job_id).result()
            # full_result might contain repeated experiments
            for iset in range(len(full_result.results) // nstates):
                try:
                    date = full_result.date
                except:
                    date = None
                try:
                    status = full_result.status
                except:
                    status = None
                try:
                    header = full_result.header
                except:
                    header = None
                    
                result = Result(full_result.backend_name, full_result.backend_version, \
                                full_result.qobj_id, full_result.job_id, \
                                full_result.success, full_result.results[iset * nstates:(iset + 1) * nstates], \
                                date=date, status=status, header=header, **full_result._metadata)

                meas_fitter.add_data(result)
        
        results = [
            AnalysisResultData('error_matrix', meas_fitter.cal_matrix, extra=state_labels)
        ]
                
        plots = []
        if plot:
            figure, ax = plt.subplots(1, 1)
            meas_fitter.plot_calibration(ax=ax)
            plots.append(figure)
        
        return results, plots
    
class MeasurementErrorExperiment(BaseExperiment):
    __analysis_class__ = MeasurementErrorAnalysis
    
    def __init__(self, qubit_list, circuits_per_state=1):
        super().__init__(qubit_list)
        
        self.circuits_per_state = circuits_per_state

    def circuits(self, backend=None):
        if backend is None:
            backend = FakeValencia()
            sys.stderr.write('MeasurementErrorExperiment: Using FakeValencia for backend\n')
            
        qreg = QuantumRegister(len(self.physical_qubits))

        circuits, state_labels = complete_meas_cal(qubit_list=list(range(qreg.size)), qr=qreg, circlabel='mcal')
        for circuit, state_label in zip(circuits, state_labels):
            circuit.metadata = {
                'experiment_type': self._type,
                'physical_qubits': self.physical_qubits,
                'state_label': state_label
            }

        return circuits * self.circuits_per_state
