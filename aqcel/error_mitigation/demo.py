from qiskit.utils.mitigation import (complete_meas_cal, tensored_meas_cal, CompleteMeasFitter, TensoredMeasFitter)
from qiskit_ibm_runtime import Options

import sys
from ..util import IBMQ


class MeasurementErrorMitigation_demo:
    def __init__(self, backend):
        self.backend = backend
        
    def measured_qubits(self, qc):
        
        measure_list = []
        for op in qc.get_instructions('measure'):
            measure_list.append(op[1][0]._index)

        meas_calibs, state_labels = complete_meas_cal(qubit_list=measure_list)
        
        info_list = [meas_calibs, state_labels, measure_list]
        
        return info_list
    
    def apply(self, qc_list, info_list, shots):
        
        meas_calibs, state_labels, measure_list = info_list[0], info_list[1], info_list[2]
        
#        managed_results = IBMQ.qc_experiment(qc_list + meas_calibs, self.backend, shots).run()
#        managed_results = self.backend.run(qc_list).result()
#        results = managed_results.combine_results()
        results = self.backend.run(qc_list + meas_calibs, shots=shots).result()
        meas_fitter = CompleteMeasFitter(results, state_labels, qubit_list=measure_list)
        meas_filter = meas_fitter.filter
        measurement_error_mitigated_results = meas_filter.apply(results)

        
        options = Options()
        options.optimization_level = 3
        options.resilience_level = 1
        options.execution.shots = 2048
        sampler = Sampler(backend=backend, options=options)
        result = sampler.run(circuit, shots=1024).result()


        return [results, measurement_error_mitigated_results]