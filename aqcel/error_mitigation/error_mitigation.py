from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options
from qiskit.utils import QuantumInstance
from qiskit.utils.mitigation import CompleteMeasFitter

class error_mitigation:
    def __init__(self, backend):
        self.backend = backend

    def apply(self, qc_list, shots):
        qi = QuantumInstance(backend=self.backend, shots=shots, measurement_error_mitigation_cls=CompleteMeasFitter, cals_matrix_refresh_period=0)
        result = qi.execute(qc_list).results[0].data
        return result
        
        #        service = QiskitRuntimeService(channel="ibm_quantum")
#        backend = service.backend(self.backend)
        
#        options = Options(resilience_level = 1) # 1 = measurement error mitigation
#        sampler = Sampler(session=backend, options=options)

        # Run
#        results = sampler.run(qc_list, shots=shots).result()
#        return results
        
#        meas_calibs, state_labels, measure_list = info_list[0], info_list[1], info_list[2]
        
#        managed_results = IBMQ.qc_experiment(qc_list + meas_calibs, self.backend, shots).run()
#        results = managed_results.combine_results()
        
 #       meas_fitter = CompleteMeasFitter(results, state_labels, qubit_list=measure_list)
 #       meas_filter = meas_fitter.filter
 #       measurement_error_mitigated_results = meas_filter.apply(results)
                
#        return [results, measurement_error_mitigated_results]
