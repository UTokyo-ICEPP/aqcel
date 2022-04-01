from . import error_mitigation, compiler, run


class pass_manager():
    
    def __init__(self, qc, level=None, backend=None, backend_tket=None, shots=None, measure_type=None, threshold_type=None, zne=None):
        
        """
        Error mitigation = Measurement error mitigation + CNOT error mitigation
        
        level 1 : Error mitigation
        level 2 : Error mitigation + AQCEL circuit optimization
        level 3 : Error mitigation + AQCEL circuit optimization + Toffoli on Qutrit (in progress)
        """
        
        self.qc = qc
        self.level = level
        self.backend = backend
        self.backend_tket = backend_tket
        self.shots = shots
        self.measure_type = measure_type
        self.threshold_type = threshold_type
        self.zne = zne
        
    def auto_manager(self):
        
        if self.level == 1:
            counts = self.mitigation_apply(self.qc, self.backend, self.shots, self.zne)
            
            return counts
            
        if self.level == 2:  
            optimized_qc = self.aqcel_apply()
            transpiled_qc = compiler.transpiler(optimized_qc, self.backend, self.backend_tket, level=3).transpile()
            #counts = self.mitigation_apply(transpiled_qc, self.backend, self.shots)
            
            return [optimized_qc, transpiled_qc]
            
    
    def aqcel_apply(self):
        
        #qc0 = compiler.other_passes().remove_adjacent(self.qc)
        qc1 = compiler.decompose().decompose_mcu(self.qc)
        qc2 = compiler.other_passes().remove_adjacent(qc1)
        
        if self.measure_type == 'cc':
            qc3 = compiler.remove_controlled_operations(qc2).run_cc()
        if self.measure_type == 'qc':
            qc3 = compiler.remove_controlled_operations(qc2).run_qc(self.shots, self.backend, self.backend_tket, self.threshold_type, self.zne)
        
        qc4 = compiler.other_passes().remove_adjacent(qc3)
        qc5 = compiler.other_passes().remove_qubits(qc4)
        
        return qc5
        
    @staticmethod
    def mitigation_apply(qc, backend, shots, zne):
        
        # ZNE for CNOT error mitigation
        if ('cx' in qc.count_ops()) and (zne == 'on'):
            qc_3cx = error_mitigation.zne(1,qc).fiim_generate_circs()
            transpiled_qc_3cx = compiler.transpiler(qc_3cx, backend, backend_tket=None, level=0).transpile()
        
            #Measurement error mitigation
            info_list = error_mitigation.MeasurementErrorMitigation_demo(backend).measured_qubits(qc)
            results = error_mitigation.MeasurementErrorMitigation_demo(backend).apply([qc,transpiled_qc_3cx], info_list, shots)
            
            result_raw = results[0].get_counts(0)
            results_meas_mitigated = results[1]

            result_1_cx = results_meas_mitigated.get_counts(0)
            result_3_cx = results_meas_mitigated.get_counts(1)

            cnot_mitigated_counts = error_mitigation.zne(1,qc).apply(result_1_cx, result_3_cx)
            
            print('Raw counts:',result_raw)
            print('Measurement error mitigated counts', result_1_cx)
            print('CNOT error mitigated counts', cnot_mitigated_counts)
            
            return [result_raw, result_1_cx, cnot_mitigated_counts]
        
        else:
            #Measurement error mitigation
            info_list = error_mitigation.MeasurementErrorMitigation_demo(backend).measured_qubits(qc)
            results = error_mitigation.MeasurementErrorMitigation_demo(backend).apply([qc], info_list, shots)
            
            result_raw = results[0].get_counts(0)
            result_meas_mitigated = results[1].get_counts(0)
            
            print('Raw counts:', result_raw)
            print('Measurement error mitigated counts', result_meas_mitigated)
        
            return [result_raw, result_meas_mitigated]