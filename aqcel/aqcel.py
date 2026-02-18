#from . import passes, util
from . import passes, util, error_mitigation
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import hellinger_fidelity
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import transpile
from qiskit_aer import AerSimulator
import math
import mthree

class aqcel():
    def __init__(self, qc, backend=None, backend_tket=None, shots=1024, threshold_type=['constant', 0.2], zne='off', fSkip=False):

        self.qc = qc
        self.backend = backend
        self.backend_tket = backend_tket
        self.shots = shots
        self.threshold_type = threshold_type
        self.zne = zne
        self.fSkip = fSkip
        self.oplevel_tket = 2
        self.oplevel_transpiler = 3
        self.sampler = None
        self.session = None
        
    def run(self):
        service = QiskitRuntimeService()
        print('AQCEL starts running')

#        with Session(backend=self.backend):
        if self.backend.name=='aer_simulator':
            self.sampler = Sampler(self.backend)
            pass
        else:
            self.session = Session(backend=self.backend)
            self.sampler = Sampler(mode=self.session)
            self.sampler.options.experimental = {"skip_transpilation": True}
            self.sampler.options.dynamical_decoupling.enable = False
            self.sampler.options.execution.rep_delay = 0.0005

        qc_aqcel = self.apply_aqcel()

        nMeas = 1            
        quasi_dists = []
        pm_real = generate_preset_pass_manager(backend=self.backend, optimization_level=3)
        print('Calculating Fidelity ...')
        QuasiDist_meas = []
        for i in range(nMeas):
            isa_circ = pm_real.run(qc_aqcel)
            job = self.sampler.run([(isa_circ)], shots=self.shots)
            result = job.result()

            # SamplerV1
            #raw_prob = result.quasi_dists[0].binary_probabilities()
            #prob = {i: raw_prob[i] for i in raw_prob.keys() if raw_prob[i]>0}

            # SamplerV2
            raw_counts = result[0].data.meas.get_counts()
            if self.backend.name=='aer_simulator':
                prob = {i: raw_counts[i]/self.shots for i in raw_counts.keys()}
            else:
                mit = mthree.M3Mitigation(self.backend)
                mapping = mthree.utils.final_measurement_mapping(isa_circ)
                mit.cals_from_system(mapping, len(mapping)*self.shots, runtime_mode=self.session)
                raw_prob = mit.apply_correction(raw_counts, mapping)
                prob = {j: abs(raw_prob[j]) for j in raw_prob.keys()}
            
            QuasiDist_meas.append(prob)

        if self.backend.name!='aer_simulator':
            self.session.close()

        pm_ideal = generate_preset_pass_manager(backend=AerSimulator(), optimization_level=0)
        backend_ideal = AerSimulator()
        self.sampler = Sampler(AerSimulator())
        self.sampler.options.default_shots = self.shots

        result_ideal = self.sampler.run([(pm_ideal.run(self.qc))], shots=self.shots).result()
        counts_ideal = result_ideal[0].data.meas.get_counts()
        IdealDistBeforeAQCEL = {i: counts_ideal[i]/self.shots for i in counts_ideal.keys()}
        
        # SamplerV1
        #IdealDistBeforeAQCEL = self.sampler.run([(pm_ideal.run(self.qc))], shots=self.shots).result().quasi_dists[0].binary_probabilities()
        print(f'   IdealDistBeforeAQCEL : {IdealDistBeforeAQCEL}')

        QuasiDist_sim = []
        for i in range(nMeas):
            job = self.sampler.run([(pm_ideal.run(qc_aqcel))], shots=self.shots)
            result = self.sampler.run([(pm_ideal.run(qc_aqcel))], shots=self.shots).result()
            raw_counts = result[0].data.meas.get_counts()
            probs = {i: raw_counts[i]/self.shots for i in raw_counts.keys()}
            QuasiDist_sim.append(probs)
            # SamplerV1
#                QuasiDist_sim.append(result.quasi_dists[0].binary_probabilities())

        sum_sim = sum2_sim = ave_sim = err_sim = 0.0
        sum_meas = sum2_meas = ave_meas = err_meas = 0.0
        fid_sim = []
        fid_meas = []
        for i in range(nMeas):
            fid = hellinger_fidelity(IdealDistBeforeAQCEL, QuasiDist_sim[i])
            fid_sim.append(fid)
            print(f'   F_sim  measurement {i} = {fid}, {QuasiDist_sim[i]}')
            sum_sim += fid
            ave_sim = sum_sim/nMeas
            fid = hellinger_fidelity(IdealDistBeforeAQCEL, QuasiDist_meas[i])
            fid_meas.append(fid)
            print(f'   F_meas measurement {i} = {fid}, {QuasiDist_meas[i]}')
            sum_meas += fid
            ave_meas = sum_meas/nMeas
        for i in range(nMeas):
            sum2_sim  += (ave_sim  - fid_sim[i]) * (ave_sim - fid_sim[i])
            sum2_meas += (ave_meas - fid_meas[i]) * (ave_meas - fid_meas[i])
        err_sim = math.sqrt(sum2_sim)/nMeas
        err_meas = math.sqrt(sum2_meas)/nMeas
        print(f'Fidelity(sim.)  = {ave_sim:.4f} +- {err_sim:.4f}')
        print(f'Fidelity(meas.) = {ave_meas:.4f} +- {err_meas:.4f}')

        return [qc_aqcel, ave_sim, err_sim, ave_meas, err_meas]

    def apply_aqcel(self):
        print(self.threshold_type)
        print('AQCEL: decompose_mcu')
        qc1 = passes.decompose().decompose_mcu(self.qc)
        print('AQCEL: remove_adjacent')
        qc2 = passes.other_passes().remove_adjacent(qc1)
        
        qc3 = passes.remove_controlled_operations(qc2).run_qc(self.session, self.sampler, self.shots, self.backend, self.threshold_type, self.fSkip)
        
        qc4 = passes.other_passes().remove_adjacent(qc3)
        qc5 = passes.other_passes().remove_qubits(qc4)
        return qc5

    def set_tket(self, backend_tket, oplevel_tket):
        self.backend_tket = backend_tket
        self.oplevel_tket = oplevel_tket

    def set_transpiler(self, backend, oplevel_transpiler):
        self.backend = backend
        self.oplevel_transpiler = oplevel_transpiler
    
    @staticmethod
    def apply_mitigation(qc, backend, shots, zne):
        
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

#            results = error_mitigation.error_mitigation(backend).apply([qc], shots)
            
            result_raw = results[0].get_counts(0)
            result_meas_mitigated = results[1].get_counts(0)
            
            print('Raw counts:', result_raw)
            print('Measurement error mitigated counts', result_meas_mitigated)
        
            return [result_raw, result_meas_mitigated]