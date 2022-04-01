import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

from qiskit import *
from qiskit.pulse import ShiftPhase,Play,Drag,GaussianSquare,DriveChannel,ControlChannel
import qiskit.pulse as pulse
import qiskit.pulse.library as pulse_lib
from qiskit.compiler import assemble
from qiskit.pulse.library import Waveform
from qiskit.tools.monitor import job_monitor
from qiskit_experiments.library import EFSpectroscopy, EFRabi


import copy
import sys
sys.path.append('..')
#from __future__ import division 

from lmfit.models import SineModel, PolynomialModel
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,
                                                 CompleteMeasFitter, TensoredMeasFitter)
from qiskit.visualization import plot_histogram


class sweep():
    
    def __init__(self, qubits, shots, backend, scale_factor, x01_freq, x01_amp, x12_freq, x12_amp):
        self.ctr_q1 = qubits[0]
        self.ctr_q2 = qubits[1]
        self.tar_q  = qubits[2]
        
        self.shots = shots
        self.backend = backend
        self.scale_factor = scale_factor
        
        self.x01_freq = x01_freq
        self.x01_amp  = x01_amp
        self.x12_freq = x12_freq
        self.x12_amp  = x12_amp
        
        # Backend
        backend_config = self.backend.configuration()
        assert backend_config.open_pulse, "Backend doesn't support Pulse"
        self.dt = backend_config.dt * 1e9 #ns
        backend_defaults = self.backend.defaults()
        
        # Default measurement pulse
        qc_meas_tar_q = QuantumCircuit(1,1)
        qc_meas_tar_q.measure(0,0)
        transpiled_qc_meas_tar_q = transpile(qc_meas_tar_q, self.backend, initial_layout = [self.tar_q])
        self.sched_meas_tar_q = schedule(transpiled_qc_meas_tar_q, self.backend)
        
        # Default x01 and x12
        from calibration import x01,x12
        test = x12.sweep(qubit=self.ctr_q2, shots=self.shots, backend=self.backend, scale_factor=self.scale_factor, x01_freq=self.x01_freq, x01_amp=self.x01_amp)
        self.sched_x = test.sched_x
        
        self.pi_pulse_12 = test.calibrated_x12(self.x12_freq, self.x12_amp)
        self.pi_pulse_12_reverse = test.calibrated_x12(self.x12_freq, -self.x12_amp)
        
        # Default CNOT
        qc_cx01 = QuantumCircuit(2)
        qc_cx01.cx(0,1)
        transpiled_qc_cx01 = transpile(qc_cx01, self.backend, initial_layout = [self.ctr_q2, self.tar_q])
        self.sched_cx01 = schedule(transpiled_qc_cx01, self.backend)
        
        #Default Rx(np.pi/2)
        qc_rx = QuantumCircuit(1)
        qc_rx.rx(np.pi/2,0)
        transpiled_qc_rx = transpile(qc_rx, self.backend, initial_layout = [self.tar_q])
        self.sched_rx = schedule(transpiled_qc_rx,self.backend)
        
    def new_cnot(self, rx_angle):
    
        sched_rx_copy = copy.deepcopy(self.sched_rx)
        sched_rx_copy.instructions[0][1].pulse._amp = sched_rx_copy.instructions[0][1].pulse._amp *rx_angle*2/np.pi

        sched_cx_copy = copy.deepcopy(self.sched_cx01)
        
        new = pulse.Schedule()
        #new += ShiftPhase(-3.141592653589793/2, ControlChannel(0))
        if rx_angle != 0:
            new += sched_rx_copy

        for instruction in sched_cx_copy.instructions:
            child = instruction[1]
            if type(child) == pulse.instructions.play.Play:
                if type(child.pulse) == pulse_lib.GaussianSquare:
                    if child.channel.__class__ == qiskit.pulse.channels.ControlChannel:
                        child.pulse._amp = -child.pulse._amp
                        new += child << new.duration
                        break
        return new
    
    def amp_sweep(self, drive_durations, amp_times, sched_cx, init_state):
        duration_schedules = []

        for cr_duration in drive_durations:
            sched = pulse.Schedule()
            sched_cx_copy = copy.deepcopy(sched_cx)

            if (init_state == 1) or (init_state == 2) :
                sched += self.sched_x
                
            if init_state == 2:
                sched += pulse.Play(self.pi_pulse_12, DriveChannel(self.ctr_q2))

            new_cx = pulse.Schedule()
            for index, instruction in enumerate(sched_cx_copy.instructions):
                child = instruction[1]

                if type(child.pulse) != pulse_lib.GaussianSquare:
                    new_cx += child
                else:
                    mini_pulse = child.pulse
                    mini_pulse._width = mini_pulse.width * cr_duration / mini_pulse.duration
                    mini_pulse.duration = cr_duration
                    mini_pulse._amp = amp_times * mini_pulse._amp
                    new_cx = new_cx.insert(new_cx.duration, child)

            sched |= new_cx << sched.duration
            sched |= self.sched_meas_tar_q << sched.duration # 駆動パルスの後に測定をシフト
            duration_schedules.append(sched)

        return duration_schedules
    
    def get_job_data(self, job, average):
        job_results = job.result()
        result_data = []
        for i in range(len(job_results.results)):
            if average: # 平均データを得る
                result_data.append(job_results.get_memory(i)[0]*self.scale_factor) 
            else: # シングルデータを得る
                result_data.append(job_results.get_memory(i)[:, 0]*self.scale_factor)  
        return result_data

    def duration_job(self, schedules):
        duration_expt_program = assemble(schedules, backend=self.backend, meas_level=1, meas_return='avg', shots=self.shots, 
                                         schedule_los=[{DriveChannel(self.ctr_q2):self.x01_freq}] * len(schedules))
        duration_job = self.backend.run(duration_expt_program)
        job_monitor(duration_job)
        duration_data = self.get_job_data(duration_job, average=True)
        return duration_data

    def duration_graph(self, duration_data, drive_durations, ylim):

        x = np.array(self.dt_to_ns(drive_durations))
        y = np.real(duration_data)

        model = SineModel()
        params = model.guess(y, x=x)
        result = model.fit(y, params, x=x)
        print(2*np.pi/result.best_values['frequency'])

        fig, ax = plt.subplots(dpi=130)
        result.plot_fit(ax=ax)
        ax.set_xlim(0, x[-1])
        ax.set_ylim(ylim[0], ylim[1])
        plt.xlabel("Drive duration [ns]", fontsize=15)
        plt.ylabel("Measured signal [a.u.]", fontsize=15)
        plt.title('Rabi Experiment (0->1)', fontsize=20)
        plt.show()
        
    def rabi02(self, amp_list, drive_durations, new_cx):
        data0_list=[]
        data2_list=[]
        for amp_times in amp_list:
            amp_schedules_0 = self.amp_sweep(drive_durations, amp_times=amp_times, sched_cx=new_cx, init_state=0)
            amp_schedules_2 = self.amp_sweep(drive_durations, amp_times=amp_times, sched_cx=new_cx, init_state=2)
            amp_data_0 = self.duration_job(schedules=amp_schedules_0)
            amp_data_2 = self.duration_job(schedules=amp_schedules_2)
            data0_list.append(amp_data_0)
            data2_list.append(amp_data_2)
        return data0_list, data2_list

    def fitting_rabi02(self, duration_data, drive_durations):
        x = np.array(self.dt_to_ns(drive_durations))
        y = np.real(duration_data)

        model = SineModel()
        params = model.guess(y, x=x)
        result = model.fit(y, params, x=x)

        return 2*np.pi/result.best_values['frequency']

    def rabi02freq(self, data0_list, data2_list, drive_durations):
        freq0_list=[]
        freq2_list=[]
        for amp_data_0 in data0_list:
            cycle0 = self.fitting_rabi02(duration_data = amp_data_0, drive_durations=drive_durations)
            freq0_list.append(cycle0)
        for amp_data_2 in data2_list:
            cycle2 = self.fitting_rabi02(duration_data = amp_data_2, drive_durations=drive_durations)
            freq2_list.append(cycle2)
        return freq0_list, freq2_list

    def rx_sweep(self, angle_list, drive_durations, amp_times):
        data0_list=[]
        data1_list=[]
        for angle in angle_list:
            new_cx = self.new_cnot(rx_angle=angle)
            amp_schedules_0 = self.amp_sweep(drive_durations, amp_times, sched_cx=new_cx, init_state=0)
            amp_schedules_1 = self.amp_sweep(drive_durations, amp_times, sched_cx=new_cx, init_state=1)
            amp_data_0 = self.duration_job(schedules=amp_schedules_0)
            amp_data_1 = self.duration_job(schedules=amp_schedules_1)
            data0_list.append(amp_data_0)
            data1_list.append(amp_data_1)
        return data0_list, data1_list
    
    def pure_cnot(self, pure_amp_times, pure_angle, pure_duration):
        pure_cnot = pulse.Schedule()
        new_cx = self.new_cnot(rx_angle=pure_angle)
        for instruction in new_cx.instructions:
            mini_pulse = instruction[1].pulse
            if type(mini_pulse) == pulse_lib.GaussianSquare:
                mini_pulse._width   = mini_pulse._width*round(pure_duration/self.dt/16)*16/mini_pulse.duration
                mini_pulse.duration = round(pure_duration/self.dt/16)*16
                mini_pulse._amp     = mini_pulse._amp*pure_amp_times
            pure_cnot += instruction[1]  << pure_cnot.duration

        return pure_cnot
    
    def dt_to_ns(self, dt_list):
        ns_list=[]
        for dt in dt_list:
            ns_list.append(dt*self.dt)
        return ns_list
    
    @staticmethod
    def interpolated_intercept(x, y1, y2):
    
        def intercept(point1, point2, point3, point4): 

            def line(p1, p2):
                A = (p1[1] - p2[1])
                B = (p2[0] - p1[0])
                C = (p1[0]*p2[1] - p2[0]*p1[1])
                return A, B, -C

            def intersection(L1, L2):
                D  = L1[0] * L2[1] - L1[1] * L2[0]
                Dx = L1[2] * L2[1] - L1[1] * L2[2]
                Dy = L1[0] * L2[2] - L1[2] * L2[0]

                x = Dx / D
                y = Dy / D
                return x,y

            L1 = line([point1[0],point1[1]], [point2[0],point2[1]])
            L2 = line([point3[0],point3[1]], [point4[0],point4[1]])

            R = intersection(L1, L2)

            return R

        list2 = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
        idx=list2[0][0]
        xc, yc = intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], y2[idx+1])))
        return xc,yc