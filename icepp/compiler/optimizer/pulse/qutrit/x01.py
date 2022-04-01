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


class sweep():
    
    def __init__(self, qubit, shots, backend, scale_factor):
        self.qubit = qubit
        self.shots = shots
        self.backend = backend
        self.scale_factor = scale_factor
        
        # Backend
        backend_config = self.backend.configuration()
        assert backend_config.open_pulse, "Backend doesn't support Pulse"
        self.dt = backend_config.dt
        backend_defaults = self.backend.defaults()
        self.default_qubit_freq = backend_defaults.qubit_freq_est[self.qubit]
        
        # Default measurement pulse
        qc_meas = QuantumCircuit(1,1)
        qc_meas.measure(0,0)
        transpiled_qc_meas = transpile(qc_meas, self.backend, initial_layout = [self.qubit])
        self.sched_meas = schedule(transpiled_qc_meas, backend)
        
        # Default pi pulse
        qc_x = QuantumCircuit(1)
        qc_x.x(0)
        transpiled_qc_x = transpile(qc_x, self.backend, initial_layout = [self.qubit])
        self.sched_x = schedule(transpiled_qc_x, self.backend)
        
        pulse_info = self.sched_x.instructions[0][1].pulse
        self.drive_samples = pulse_info.duration
        self.pi_amp_01 = pulse_info.amp
        self.drive_sigma = pulse_info.sigma
        
        self.times = 50
    
    def freq(self, amp, freq_med):
        if freq_med == None:
            ground_sweep_freqs = self.default_qubit_freq + np.linspace(-40e6, 40e6, self.times)
        else:
            ground_sweep_freqs = freq_med + np.linspace(-20e6, 20e6, self.times)
            
        ground_freq_sweep_program = self.create_ground_freq_sweep_program(ground_sweep_freqs, amp=amp)
        
        ground_freq_sweep_job = self.backend.run(ground_freq_sweep_program)
        job_monitor(ground_freq_sweep_job)
        ground_freq_sweep_data = self.get_job_data(ground_freq_sweep_job, average=True)
        
        return ground_freq_sweep_data, ground_sweep_freqs
    
    def fitting_freq(self, ground_freq_sweep_data, ground_sweep_freqs, parameters_real, parameters_imag):
        GHz = 1e9
        
        # Real fit
        (ground_sweep_fit_params_real, 
         ground_sweep_i_fit_real) = self.fit_function_single(ground_sweep_freqs, np.real(ground_freq_sweep_data), 
                                           lambda x, A, q_freq, B, C: (A / np.pi) * (B / ((x - q_freq)**2 + B**2)) + C,
                                           parameters_real, # initial parameters for curve_fit
                                           ([-np.inf, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf])) # ranges of parameters

        # Imag fit
        (ground_sweep_fit_params_imag, 
         ground_sweep_i_fit_imag) = self.fit_function_single(ground_sweep_freqs, np.imag(ground_freq_sweep_data), 
                                           lambda x, A, q_freq, B, C: (A / np.pi) * (B / ((x - q_freq)**2 + B**2)) + C,
                                           parameters_imag, # initial parameters for curve_fit
                                           ([-np.inf, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf])) # ranges of parameters
        
        print(ground_sweep_fit_params_real, ground_sweep_fit_params_imag)
        
        # 注：シグナルの実部だけをプロットします
        plt.scatter(ground_sweep_freqs/GHz, np.real(ground_freq_sweep_data), color='black')
        plt.plot(ground_sweep_freqs/GHz, ground_sweep_i_fit_real, color='red')
        plt.xlim([min(ground_sweep_freqs/GHz)+0.01, max(ground_sweep_freqs/GHz)]) # ignore min point (is off)
        plt.xlabel("Frequency [GHz]", fontsize=15)
        plt.ylabel("Measured Signal (Real) [a.u.]", fontsize=15)
        plt.title("0->1 Frequency Sweep (first pass)", fontsize=15)
        plt.show()
        
        # 注：シグナルの虚部だけをプロットします
        plt.scatter(ground_sweep_freqs/GHz, np.imag(ground_freq_sweep_data), color='black')
        plt.plot(ground_sweep_freqs/GHz, ground_sweep_i_fit_imag, color='red')
        plt.xlim([min(ground_sweep_freqs/GHz)+0.01, max(ground_sweep_freqs/GHz)]) # ignore min point (is off)
        plt.xlabel("Frequency [GHz]", fontsize=15)
        plt.ylabel("Measured Signal (Imag) [a.u.]", fontsize=15)
        plt.title("0->1 Frequency Sweep (first pass)", fontsize=15)
        plt.show()
        
        return ground_sweep_fit_params_real[1], ground_sweep_fit_params_imag[1]    
        
    def create_ground_freq_sweep_program(self, freqs, amp):
        if len(freqs) > 75:
            raise ValueError("You can only run 75 schedules at a time.")

        # Define the drive pulse
        ground_sweep_drive_pulse = pulse_lib.gaussian(duration=self.drive_samples, sigma=self.drive_sigma,
                                                      amp=amp, name='ground_sweep_drive_pulse')
        # Create the base schedule
        schedule = pulse.Schedule(name='Frequency sweep starting from ground state.')
        schedule |= pulse.Play(ground_sweep_drive_pulse, DriveChannel(self.qubit))
        schedule |= self.sched_meas << schedule.duration

        # define frequencies for the sweep
        schedule_freqs = [{DriveChannel(self.qubit): freq} for freq in freqs]

        # assemble the program
        ground_freq_sweep_program = assemble(schedule, backend=self.backend, meas_level=1, meas_return='avg',
                                             shots=self.shots, schedule_los=schedule_freqs)

        return ground_freq_sweep_program
        
        
    def amp(self, rabi_freq, amp_med):
        if amp_med == None:
            drive_amp_min = 0
            drive_amp_max = 0.5
            drive_amps = np.linspace(drive_amp_min, drive_amp_max, self.times)
        else:
            drive_amps = np.linspace(0, amp_med+0.1, self.times)

        # スケジュールの作成
        rabi_01_schedules = []

        # loop over all drive amplitudes
        for ii, drive_amp in enumerate(drive_amps):
            # drive pulse
            rabi_01_pulse = pulse_lib.gaussian(duration=self.drive_samples, amp=drive_amp, 
                                               sigma=self.drive_sigma, name='rabi_01_pulse_%d' % ii)

            # add commands to schedule
            schedule = pulse.Schedule(name='Rabi Experiment at drive amp = %s' % drive_amp)
            schedule |= pulse.Play(rabi_01_pulse, DriveChannel(self.qubit))
            schedule |= self.sched_meas << schedule.duration # shift measurement to after drive pulse
            rabi_01_schedules.append(schedule)
            
        rabi_01_expt_program = assemble(rabi_01_schedules, backend=self.backend, meas_level=1,
                                        meas_return='avg', shots=self.shots,
                                        schedule_los=[{DriveChannel(self.qubit): rabi_freq}] * self.times)
        
        rabi_01_job = self.backend.run(rabi_01_expt_program)
        job_monitor(rabi_01_job)
        rabi_01_data = self.get_job_data(rabi_01_job, average=True)

        return rabi_01_data, drive_amps
    
    def fitting_amp(self, rabi_12_data, drive_amps, parameters_real, parameters_imag):

        (rabi_12_fit_params_real, rabi_12_y_fit_real) = self.fit_function(drive_amps, np.real(rabi_12_data), 
                                    lambda x, A, B, drive_12_period, phi: (A*np.cos(2*np.pi*x/drive_12_period - phi) + B),
                                    parameters_real)
        
        (rabi_12_fit_params_imag, rabi_12_y_fit_imag) = self.fit_function(drive_amps, np.imag(rabi_12_data), 
                                    lambda x, A, B, drive_12_period, phi: (A*np.cos(2*np.pi*x/drive_12_period - phi) + B),
                                    parameters_imag)
        
        pi_amp_12_real = rabi_12_fit_params_real[2]/2
        pi_amp_12_imag = rabi_12_fit_params_imag[2]/2
        
        print(rabi_12_fit_params_real, rabi_12_fit_params_imag)

        plt.scatter(drive_amps, np.real(rabi_12_data), color='black')
        plt.plot(drive_amps, rabi_12_y_fit_real, color='red')
        plt.xlabel("Drive amp [a.u.]", fontsize=15)
        plt.ylabel("Measured signal (Real) [a.u.]", fontsize=15)
        plt.title('Rabi Experiment (0->1)', fontsize=20)
        plt.show()
        
        plt.scatter(drive_amps, np.imag(rabi_12_data), color='black')
        plt.plot(drive_amps, rabi_12_y_fit_imag, color='red')
        plt.xlabel("Drive amp [a.u.]", fontsize=15)
        plt.ylabel("Measured signal (Imag) [a.u.]", fontsize=15)
        plt.title('Rabi Experiment (0->1)', fontsize=20)
        plt.show()
        
        return pi_amp_12_real, pi_amp_12_imag
    
    def check(self, pi_pulse_01, rabi_freq):
        # 基底状態のスケジュール
        zero_schedule = pulse.Schedule(name="zero schedule")
        zero_schedule |= self.sched_meas

        # 励起状態のスケジュール
        one_schedule = pulse.Schedule(name="one schedule")
        one_schedule |= pulse.Play(pi_pulse_01, DriveChannel(self.qubit))
        one_schedule |= self.sched_meas << one_schedule.duration
        
        # プログラムにスケジュールを組み込みます
        IQ_01_program = assemble([zero_schedule, one_schedule], backend=self.backend, meas_level=1,
                                   meas_return='single', shots=self.shots, schedule_los=[{DriveChannel(self.qubit): rabi_freq}] * 2)
        
        IQ_01_job = self.backend.run(IQ_01_program)
        job_monitor(IQ_01_job)
        
        IQ_01_data = self.get_job_data(IQ_01_job, average=False)
        
        return IQ_01_data
    
    def IQ_01_plot(self, IQ_01_data, x_min, x_max, y_min, y_max):
        zero_data = IQ_01_data[0]
        one_data = IQ_01_data[1]

        """0、1のIQ平面をプロットするための補助関数。引数としてプロットの制限を与えます。
        """
        # 0のデータは青でプロット
        plt.scatter(np.real(zero_data), np.imag(zero_data), 
                        s=5, cmap='viridis', c='blue', alpha=0.5, label=r'$|0\rangle$')
        # 1のデータは赤でプロット
        plt.scatter(np.real(one_data), np.imag(one_data), 
                        s=5, cmap='viridis', c='red', alpha=0.5, label=r'$|1\rangle$')

        # 0、1の状態の結果の平均を大きなドットでプロット
        mean_zero = np.mean(zero_data) # 実部と虚部それぞれの平均をとる
        mean_one = np.mean(one_data)
        plt.scatter(np.real(mean_zero), np.imag(mean_zero), 
                    s=200, cmap='viridis', c='black',alpha=1.0)
        plt.scatter(np.real(mean_one), np.imag(mean_one), 
                    s=200, cmap='viridis', c='black',alpha=1.0)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min,y_max)
        plt.legend()
        plt.ylabel('Imag [a.u.]', fontsize=15)
        plt.xlabel('Real [a.u.]', fontsize=15)
        plt.title("0-1 discrimination", fontsize=15)
        
    def discriminator(self, IQ_01_data):
        # IQベクトルを作成します（実部と虚部で構成されています）
        zero_data_reshaped = self.reshape_complex_vec(IQ_01_data[0])
        one_data_reshaped = self.reshape_complex_vec(IQ_01_data[1])   

        IQ_01_data_copy = np.concatenate((zero_data_reshaped, one_data_reshaped))

        # （テスト用に）0と1の値が含まれたベクトルを構築します
        state_01 = np.zeros(self.shots) # 実験のショット数
        state_01 = np.concatenate((state_01, np.ones(self.shots)))

        # データをシャッフルして学習用セットとテスト用セットに分割します
        IQ_01_train, IQ_01_test, state_01_train, state_01_test = train_test_split(IQ_01_data_copy, state_01, test_size=0.5)

        # LDAを設定します
        LDA_01 = LinearDiscriminantAnalysis()
        LDA_01.fit(IQ_01_train, state_01_train)

        # 精度を計算します
        score_01 = LDA_01.score(IQ_01_test, state_01_test)
        print(score_01)
        
        return LDA_01
    
    # セパラトリックスを表示データの上にプロットします
    def separatrixPlot(self, lda, x_min, x_max, y_min, y_max):
        nx, ny = self.shots, self.shots

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
        Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z[:, 1].reshape(xx.shape)

        plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='black')
        
        
    def calibrated_x01(self, pi_amp_01):
        pi_pulse_01 = pulse_lib.gaussian(duration=self.drive_samples, amp=pi_amp_01, sigma=self.drive_sigma, name='pi_pulse_01')
        
        return pi_pulse_01
    
    def get_job_data(self, job, average):
        job_results = job.result() # タイムアウトパラメーターは120秒にセット
        result_data = []
        for i in range(len(job_results.results)):
            if average: # 平均データを得る
                result_data.append(job_results.get_memory(i)[0]*self.scale_factor) 
            else: # シングルデータを得る
                result_data.append(job_results.get_memory(i)[:, 0]*self.scale_factor)  
        return result_data

    @staticmethod
    def fit_function(x_values, y_values, function, init_params):
        """Fit a function using scipy curve_fit."""
        fitparams, conv = curve_fit(function, x_values, y_values, init_params)
        y_fit = function(x_values, *fitparams)

        return fitparams, y_fit
    
    @staticmethod
    def fit_function_single(x_values, y_values, function, init_params, ranges):
        """Fit a function using scipy curve_fit."""
        fitparams, conv = curve_fit(function, x_values, y_values, p0 = init_params, bounds = ranges)
        y_fit = function(x_values, *fitparams)

        return fitparams, y_fit
    
    @staticmethod
    def reshape_complex_vec(vec):
        length = len(vec)
        vec_reshaped = np.zeros((length, 2))
        for i in range(len(vec)):
            vec_reshaped[i]=[np.real(vec[i]), np.imag(vec[i])]
            
        return vec_reshaped