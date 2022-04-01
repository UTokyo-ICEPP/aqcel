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
    
    def __init__(self, qubit, shots, backend, scale_factor, x01_freq, x01_amp):
        self.qubit = qubit
        self.shots = shots
        self.backend = backend
        self.scale_factor = scale_factor
        
        # Backend
        backend_config = self.backend.configuration()
        assert backend_config.open_pulse, "Backend doesn't support Pulse"
        self.dt = backend_config.dt
        backend_defaults = self.backend.defaults()
        
        # Default measurement pulse
        qc_meas = QuantumCircuit(1,1)
        qc_meas.measure(0,0)
        transpiled_qc_meas = transpile(qc_meas, self.backend, initial_layout = [self.qubit])
        self.sched_meas = schedule(transpiled_qc_meas, backend) 
        
        # Default pi pulse
        qc_x = QuantumCircuit(1)
        qc_x.x(0)
        transpiled_qc_x = transpile(qc_x, self.backend, initial_layout = [self.qubit])
        sched_x = schedule(transpiled_qc_x, self.backend)
        
        pulse_info = sched_x.instructions[0][1].pulse
        self.drive_samples = pulse_info.duration
        self.drive_sigma = pulse_info.sigma
        
        if x01_amp == None:
            self.sched_x = sched_x
            self.default_qubit_freq = backend_defaults.qubit_freq_est[self.qubit]
        else:
            pi_pulse_01 = pulse_lib.gaussian(duration=self.drive_samples, amp=x01_amp, sigma=self.drive_sigma)
            self.sched_x = pulse.Schedule()
            self.sched_x |= pulse.Play(pi_pulse_01, DriveChannel(self.qubit))
            self.default_qubit_freq = x01_freq
            
        self.times = 50
    
    # To be updated
    def freq_EFfreq(self, amp):
        frequencies = np.linspace(self.default_qubit_freq -400e6, self.default_qubit_freq -300e6, self.times)
        spec = EFSpectroscopy(self.qubit, frequencies)
        spec.set_experiment_options(amp=amp)
        
        spec_data = spec.run(self.backend)
        spec_data.block_for_results()
        print(spec_data.analysis_results("f12"))   
        
        rabi_freq = spec_data.analysis_results("f12").value.value
        
        return rabi_freq, spec_data.figure(0)
    
    def freq(self, amp, freq_med):
        if freq_med == None:
            excited_sweep_freqs = self.default_qubit_freq + np.linspace(-400e6, -300e6, self.times)
        else:
            excited_sweep_freqs = freq_med + np.linspace(-20e6, 20e6, self.times)
            
        excited_freq_sweep_program = self.create_excited_freq_sweep_program(excited_sweep_freqs, amp=amp)
        
        excited_freq_sweep_job = self.backend.run(excited_freq_sweep_program)
        job_monitor(excited_freq_sweep_job)
        excited_freq_sweep_data = self.get_job_data(excited_freq_sweep_job, average=True)
        
        return excited_freq_sweep_data, excited_sweep_freqs
    
    def fitting_freq(self, excited_freq_sweep_data, excited_sweep_freqs, parameters_real, parameters_imag):
        GHz = 1e9
        
        # Real fit
        (excited_sweep_fit_params_real, 
         excited_sweep_i_fit_real) = self.fit_function_single(excited_sweep_freqs, np.real(excited_freq_sweep_data), 
                                           lambda x, A, q_freq, B, C: (A / np.pi) * (B / ((x - q_freq)**2 + B**2)) + C,
                                           parameters_real, # initial parameters for curve_fit
                                           ([-np.inf, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf])) # ranges of parameters

        # Imag fit
        (excited_sweep_fit_params_imag, 
         excited_sweep_i_fit_imag) = self.fit_function_single(excited_sweep_freqs, np.imag(excited_freq_sweep_data), 
                                           lambda x, A, q_freq, B, C: (A / np.pi) * (B / ((x - q_freq)**2 + B**2)) + C,
                                           parameters_imag, # initial parameters for curve_fit
                                           ([-np.inf, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf])) # ranges of parameters
        
        print(excited_sweep_fit_params_real, excited_sweep_fit_params_imag)
        
        # 注：シグナルの実部だけをプロットします
        plt.scatter(excited_sweep_freqs/GHz, np.real(excited_freq_sweep_data), color='black')
        plt.plot(excited_sweep_freqs/GHz, excited_sweep_i_fit_real, color='red')
        plt.xlim([min(excited_sweep_freqs/GHz)+0.01, max(excited_sweep_freqs/GHz)]) # ignore min point (is off)
        plt.xlabel("Frequency [GHz]", fontsize=15)
        plt.ylabel("Measured Signal (Real) [a.u.]", fontsize=15)
        plt.title("1->2 Frequency Sweep (first pass)", fontsize=15)
        plt.show()
        
        # 注：シグナルの虚部だけをプロットします
        plt.scatter(excited_sweep_freqs/GHz, np.imag(excited_freq_sweep_data), color='black')
        plt.plot(excited_sweep_freqs/GHz, excited_sweep_i_fit_imag, color='red')
        plt.xlim([min(excited_sweep_freqs/GHz)+0.01, max(excited_sweep_freqs/GHz)]) # ignore min point (is off)
        plt.xlabel("Frequency [GHz]", fontsize=15)
        plt.ylabel("Measured Signal (Imag) [a.u.]", fontsize=15)
        plt.title("1->2 Frequency Sweep (first pass)", fontsize=15)
        plt.show()
        
        return excited_sweep_fit_params_real[1], excited_sweep_fit_params_imag[1]    
        
        
    def create_excited_freq_sweep_program(self, freqs, amp):
        if len(freqs) > 75:
            raise ValueError("You can only run 75 schedules at a time.")

        base_12_pulse = pulse_lib.gaussian(duration=self.drive_samples, sigma=self.drive_sigma,
                                            amp=amp, name='base_12_pulse')
        schedules = []
        for jj, freq in enumerate(freqs):
            freq_sweep_12_pulse = self.apply_sideband(base_12_pulse, freq)
            schedule = pulse.Schedule(name="Frequency = {}".format(freq))
            schedule |= self.sched_x
            schedule |= pulse.Play(freq_sweep_12_pulse, DriveChannel(self.qubit)) << schedule.duration 
            schedule |= self.sched_meas << schedule.duration # 駆動パルスの後に測定をシフト

            schedules.append(schedule)

        excited_freq_sweep_program = assemble(schedules,backend=self.backend, meas_level=1, meas_return='avg', shots=self.shots,
                                              schedule_los=[{DriveChannel(self.qubit): self.default_qubit_freq}] * len(freqs))

        return excited_freq_sweep_program
        

    # To be updated
    def amp_EFRabi(self, rabi_freq):
        rabi = EFRabi(self.qubit)
        rabi.set_experiment_options(amplitudes=np.linspace(-1, 1, self.times), frequency_shift=rabi_freq)
        
        rabi_data = rabi.run(self.backend)
        rabi_data.block_for_results()
        print(rabi_data.block_for_results("rabi_state"))
        
        rabi_amp = rabi_data.analysis_results("rabi_rate_12").value.value
        
        return rabi_amp, rabi_data.figure(0)
    
    def amp(self, rabi_freq, amp_med):
        if amp_med == None:
            drive_amp_min = 0
            drive_amp_max = 0.5
            drive_amps = np.linspace(drive_amp_min, drive_amp_max, self.times)
        else:
            drive_amps = np.linspace(0, amp_med+0.1, self.times)

        # スケジュールの作成
        rabi_12_schedules = []

        # すべての駆動振幅をループします
        for ii, drive_amp in enumerate(drive_amps):

            base_12_pulse = pulse_lib.gaussian(duration=self.drive_samples,
                                               sigma=self.drive_sigma,
                                               amp=drive_amp,
                                               name='base_12_pulse')
            # 1->2の周波数においてサイドバンドを適用
            rabi_12_pulse = self.apply_sideband(base_12_pulse, rabi_freq)

            # スケジュールにコマンドを追加
            sched = pulse.Schedule(name='Rabi Experiment at drive amp = %s' % drive_amp)
            sched |= self.sched_x# 0->1
            sched |= pulse.Play(rabi_12_pulse, DriveChannel(self.qubit)) << sched.duration # 1->2のラビパルス
            sched |= self.sched_meas << sched.duration # 駆動パルスの後に測定をシフト

            rabi_12_schedules.append(sched)
            
        rabi_12_expt_program = assemble(rabi_12_schedules, backend=self.backend, meas_level=1, 
                                        meas_return='avg', shots=self.shots, 
                                        schedule_los=[{DriveChannel(self.qubit): self.default_qubit_freq}] * len(drive_amps))
        
        rabi_12_job = self.backend.run(rabi_12_expt_program)
        job_monitor(rabi_12_job)
        rabi_12_data = self.get_job_data(rabi_12_job, average=True)

        return rabi_12_data, drive_amps
    
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
        plt.title('Rabi Experiment (1->2)', fontsize=20)
        plt.show()
        
        plt.scatter(drive_amps, np.imag(rabi_12_data), color='black')
        plt.plot(drive_amps, rabi_12_y_fit_imag, color='red')
        plt.xlabel("Drive amp [a.u.]", fontsize=15)
        plt.ylabel("Measured signal (Imag) [a.u.]", fontsize=15)
        plt.title('Rabi Experiment (1->2)', fontsize=20)
        plt.show()
        
        return pi_amp_12_real, pi_amp_12_imag
    
    def check(self, pi_pulse_12):
        # 基底状態のスケジュール
        zero_schedule = pulse.Schedule(name="zero schedule")
        zero_schedule |= self.sched_meas

        # 励起状態のスケジュール
        one_schedule = pulse.Schedule(name="one schedule")
        one_schedule |= self.sched_x
        one_schedule |= self.sched_meas << one_schedule.duration

        # 励起状態のスケジュール
        two_schedule = pulse.Schedule(name="two schedule")
        two_schedule |= self.sched_x
        two_schedule |= pulse.Play(pi_pulse_12, DriveChannel(self.qubit)) << two_schedule.duration
        two_schedule |= self.sched_meas << two_schedule.duration
        
        # プログラムにスケジュールを組み込みます
        IQ_012_program = assemble([zero_schedule, one_schedule, two_schedule], backend=self.backend, meas_level=1,
                                   meas_return='single', shots=self.shots,
                                  schedule_los=[{DriveChannel(self.qubit): self.default_qubit_freq}] * 3)
        
        IQ_012_job = self.backend.run(IQ_012_program)
        job_monitor(IQ_012_job)
        
        IQ_012_data = self.get_job_data(IQ_012_job, average=False)
        
        return IQ_012_data
    
    def IQ_012_plot(self, IQ_012_data, x_min, x_max, y_min, y_max):
        zero_data = IQ_012_data[0]
        one_data = IQ_012_data[1]
        two_data = IQ_012_data[2]

        """0、1、2のIQ平面をプロットするための補助関数。引数としてプロットの制限を与えます。
        """
        # 0のデータは青でプロット
        plt.scatter(np.real(zero_data), np.imag(zero_data), 
                        s=5, cmap='viridis', c='blue', alpha=0.5, label=r'$|0\rangle$')
        # 1のデータは赤でプロット
        plt.scatter(np.real(one_data), np.imag(one_data), 
                        s=5, cmap='viridis', c='red', alpha=0.5, label=r'$|1\rangle$')
        # 2のデータは緑でプロット
        plt.scatter(np.real(two_data), np.imag(two_data), 
                        s=5, cmap='viridis', c='green', alpha=0.5, label=r'$|2\rangle$')

        # 0、1、2の状態の結果の平均を大きなドットでプロット
        mean_zero = np.mean(zero_data) # 実部と虚部それぞれの平均をとる
        mean_one = np.mean(one_data)
        mean_two = np.mean(two_data)
        plt.scatter(np.real(mean_zero), np.imag(mean_zero), 
                    s=200, cmap='viridis', c='black',alpha=1.0)
        plt.scatter(np.real(mean_one), np.imag(mean_one), 
                    s=200, cmap='viridis', c='black',alpha=1.0)
        plt.scatter(np.real(mean_two), np.imag(mean_two), 
                    s=200, cmap='viridis', c='black',alpha=1.0)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min,y_max)
        plt.legend()
        plt.ylabel('Imag [a.u.]', fontsize=15)
        plt.xlabel('Real [a.u.]', fontsize=15)
        plt.title("0-1-2 discrimination", fontsize=15)
        
    def discriminator(self, IQ_012_data):
        # IQベクトルを作成します（実部と虚部で構成されています）
        zero_data_reshaped = self.reshape_complex_vec(IQ_012_data[0])
        one_data_reshaped = self.reshape_complex_vec(IQ_012_data[1])  
        two_data_reshaped = self.reshape_complex_vec(IQ_012_data[2])  

        IQ_012_data_copy = np.concatenate((zero_data_reshaped, one_data_reshaped, two_data_reshaped))

        # （テスト用に）0と1と2の値が含まれたベクトルを構築します
        state_012 = np.zeros(self.shots) # 実験のショット数
        state_012 = np.concatenate((state_012, np.ones(self.shots)))
        state_012 = np.concatenate((state_012, 2*np.ones(self.shots)))

        # データをシャッフルして学習用セットとテスト用セットに分割します
        IQ_012_train, IQ_012_test, state_012_train, state_012_test = train_test_split(IQ_012_data_copy, state_012, test_size=0.5)

        # LDAを設定します
        LDA_012 = LinearDiscriminantAnalysis()
        LDA_012.fit(IQ_012_train, state_012_train)

        # 精度を計算します
        score_012 = LDA_012.score(IQ_012_test, state_012_test)
        print(score_012)
        
        return LDA_012
    
    # セパラトリックスを表示データの上にプロットします
    def separatrixPlot(self, lda, x_min, x_max, y_min, y_max):
        nx, ny = self.shots, self.shots

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
        Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z[:, 1].reshape(xx.shape)

        plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='black')
        
        
    def calibrated_x12(self, rabi_freq, pi_amp_12):
        pi_pulse_12 = pulse_lib.gaussian(duration=self.drive_samples, amp=pi_amp_12, sigma=self.drive_sigma, name='pi_pulse_12')
        pi_pulse_12 = self.apply_sideband(pi_pulse_12, rabi_freq)
        
        return pi_pulse_12
        
    
    def apply_sideband(self, pulse, freq):
        t_samples = np.linspace(0, self.dt*self.drive_samples, self.drive_samples)
        sine_pulse = np.sin(2*np.pi*(freq-self.default_qubit_freq)*t_samples) # no amp for the sine

        sideband_pulse = Waveform(np.multiply(np.real(pulse.samples), sine_pulse), name='sideband_pulse')

        return sideband_pulse 
    
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
    def baseline_remove(values):
        """Center data around 0."""
        return np.array(values) - np.mean(values)

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