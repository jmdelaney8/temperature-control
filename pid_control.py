from simple_pid import PID
import tclab
import matplotlib.pyplot as plt
import time

"""
Model paramaters:
foptd
K = 0.6625
tau = 149.107
time_delay = 14.927
"""


if __name__ == '__main__':
    """
    P Tuning:
    Kc = 5
    
    PI Tuning moderate: 
    tau_i = max(tau, 8 * time_delay) == tau
    Kc = 1.37
    P_gain = Kc
    I_gain = Kc / tau_i
    
    """
    setpoint = 40  # [degC]
    lab = tclab.TCLab()
    pid = PID(1.37, 0.0092, 0, sample_time=1.0, setpoint=setpoint,
              output_limits=(0, 100))

    t_arr = []
    setpoint_arr = []
    temp_arr = []
    control_arr = []
    plt.figure()
    plt.ion()
    plt.show()

    start_time = time.time()
    cycle_start = start_time
    i = 0

    while True:
        cycle_time = time.time() - cycle_start
        sleep_time = pid.sample_time - cycle_time
        cycle_start = time.time()
        print(f'cycle time = {cycle_time} | sleep time = {sleep_time}')
        if sleep_time > 0:
            time.sleep(sleep_time)

        temp = lab.T1
        control = pid(temp)
        control_arr.append(control)
        lab.Q1(control)
        t_arr.append(time.time() - start_time)
        temp_arr.append(temp)
        setpoint_arr.append(setpoint)
        if i % 1 == 0:
            print(f't = {t_arr[-1]}: error = {temp - setpoint}, control = {control}')
            plt.clf()
            ax = plt.subplot(2, 1, 1)
            plt.plot(t_arr, temp_arr, t_arr, setpoint_arr)
            plt.legend(['temp', 'setpoint'])
            plt.ylabel('Temperature [degC]')

            ax = plt.subplot(2, 1, 2)
            plt.plot(t_arr, control_arr)
            plt.ylabel('Heater [%]')
            plt.xlabel('Time [s]')

            plt.draw()
            plt.pause(0.05)

        i += 1
