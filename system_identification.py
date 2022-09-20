import time

import numpy as np
import matplotlib.pyplot as plt
import tclab
from scipy.optimize import minimize



def foptd(t, K, tau, time_delay):
    """Compute y(t) for a unit step input from 0->1 for a function:
    y(t) = y0 + K(u_inf - u0) * (1 - exp(-(t - time_delay) / tau)

    assumes y0 = 0, u0 = 0 and u_inf = 1
    """
    time_delay = max(0, time_delay)
    tau = max(0, tau)
    return np.array([K * (1 - np.exp(-(t - time_delay) / tau)) if t >= time_delay else 0 for t in t])


def error(X, t, y):
    K, tau, time_delay = X
    z = foptd(t, K, tau, time_delay)
    iae = sum(abs(z - y)) * (max(t) - min(t)) / len(t)
    return iae


if __name__ == '__main__':
    lab = tclab.TCLab()

    # Get initial temperature
    print('Measuring initial temperature')
    initialize_duration = 3  # [s]
    wait_time = 0.25  # [s]
    T_arr = np.zeros(int(initialize_duration / wait_time))
    for i in range(T_arr.shape[0]):
        T = lab.T1
        T_arr[i] = T
        print(f'T = {T}')
        time.sleep(wait_time)
    T0 = np.mean(T_arr)

    # Step response
    step_size = 40.0  # [%]
    duration = 15 * 60 # [s]
    sample_period = 0.25  # [s]
    n_samples = int(duration / sample_period)
    T_arr = np.zeros(n_samples)
    t_arr = np.zeros(n_samples)
    start_time = time.time()
    prev_cycle_time = start_time
    print(f'Step response test: heater @ {step_size}% for {duration} s and {n_samples} samples')
    try:
        # NOTE: tclab calls are really slow (~0.1s)
        lab.Q1(step_size)
        for i in range(n_samples):
            sleep_time = sample_period - (time.time() - prev_cycle_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            prev_cycle_time = time.time()

            t = time.time() - start_time
            t_arr[i] = t
            # Scale the measured temperature to a unit step input
            T = lab.T1
            T_arr[i] = (T - T0) / step_size
            print(f't = {round(t, 2)}: heater = {step_size}  T = {T}')

        lab.Q1(0)

    except:
        lab.Q1(0)
        raise

    # Solve for model constants
    X = [0.005, 10, 3]
    K, tau, time_delay = minimize(error, X, args=(t_arr, T_arr)).x

    print(f'Model params: K = {K}, tau = {tau}, time_delay = {time_delay}')

    # Plot prediction & actual temperatures
    T_pred_arr = foptd(t_arr, K, tau, time_delay)
    plt.plot(t_arr, T_arr * step_size + T0, t_arr, T_pred_arr * step_size + T0)
    plt.legend(['measured', 'FOPTD'])
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [degC]')
    plt.show()




