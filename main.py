import numpy as np
import matplotlib.pyplot as plt
import tclab
from scipy.integrate import odeint
import time


# FOPDT model
Kp = 0.5  # [degC/%]
tauP = 120.0  # [s]
thetaP = 10  # [s]
Tss = 23  # [degC]  (ambient temperature)
Qss = 0  # [% heater]


def kelvin(c):
    return c + 273.15

def celsius(k):
    return k - 273.15


# define energy balance model
def heat(x,t,Q):
    # Parameters
    Ta = kelvin(23)   # K
    U = 10.0           # W/m^2-K
    m = 4.0/1000.0     # kg
    Cp = 0.5 * 1000.0  # J/kg-K
    A = 12.0 / 100.0**2  # Area in m^2
    alpha = 0.01       # W / % heater
    eps = 0.9          # Emissivity
    sigma = 5.67e-8    # Stefan-Boltzman

    # Temperature State
    T = x[0]

    # Nonlinear Energy Balance
    dTdt = (1.0/(m*Cp)) * (U*A*(Ta-T) + eps * sigma * A * (Ta**4 - T**4) + alpha*Q)
    return dTdt


def save_txt(t, u1, u2, y1, y2, sp1, sp2):
    data = np.vstack((t, u1, u2, y1, y2, sp1, sp2))
    data = data.T
    top = ('Time (s), Heater 1 (%), Heater 2 (%), Temperature 1 (decC) ,'
           'Temperature 2 (degC), Set Point 1 (degC), Set Point 2 (degC)')
    np.savetxt('data.txt', data, delimiter=',', header=top, comments=' ')


lab = tclab.TCLab()

print('LED On')
lab.LED(100)

run_time = 10.0  # [minutes]

loops = int(60 * run_time)
t_arr = np.zeros(loops)
Tsp1_arr = np.ones(loops) * 23.0
T1_arr  = np.ones(loops) * lab.T1
Tsp2_arr = np.ones(loops) * 23.0
T2_arr = np.ones(loops) * lab.T2
Tp_arr = np.ones(loops) * lab.T1
error_eb = np.zeros(loops)
Tpl_arr = np.ones(loops) * lab.T1
error_fopdt = np.zeros(loops)

Q1_arr = np.ones(loops) * 0.0
Q2_arr = np.ones(loops) * 0.0
Q1_arr[10:110] = 50.0
Q1_arr[200:300] = 90.0
Q1_arr[400:500] = 70.0

plt.figure()
plt.ion()
plt.show()

start_time = time.time()
prev_time = start_time

try:
    for i in range(1, loops):
        sleep_max = 1.0
        sleep = sleep_max - (time.time() - prev_time)
        if sleep >= 0.01:
            time.sleep(sleep - 0.01)
        else:
            time.sleep(0.1)

        t = time.time()
        dt = t - prev_time
        prev_time = t
        t_arr[i] = t - start_time

        T1_arr[i] = lab.T1
        T2_arr[i] = lab.T2

        # Simulate
        Tnext = odeint(heat, kelvin(Tp_arr[i-1]), [0, dt], args=(Q1_arr[i-1], ))
        Tp_arr[i] = celsius(Tnext[1][0])
        error_eb[i] = error_eb[i-1] + abs(Tp_arr[i] - T1_arr[i])

        # Simulate FOPDT
        z = np.exp(-dt / tauP)
        Tpl_arr[i] = ((Tpl_arr[i-1] - Tss) * z
                     + (Q1_arr[max(0, i-int(thetaP)-1)] - Qss) * (1 - z) * Kp
                     + Tss)
        error_fopdt[i] = error_fopdt[i-1] + abs(Tpl_arr[i] - T1_arr[i])

        lab.Q1(Q1_arr[i])
        lab.Q2(Q2_arr[i])

        print(f'{t_arr[i]} {Q1_arr[i]} {Q2_arr[i]} {T1_arr[i]} {T2_arr[i]}')

        plt.clf()
        ax = plt.subplot(3, 1, 1)
        ax.grid()
        plt.plot(t_arr[:i], T1_arr[:i], 'ro', label='T1 measured')
        plt.plot(t_arr[:i], Tp_arr[:i], 'k-', label='T1 energy balance')
        plt.plot(t_arr[:i], Tpl_arr[:i], 'g:', label='T1 FOPDT')
        plt.plot(t_arr[:i], T2_arr[:i], 'bx', label='T2 measured')
        plt.ylabel('Temperature (degC)')
        plt.legend(loc=2)

        ax = plt.subplot(3, 1, 2)
        ax.grid()
        plt.plot(t_arr[:i], error_eb[:i], 'k-', label='Energy Balance')
        plt.plot(t_arr[:i], error_fopdt[:i], 'g:', label='Linear')
        plt.ylabel('Cumulative Error')
        plt.legend(loc='best')

        ax = plt.subplot(3, 1, 3)
        ax.grid()
        plt.plot(t_arr[:i], Q1_arr[:i], 'r-', label='Q1')
        plt.plot(t_arr[:i], Q2_arr[:i], 'b:', label='Q2')
        plt.ylabel('Heaters')
        plt.xlabel('Time (s)')
        plt.legend(loc='best')

        plt.draw()
        plt.pause(0.05)

    # Shutdown
    print('Shutting Down')
    lab.Q1(0)
    lab.Q2(0)
    save_txt(t_arr, Q1_arr, Q2_arr, T1_arr, T2_arr, Tsp1_arr, Tsp2_arr)
    plt.savefig('test_models.png')

except KeyboardInterrupt:
    lab.Q1(0)
    lab.Q2(0)
    print('Shutting down')
    lab.close()
    save_txt(t_arr[:i], Q1_arr[:i], Q2_arr[:i], T1_arr[:i], T2_arr[:i],
             Tsp1_arr[:i], Tsp2_arr[:i])
    plt.savefig('test_models.png')

except:
    print('Error: shutting down')
    lab.Q1(0)
    lab.Q2(0)
    lab.close()
    save_txt(t_arr[:i], Q1_arr[:i], Q2_arr[:i], T1_arr[:i], T2_arr[:i],
             Tsp1_arr[:i], Tsp2_arr[:i])
    plt.savefig('test_models.png')
    raise
