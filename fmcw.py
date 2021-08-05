import numpy as np
import matplotlib.pyplot as plt

c = 299792458  # velocity of light
f_0 = 77.7 * 1e9  # 77.7 GHz

phi_t = 0
A_t = 1
f_ramp = 425 * 1e6  # 425 MHz
T_ramp = 0.010
m_w = f_ramp / T_ramp

v = 50 / 3.6  # 50km/h
r = 100  # 100m distance

t = np.arange(0, 2 * T_ramp, 1e-8)

T_ramp1 = T_ramp*0.7
T_ramp2 = T_ramp*0.3
m_w1 = f_ramp/T_ramp1
m_w2 = -f_ramp/T_ramp2


def f_t(t):
    r1 = f_0 + m_w1 * (t % (T_ramp1 + T_ramp2))
    r1[(t % (T_ramp1 + T_ramp2)) > T_ramp1] = 0

    r2 = f_0 + m_w1 * T_ramp1 + m_w2 * ((t - T_ramp1) % (T_ramp1 + T_ramp2))
    r2[(t % (T_ramp1 + T_ramp2)) <= T_ramp1] = 0

    return r1 + r2


def f_r(t):
    r1 = f_0 + m_w1 * (t % (T_ramp1 + T_ramp2) - 2 * r / c) - 2 * v * f_0 / c
    r1[(t % (T_ramp1 + T_ramp2)) > T_ramp1] = 0

    r2 = f_0 + m_w1 * T_ramp1 + m_w2 * ((t - T_ramp1) % (T_ramp1 + T_ramp2) - 2 * r / c) - 2 * v * f_0 / c
    r2[(t % (T_ramp1 + T_ramp2)) <= T_ramp1] = 0
    return r1 + r2


def main():
    t_op = T_ramp
    plt.figure(figsize=(10, 5))
    plt.plot(t, f_t(t), label="$f_t$")
    plt.plot(t, f_r(t), label="$f_r$")
    plt.legend()
    plt.xlim([t_op - T_ramp / 5000, t_op + T_ramp / 5000])
    plt.ylim([f_0 - 100e3, f_0 + 600e3])
    plt.xlabel("t [s]")
    plt.ylabel("frequency [Hz]")
    plt.title("FMCW ramps with different $m_{w,1} \\neq m_{w,2}$")
    plt.grid()
    # plt.show()

    t_f1 = np.asarray([T_ramp * 0.1])  # beginning of rising ramp
    t_f2 = np.asarray([T_ramp * 0.8])  # beginning of falling ramp
    delta_f_1 = f_t(t_f1) - f_r(t_f1)
    delta_f_2 = f_t(t_f2) - f_r(t_f2)

    A = np.asarray([[2 * m_w1 / c, 2 * f_0 / c], [2 * m_w2 / c, 2 * f_0 / c]])
    Y = np.asarray([delta_f_1, delta_f_2])

    x = np.linalg.solve(A, Y).flatten()
    print("range r:", x[0])
    print("velocity v:", x[1] * 3.6)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
