import numpy as np

def panel_transfer_matrix(freq, d, eps_r=1.0, sigma=0.0, mu_r=1.0):
    freq = np.atleast_1d(np.asarray(freq, dtype=complex))
    omega = 2.0 * np.pi * freq

    eps_c = eps_r - 1j * sigma / omega
    gamma = 1j * omega * np.sqrt(mu_r * eps_c)
    eta = np.sqrt(mu_r / eps_c)

    gd = gamma * d
    ch = np.cosh(gd)
    sh = np.sinh(gd)

    Phi = np.zeros((len(freq), 2, 2), dtype=complex)
    Phi[:, 0, 0] = ch
    Phi[:, 0, 1] = eta * sh
    Phi[:, 1, 0] = sh / eta
    Phi[:, 1, 1] = ch
    return Phi


def stack_transfer_matrix(freq, layers):
    freq = np.atleast_1d(np.asarray(freq, dtype=complex))
    Phi_total = np.zeros((len(freq), 2, 2), dtype=complex)
    Phi_total[:, 0, 0] = 1.0
    Phi_total[:, 1, 1] = 1.0

    for layer in layers:
        Phi_i = panel_transfer_matrix(
            freq,
            d=layer['d'],
            eps_r=layer.get('eps_r', 1.0),
            sigma=layer.get('sigma', 0.0),
            mu_r=layer.get('mu_r', 1.0),
        )
        Phi_new = np.zeros_like(Phi_total)
        Phi_new[:, 0, 0] = Phi_total[:, 0, 0] * Phi_i[:, 0, 0] + Phi_total[:, 0, 1] * Phi_i[:, 1, 0]
        Phi_new[:, 0, 1] = Phi_total[:, 0, 0] * Phi_i[:, 0, 1] + Phi_total[:, 0, 1] * Phi_i[:, 1, 1]
        Phi_new[:, 1, 0] = Phi_total[:, 1, 0] * Phi_i[:, 0, 0] + Phi_total[:, 1, 1] * Phi_i[:, 1, 0]
        Phi_new[:, 1, 1] = Phi_total[:, 1, 0] * Phi_i[:, 0, 1] + Phi_total[:, 1, 1] * Phi_i[:, 1, 1]
        Phi_total = Phi_new

    return Phi_total


def RT_from_transfer_matrix(Phi):
    A = Phi[:, 0, 0]
    B = Phi[:, 0, 1]
    C = Phi[:, 1, 0]
    D = Phi[:, 1, 1]
    denom = A + B + C + D
    R = (A + B - C - D) / denom
    T = 2.0 / denom
    return R, T


def reflection_transmission(freq, d, eps_r=1.0, sigma=0.0, mu_r=1.0):
    Phi = panel_transfer_matrix(freq, d, eps_r, sigma, mu_r)
    return RT_from_transfer_matrix(Phi)


def reflection_transmission_stack(freq, layers):
    Phi = stack_transfer_matrix(freq, layers)
    return RT_from_transfer_matrix(Phi)
