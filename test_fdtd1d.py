import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pytest
from fdtd1d import FDTD1D

def gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0)/sigma)**2)

C = 1.0

def test_example():
    # Given...
    num1 = 1
    num2 = 1

    # When...
    result = num1 + num2

    # Expect...
    assert result == 2

def test_fdtd_solves_one_wave():
    x = np.linspace(-1, 1, 201)
    x0 = 0.0
    sigma = 0.05
    initial_e = gaussian(x, x0, sigma)
    fdtd = FDTD1D(x)
    fdtd.load_initial_field(initial_e)

    t_final = 1.8
    n_frames = 180
    dt_frame = t_final / n_frames

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.7, 0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("E(x, t)")
    ax.set_title("Propagación de onda FDTD 1D")
    line, = ax.plot([], [], lw=2)
    time_text = ax.text(0.02, 0.92, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def update(frame):
        fdtd.run_until((frame + 1) * dt_frame)
        line.set_data(x, fdtd.get_e())
        time_text.set_text(f"t = {fdtd.t:.3f}")
        return line, time_text

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=30, blit=True
    )
    plt.tight_layout()
    plt.show()

    t_check = 0.2
    fdtd2 = FDTD1D(x)
    fdtd2.load_initial_field(gaussian(x, x0, sigma))
    fdtd2.run_until(t_check)
    e_solved = fdtd2.get_e()
    e_expected = 0.5 * gaussian(x, -t_check*C, sigma) \
               + 0.5 * gaussian(x,  t_check*C, sigma)
    assert np.allclose(e_solved, e_expected)

if __name__ == "__main__":
    pytest.main([__file__])
