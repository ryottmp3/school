# Homework - 13 Jan 2024
# Vibrations, Waves, and Optics -- PHYS 225
# H. Ryott Glayzer
# MIT License

"""
A 1-kg object released at position x_0 feels a force (in Newtons)
F(x) = -(x/10 + x^3/1000)
restoring it to equilibrium at x = 0.
Write computer program that returns numerical solution
for its position as a function of time for x_0 = 0.1, 1, 10, and 50 mm.
Plot x/x_0 as a function of time for several periods
with each curve overlaid on the same axis. Turn in program and plot.

---

mx`` = -kx -> x(t) = A*sin(ωt-ø_0) where ω=sqrt(k/m)
"""

import matplotlib as mpl
mpl.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True


def main():
    """
    normalized displacement: x(t)/x_0 = cos(t/sqrt(10))
    """
    pos = []
    time = []
    t = 0

    for i in range(1000):
        time.append(float(t))  # Add the time component to its list
        pos.append(float(np.cos(t)/np.sqrt(10)))  # add the pos component
        t += 0.02

    plt.figure()
    plt.title("Position vs. Time for the Weird Thing")
    plt.plot(
        time,
        pos,
        label=r"$\frac{x(t)}{x_0}=\cos{\frac{t}{\sqrt{10}}}$"
    )
    plt.xlabel("Time, sec")
    plt.ylabel("Position, mm")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()

