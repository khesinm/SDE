import matplotlib.pyplot as plt
import numpy as np
import sdeint

g = 0.3
ω = np.pi/3
β = 0.05
Γ = 0.1
ϕ = 0.000001

num_periods = 320
T = int(2*np.pi/ω)
y0 = np.array([0, 0.1, 0.1, 0.001])
spacing = 120000
tspan = np.linspace(0, num_periods*T, spacing)
samp_freq = int(spacing/num_periods)

# main function (including dissipation)
def f(y, t):
    x = y[0]
    p = y[1]
    χ = y[2]
    Π = y[3]

    f0 = p
    f1 = x - β ** 2 * x ** 3 + g / β * np.cos(ω * t) - 2 * Γ * p - 3*x*β**2*χ**2
    f2 = Π + Γ * ((χ - χ ** 3 + χ * Π ** 2 - 1 / (4 * χ)) * np.cos(2*ϕ) - Π * (-1 + 2*χ**2) * np.sin(2*ϕ) + χ - χ**3 - χ*Π**2 + 1/(4*χ))
    f3 = χ * (1 - 3 * β ** 2 * (x ** 2 + χ ** 2)) + 1 / (4 * χ ** 3) + Γ * ((Π ** 3 - Π + 3 * Π / (4 * χ ** 2) - Π * χ ** 2) * np.cos(2 * ϕ) - (-1 / (4 * χ ** 3) + 1 / χ - χ + 2 * χ * Π ** 2) * np.sin(2 * ϕ) + (-Π ** 3 - Π - 3 * Π / (4 * χ ** 2) - Π * χ ** 2))
    return np.array([f0, f1, f2, f3])

# noise function
def N(y, t):
    x = y[0]
    p = y[1]
    χ = y[2]
    Π = y[3]
    return np.diag([
        np.sqrt(Γ) * ((2 * (χ ** 2 - 1 / 2)) * np.cos(ϕ) + 2 * χ * Π * np.sin(ϕ)),
        np.sqrt(Γ) * (2 * χ * Π * np.cos(ϕ) + 2 * (-1 / 2 + Π ** 2 + 1 / (4 * χ ** 2)) * np.sin(ϕ)),
        0,
        0])

# graphing the Poincare section
xs = sdeint.itoint(f, N, y0, tspan)
x = [xs[samp_freq*i][0] for i in range(int(len(xs)/samp_freq))]
y = [xs[samp_freq*i][1] for i in range(int(len(xs)/samp_freq))]

plt.scatter(x,y,s=.4)
plt.xlabel("x")
plt.ylabel("p")
plt.show()