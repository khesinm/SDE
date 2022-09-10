import matplotlib.pyplot as plt
import numpy as np
import sdeint
from scipy.integrate import odeint

g = 0.3
ω = 1
β = 0.01
Γ = 0.1
ϕ = 0

y0 = np.array([0, 0.1, 0.1, 0.001])
tspan = np.linspace(0, 1000., 200000)

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

result = sdeint.itoint(f, N, y0, tspan)

# plotting phase space diagram
fullArray = [[],[],[],[]]
for i in range (len(result)):
    fullArray[0].append(result[i][0])
    fullArray[1].append(result[i][1])
    fullArray[2].append(result[i][2])
    fullArray[3].append(result[i][3])

plt.scatter(fullArray[0],fullArray[1], s=.4)
plt.xlabel("x")
plt.ylabel("p")
plt.show()