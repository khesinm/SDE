import matplotlib.pyplot as plt
import numpy as np
import sdeint
from scipy.integrate import odeint

# 06/23/2022: removed oscillating term from U1 and divided by minimum x value
# 06/24/2022: updated formulae for f and N

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

# building individual result arrays for each variable
result = sdeint.itoint(f, N, y0, tspan)

fullArray= [[],[],[],[]]
for i in range (len(result)):
    fullArray[0].append(result[i][0])
    fullArray[1].append(result[i][1])
    fullArray[2].append(result[i][2])
    fullArray[3].append(result[i][3])

x_series = fullArray[0]
χ_series = fullArray[2]

# --------------- energy functions-------------------
# classical energy function
def U1(x):
    return (-1/2 * x**2 + 1/4 * β**2*x**4)*β

# quantum energy function
def U2(χ):
    return 3/4 * β**2*χ**4 - 1/2 * χ**2 + 1/(8*χ**2)

# coupled energy function
def U12(x, χ):
    return 3/2 * β**2*x**2*χ**2

# constructing arrays of energies for different values of x and χ
'''
U1s = []
for i in range(len(tspan)):
    x = x_series[i]
    U1s.append(U1(x))
'''
'''
U2s = []
for i in range(len(tspan)):
    χ = χ_series[i]
    U2s.append(U2(χ))
'''

U12s = []
for i in range(len(tspan)):
    x = x_series[i]
    χ = χ_series[i]
    U12s.append(U12(x,χ))

# plotting energes
plt.scatter(tspan, U12s, s=.4)
plt.xlabel("t")
plt.ylabel("U12")
plt.show()
