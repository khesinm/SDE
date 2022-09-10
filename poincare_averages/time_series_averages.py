import matplotlib.pyplot as plt
import numpy as np
import sdeint

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
    f1 = -β ** 2 * x ** 3 + (1 - 3 * β ** 2 * χ ** 2) * x - 2 * Γ * p + g / β * np.cos(ω * t)
    f2 = 2 * (Π + Γ * (χ - χ ** 3 + χ * Π ** 2 - 1 / (4 * χ))) + Γ * (χ - χ**3 + χ*Π**2 - 1/(4*χ)) * np.cos(2*ϕ)
    f3 = χ * (1 - 3 * β ** 2 * (x ** 2 + χ ** 2)) + 1 / (4 * χ ** 3) - Γ * (Π + Π * χ ** 2) + Γ * ((Π ** 3 - Π + 3 * Π / (4 * χ ** 2) - Π * χ ** 2) * np.cos(2 * ϕ)
             - (-1 / (4 * χ ** 3) + 1 / χ - χ + 2 * χ * Π ** 2) * np.sin(2 * ϕ)
             + (-Π ** 3 - Π - 3 * Π / (4 * χ ** 2) - Π * χ ** 2))
    return np.array([f0, f1, f2, f3])

# noise function
def N(y, t):
    x = y[0]
    p = y[1]
    χ = y[2]
    Π = y[3]
    return np.diag([
        2 * np.sqrt(Γ) * (χ ** 2 - 1 / 2) * np.cos(ϕ) + 2 * np.sqrt(Γ) * χ * Π * np.sin(ϕ),
        2 * np.sqrt(Γ) * χ * Π * np.cos(ϕ) + 2 * np.sqrt(Γ) * (-1 / 2 + Π ** 2 + 1 / (4 * χ ** 2)) * np.sin(ϕ),
        0,
        0])

# calculating the trajectories
num_xs = 160
xs = []
for i in range(num_xs):
    total = sdeint.itoint(f, N, y0, tspan)
    xs.append(total)

# taking the averages
averages = [[], [], [], []]
total0 = 0
total1 = 0
total2 = 0
total3 = 0
for i in range(len(total)):
    total0 = 0
    total1 = 0
    total2 = 0
    total3 = 0
    for x in xs:
        total0 += x[i][0]
        total1 += x[i][1]
        total2 += x[i][2]
        total3 += x[i][3]
    average0 = (1 / len(xs)) * total0
    average1 = (1 / len(xs)) * total1
    average2 = (1 / len(xs)) * total2
    average3 = (1 / len(xs)) * total3
    averages[0].append(average0)
    averages[1].append(average1)
    averages[2].append(average2)
    averages[3].append(average3)

# plotting the time series
plt.scatter(tspan, averages[0], s=.2)
plt.xlabel("t")
plt.ylabel("χ")
plt.show()