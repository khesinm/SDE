import matplotlib.pyplot as plt
import numpy as np
import sdeint
from scipy.integrate import odeint

g = 0.3
ω = np.pi/3
β = 0.01
Γ = 0.05
ϕ = 0

num_periods = 50
T = int(2*np.pi/ω)
y0 = np.array([0, 0.1, 0.1, 0.001])
spacing = num_periods*20000
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

# calculating the trajectories
num_xs = 40
xs = []

# calculating the trajectories
for i in range(num_xs):
    total = sdeint.itoint(f, N, y0, tspan)
    xs.append(total)

# taking the averages
averages = [[], [], [], []]
for i in range(len(tspan)):
    total0 = 0
    total1 = 0
    total2 = 0
    total3 = 0
    for x in xs:
        total0 += x[i][0]
        total1 += x[i][1]
        total2 += x[i][2]
        total3 += x[i][3]
    average0 = (1/len(xs)) * total0
    average1 = (1/len(xs)) * total1
    average2 = (1/len(xs)) * total2
    average3 = (1/len(xs)) * total3
    averages[0].append(average0)
    averages[1].append(average1)
    averages[2].append(average2)
    averages[3].append(average3)

# plotting poincare sections
x = [averages[0][samp_freq*i] for i in range(int(len(averages[2])/samp_freq))]
y = [averages[1][samp_freq*i] for i in range(int(len(averages[3])/samp_freq))]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x, y, s=.4)
plt.xlabel("x")
plt.ylabel("p")
plt.show()
# plt.savefig('poincare.png')