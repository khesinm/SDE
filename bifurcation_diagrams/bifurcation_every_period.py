import matplotlib.pyplot as plt
import numpy as np
import sdeint
import math
import warnings
warnings.filterwarnings("ignore")
g = 0.3
ω = np.pi/3
β = 0.10
Γ = 0.1
ϕ = 0

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

# creates 1-dimensional poincare section (sqrt(x^2+p^2) or sqrt(χ^2+Π^2) - effectively a Pythagorean distance)
def bifurcation(g, ω, β, Γ, ϕ, y0, t, periods, acc):
    β, g, ω, Γ, ϕ, y0, t, periods, acc
    xs = sdeint.itoint(f, N, y0, t)
    x = [xs[samp_freq * i][0] for i in range(int(len(xs) / samp_freq))]
    y = [xs[samp_freq * i][1] for i in range(int(len(xs) / samp_freq))]
    distances = [math.sqrt((x[i]) ** 2 + (y[i]) ** 2)
                 for i in range(len(x))]
    return distances

# bifurcating different parameters (determines which parameter is being changed and appears on the x-axis)
def bifurcateβ(g, ω, Γ, ϕ, y0, t, periods, acc, points):
    βs = np.linspace(0, 1, points)
    results = []
    t_new = [t[i] for i in range(int(len(t)*.0025), len(t))]
    for i in range(len(βs)):
        j = βs[i]
        try:
            results.append([j, bifurcation(j, g, ω, Γ, ϕ, y0, t_new, periods, acc)])
        except:
            print('error')
    for result in results:
        plt.scatter([result[0]] * len(result[1]), [result[1][i] for i in range(1, len(result[1]))],
                    color='blue', s=.4)

    plt.xlabel('β', fontsize=10)
    plt.ylabel('d', fontsize=10)
    plt.tick_params(labelsize=10)
    plt.title('The bifurcation diagram (beta)')
    plt.show()

def bifurcateϕ(g, ω, β, Γ, y0, t, periods, acc, points):
    ϕs = np.linspace(0, math.pi / 2, points)
    t_new = [t[i] for i in range(int(len(t)*.0025), len(t))]
    results = []
    for i in range(len(ϕs)):
        j = ϕs[i]
        try:
            print(j, bifurcation(g, ω, β, j, Γ, y0, t_new, periods, acc))
            results.append([j, bifurcation(g, ω, β, j, Γ, y0, t_new, periods, acc)])
        except:
            print('error')
    for result in results:
        new_result = [result[1][i] for i in range(2, len(result[1]))]
        print(new_result)
        plt.scatter([result[0]] * len(new_result), new_result,
                    color='blue', s=.4)

    plt.xlabel('ϕ', fontsize=10)
    plt.ylabel('d', fontsize=10)
    plt.tick_params(labelsize=10)
    plt.title('The bifurcation diagram (phi)')
    plt.show()

def bifurcateΓ(g, ω, β, ϕ, y0, t, periods, acc, points):
    Γs = np.linspace(0, math.pi / 2, points)
    results = []
    for i in range(len(Γs)):
        j = Γs[i]
        try:
            results.append([j, bifurcation(g, ω, β, ϕ, j, y0, t, periods, acc)])
        except:
            print('error')

    for result in results:
        plt.scatter([result[0]] * len(result[1]), result[1],
                    color='blue', s=.4)

    plt.xlabel('Γ', fontsize=10)
    plt.ylabel('d', fontsize=10)
    plt.tick_params(labelsize=10)
    plt.title('The bifurcation diagram (gamma)')
    plt.show()

acc = 100000
points = 40
periods = 50
T = 2 * np.pi / ω
y0 = np.array([0, 0.1, 0.01, 0.001])
spacing = periods*acc
samp_freq = int(spacing/periods)
t = np.linspace(0, periods*T, spacing)
xs = sdeint.itoint(f, N, y0, t)
bifurcateϕ(g, ω, β, Γ, y0, t, periods, acc, points)
y0 = np.array([0, 0.1, 0.1, 0.001])