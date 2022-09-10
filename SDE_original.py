# original pull from github (Yusuf's code)
# import numba
import matplotlib.pyplot as plt
import numpy as np
import sdeint
from scipy.integrate import odeint
import math
import concurrent.futures
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import threading
from queue import Queue



class oscillator:

    def solve(self, g, ω, β, Γ, ϕ, y0, t, periods, acc):
        cosϕ = math.cos(ϕ)
        cos2ϕ = math.cos(2*ϕ)
        sinϕ = math.sin(ϕ)
        sin2ϕ = math.cos(2*ϕ)
        sqrtΓ = math.sqrt(Γ)

        def f(y, t):
            x = y[0]
            p = y[1]
            χ = y[2]
            Π = y[3]

            f0 = p
            f1 = -β**2 * x**3 + (1-3*β**2*χ**2) * x - 2 * \
                Γ*p + g/β * math.cos(ω*t)
            f2 = Π + Γ*((χ-χ**3+χ*Π**2-1/(4*χ))*math.cos(2*ϕ) - Π*(-1+2*χ**2)
                        * math.sin(2*ϕ) + χ - χ**3 - χ*Π**2 + 1/(4*χ))
            f3 = χ*(1-3*β**2*(x**2 + χ**2)) + 1/(4*χ**3) + Γ*((Π**3 - Π + 3*Π/(4*χ**2) - Π*χ**2) * math.cos(2*ϕ)
                                                              - (-1/(4*χ**2)+1/χ-χ+2*χ*Π**2)*math.sin(2*ϕ)
                                                              + (-Π**3 - Π - 3*Π/(4*χ**2)-Π*χ**2))
            return np.array([f0, f1, f2, f3])

        def W(y, t):
            x = y[0]
            p = y[1]
            χ = y[2]
            Π = y[3]
            return np.diag([
                sqrtΓ*(2*(χ ** 2 - 1/2)*math.cos(ϕ) + 2 * χ*Π*math.sin(ϕ)),
                sqrtΓ * (2 * (1/(4*χ**2) + Π**2 - 1/2)*math.sin(ϕ)+2*χ*Π*math.cos(ϕ)),
                0,
                0])
            # return np.diag([
            #     sqrtΓ*(2*(χ ** 2 - 1/2)*cosϕ + 2 * χ*Π*sinϕ),
            #     sqrtΓ * (2 * (1/(4*χ**2) + Π**2 - 1/2)*sinϕ+2*χ*Π*cosϕ),
            #     0,
            #     0])

        T = 2*np.pi/ω
        xs = sdeint.itoint(f, W, y0, t)

        x = [β*xs[acc*i, 0] for i in range(periods)]
        y = [β*xs[acc*i, 1] for i in range(periods)]
        plt.scatter(x, y, color='blue', s=periods/(acc*10))
        plt.xlabel('x', fontsize=10)
        plt.ylabel('y', fontsize=10)
        plt.tick_params(labelsize=10)
        plt.title('The Poincare section')

    def bifurcation(self, arr):
        β, g, ω, Γ, ϕ, y0, t, periods, acc = arr
        cosϕ = math.cos(ϕ)
        cos2ϕ = math.cos(2*ϕ)
        sinϕ = math.sin(ϕ)
        sin2ϕ = math.cos(2*ϕ)
        sqrtΓ = math.sqrt(Γ)
        def f(y, t):
            x = y[0]
            p = y[1]
            χ = y[2]
            Π = y[3]

            f0 = p
            f1 = -β**2 * x**3 + (1-3*β**2*χ**2) * x - 2 * \
                Γ*p + g/β * math.cos(ω*t)
            f2 = Π + Γ*((χ-χ**3+χ*Π**2-1/(4*χ))*math.cos(2*ϕ) - Π*(-1+2*χ**2)
                        * math.sin(2*ϕ) + χ - χ**3 - χ*Π**2 + 1/(4*χ))
            f3 = χ*(1-3*β**2*(x**2 + χ**2)) + 1/(4*χ**3) + Γ*((Π**3 - Π + 3*Π/(4*χ**2) - Π*χ**2) * math.cos(2*ϕ)
                                                              - (-1/(4*χ**2)+1/χ-χ+2*χ*Π**2)*math.sin(2*ϕ) + (-Π**3 - Π - 3*Π/(4*χ**2)-Π*χ**2))
            return np.array([f0, f1, f2, f3])

        def W(y, t):
            x = y[0]
            p = y[1]
            χ = y[2]
            Π = y[3]
            return np.diag([
                sqrtΓ*(2*(χ ** 2 - 1/2)*math.cos(ϕ) + 2 * χ*Π*math.sin(ϕ)),
                sqrtΓ * (2 * (1/(4*χ**2) + Π**2 - 1/2)*math.sin(ϕ)+2*χ*Π*math.cos(ϕ)),
                0,
                0])

        T = 2*np.pi/ω
        xs = sdeint.itoint(f, W, y0, t)
        x = [β*xs[acc*i, 0] for i in range(periods)]
        y = [β*xs[acc*i, 1] for i in range(periods)]

        distances = [math.sqrt((β*x[i])**2+(β*y[i])**2)
                     for i in range(len(x))]
        # return [β, distances]
        return [ϕ, distances[int(len(distances)/10):]]

    def bifurcateβ(self, g, ω, Γ, ϕ, y0, t, periods, acc, points):
        βs = np.linspace(0.1, 1, points)
        callinglist = np.array(
            [[βs[0], g, ω, Γ, ϕ, y0, t, periods, acc]], dtype=object)
        for i in βs:
            callinglist = np.append(
                callinglist, np.array([[i, g, ω, Γ, ϕ, y0, t, periods, acc]], dtype=object), axis=0)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self.bifurcation, callinglist)
        print("--- %s seconds ---" % (time.time() - start_time))

        for result in results:
            plt.scatter([result[0]]*len(result[1]), result[1],
                        color='blue', s=periods/(acc*100), alpha=.45)

        plt.xlabel('β/β', fontsize=10)
        plt.ylabel('d/β', fontsize=10)
        plt.tick_params(labelsize=10)
        plt.title('The bifurcation diagram')

    def bifurcateϕ(self, g, ω, β, Γ, y0, t, periods, acc, points):
        ϕs = np.linspace(0.0001, math.pi/2, points)
        callinglist = np.array(
            [[β, g, ω, Γ, ϕs[0], y0, t, periods, acc]], dtype=object)
        for i in ϕs:
            callinglist = np.append(
                callinglist, np.array([[β, g, ω, Γ, i, y0, t, periods, acc]], dtype=object), axis=0)


        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self.bifurcation, callinglist)

        for result in results:
            plt.scatter([result[0]]*len(result[1]), result[1],
                        color='blue', s=periods/(acc*300), alpha=.6)

        plt.xlabel('β/β', fontsize=10)
        plt.ylabel('d/β', fontsize=10)
        plt.tick_params(labelsize=10)
        plt.title('The bifurcation diagram')


if __name__ == '__main__':
    g = 0.3
    ω = 1
    β = 0.03901
    Γ = 0.110
    ϕ = 0.5
    periods = 500
    acc = 150

    y0 = np.array([0, 0.1, 0.1, 0.001])
    T = 2*np.pi/ω
    t = np.linspace(0, periods*T, acc*periods)
    osc = oscillator()
    osc2 = oscillator()
    points = 10
    # osc.bifurcateβ(g, ω, Γ, ϕ, y0, t, periods, acc, points)

    osc.bifurcateϕ(g, ω, β, Γ, y0, t, periods, acc, points)