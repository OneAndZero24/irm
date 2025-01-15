import numpy as np
import matplotlib.pyplot as plt

from src.solver import EulerMaruyama
from src.model import Vasicek


theta  = 0.7
mu     = 1.4
sigma  = 0.06

dt   = 7 / 200
t    = np.arange(0, 7, dt)

if __name__ == "__main__":
    model = Vasicek(theta, mu, sigma)
    solver = EulerMaruyama(model.a, model.b, 0, 7, 200, 5, 4)
    
    Ys = solver.run()

    for Y in Ys:
        plt.plot(t, Y)
    
    plt.grid(True)
    plt.show()