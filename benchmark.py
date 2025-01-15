import time
import numpy as np
import matplotlib.pyplot as plt

from src.solver import EulerMaruyama


theta  = 0.7
mu     = 1.4
sigma  = 0.06

dt   = 7 / 200
t    = np.arange(0, 7, dt)

def a(x):
    return theta*(mu-x)

def b(x):
    return sigma

if __name__ == "__main__":
    em1 = EulerMaruyama(a, b, 0, 7, 2000, 1000, 1)
    em4 = EulerMaruyama(a, b, 0, 7, 2000, 1000, 4)

    def timerun(em):
        t1 = time.time()
        Y = em.run()
        t2 = time.time()
        return t2-t1
    
    r1 = []
    r4 = []
    avg = lambda x : sum(x)/len(x)
    for i in range(10):
        r1.append(timerun(em1))
        r4.append(timerun(em4))

    print(avg(r1))
    print(avg(r4))