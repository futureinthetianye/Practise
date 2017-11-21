# Practiseimport numpy as np
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
#(1.1)
def y(x, W, M):
    Y = np.array([W[i] * (x ** i) for i in range(M+1)])
    return Y.sum()

def E(x, t, M):
    A =np.zeros((M+1, M+1))
    for i in range(M+1):
        for j in range(M+1):
            A[i,j] = (x**(i+j)).sum()

    T = np.array([((x**i)*t).sum() for i in range(M+1)])
    return  np.linalg.solve(A, T)

if __name__ == "__main__":
    
    x_real = np.arange(0, 1, 0.01)
    y_real = np.sin(2*np.pi*x_real)
    
    N=10
    x_train = np.arange(0, 1, 0.1)

    loc = 0
    scale = 0.3
    y_train =  np.sin(2*np.pi*x_train) + np.random.normal(loc,scale,N)

    for M in [0,1,3,9]:
        W = E(x_train, y_train, M)
        print (W)

        y_estimate = [y(x, W, M) for x in x_real]


        plt.plot(x_real, y_estimate, 'r-')
        plt.plot(x_train, y_train,'bo') 
        plt.plot(x_real, y_real, 'g-')
        xlim(0.0, 1.0)
        ylim(-2, 2)
        plt.show()
        title("Figure 1.4")
