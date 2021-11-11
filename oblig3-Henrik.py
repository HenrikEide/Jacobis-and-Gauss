from os import name
import numpy as np
from numpy.lib.twodim_base import mask_indices


def jacobis_method_steps(A, b):
    x = [1.1,2.1,3.1]
    diag = np.diag(A)
    diff = A - np.diagflat(diag)

    for _ in range(100):
        x = (b-np.dot(diff,x))/diag
    
    return list(x)


def jacobis_method_accuracy(A, b):
    x = [1.1,2.1,3.1]
    diag = np.diag(A)
    diff = A - np.diagflat(diag)
    x_last = [-1,-1,-1]
    steps = 0

    while abs(x[0]-x_last[0]) > 0.0000000005:
        x_last = x
        x = (b-np.dot(diff,x))/diag
        steps += 1
    
    return (list(x),steps)


def gauss_seidel(A,b):
    x = [1.1,2.1,3.1]
    x_last = [-1,-1,-1]
    steps = 0

    while x[0]-x_last[0] > 0.0000000005:
        x_last = x
        steps += 1

        for i in range(len(x)):
            x = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,(i+1):], x_last[(i+1):])) / A[i ,i]
    return list(x)

if __name__=="__main__":
    A = np.array(([9,9,1],[2,7,2],[2,2,5]))
    b = np.array([30,22,21])
    print("The matrix A:\n",A)
    print("\nThe vector b:\n",b)

    print("\nJacobis method with 100 steps:")
    print(jacobis_method_steps(A, b))

    print("\nJacobis method with accuracy down to 10 decimals:")
    acc = jacobis_method_accuracy(A,b)
    print(f"{acc[0]}\nin {acc[1]} steps.")

    print("\nGauss Seidel method with 10 dec accuracy:")
    print(gauss_seidel(A,b))




