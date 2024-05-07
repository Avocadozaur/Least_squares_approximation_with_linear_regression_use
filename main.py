import math
import matplotlib.pyplot as plt
import numpy as np

# xw = [-1.0, 0.0, 1.0, 2.0]
# yw = [4.0, -1.0, 0.0, 7.0]
# n=3

# xw=[0.0,1.0,-2.0]
# yw=[5.0,3.0,9.0]
# n=2

# xw=[0.0,1.0,-3.0,-2.0]
# yw=[6.0,0.0,0.0,12.0]
# n=4

# xw = [-3.0, 1.0, 2.0, 4.0, 0.0]
# yw = [0.0, 0.0, 0.0, 0.0, -24.0]
# n = 5
xw = np.array([-1, 0, 1, 2])
yw = np.array([4, -1, 0, 7])
n = 2


def Least_squares_approx_wLR(xw, yw, n):
    n += 1
    if n < 2:
        return ("n can't be less than 2")
    if len(xw) != len(yw):
        return ("not equal set of coefficients")
    def Buckets():
    #my idea for implementing a matrix with subsequent powers of x diagonally
    #simple implementation of the linear regression formula
        Buckets = []
        for i in range(2 * n - 1):
            Buckets.append([])
        counter, i, j = 0, 0, 0
        while i != (n * n):
            Buckets[counter + j].append(i)
            counter += 1
            i += 1
            if counter == n:
                j += 1
                counter = 0
        return Buckets

    M = np.zeros((n, n))

    def XSum(x_array, power):
        suma = 0
        for i in range(len(x_array)):
            suma += math.pow(x_array[i], power)
        return suma

    def YSum(x_array, y_array, power):
        suma = 0
        for i in range(len(x_array)):
            if power == 0:
                suma += y_array[i]
            else:
                suma += y_array[i] * math.pow(x_array[i], power)
        return suma

    counter = 0
    for bucket in Buckets():
        for num in bucket:
            a = num % n
            b = num // n
            M[b][a] = XSum(xw, counter)
        counter += 1
    counter = 0
    W = np.zeros((1, n))
    for i in range(n):
        W[0][i] = YSum(xw, yw, counter)
        counter += 1

    result = np.matmul(np.linalg.inv(M), np.transpose(W))

    x_values = np.linspace(min(xw), max(xw), 100)
    y_values = np.zeros_like(x_values)
    for i in range(len(result)):
        y_values += result[i] * (x_values ** i)

    plt.scatter(xw, yw, label='Punkty')
    plt.plot(x_values, y_values, color='red', label='Wielomian aproksymowanyy')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    return result


print(Least_squares_approx_wLR(xw, yw, n))
