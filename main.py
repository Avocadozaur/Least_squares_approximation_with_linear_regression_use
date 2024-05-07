import math
import matplotlib.pyplot as plt
import numpy as np

# x = [-1.0, 0.0, 1.0, 2.0]
# y = [4.0, -1.0, 0.0, 7.0]
# n=3

# x=[0.0,1.0,-2.0]
# y=[5.0,3.0,9.0]
# n=2

# x=[0.0,1.0,-3.0,-2.0]
# y=[6.0,0.0,0.0,12.0]
# n=4

# x = [-3.0, 1.0, 2.0, 4.0, 0.0]
# y = [0.0, 0.0, 0.0, 0.0, -24.0]
# n = 5
x = np.array([-1, 0, 1, 2])
y = np.array([4, -1, 0, 7])
n = 2


def Least_squares_approx_wLR(x, y, n):
    n += 1
    if n < 2:
        return "n can't be less than 2"
    if len(x) != len(y):
        return "not equal set of coefficients"
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
        sum = 0
        for i in range(len(x_array)):
            sum += math.pow(x_array[i], power)
        return sum

    def YSum(x_array, y_array, power):
        sum = 0
        for i in range(len(x_array)):
            if power == 0:
                sum += y_array[i]
            else:
                sum += y_array[i] * math.pow(x_array[i], power)
        return sum

    counter = 0
    for bucket in Buckets():
        for num in bucket:
            a = num % n
            b = num // n
            M[b][a] = XSum(x, counter)
        counter += 1
    counter = 0
    W = np.zeros((1, n))
    for i in range(n):
        W[0][i] = YSum(x, y, counter)
        counter += 1

    result = np.matmul(np.linalg.inv(M), np.transpose(W))

    x_values = np.linspace(min(x), max(x), 100)
    y_values = np.zeros_like(x_values)
    for i in range(len(result)):
        y_values += result[i] * (x_values ** i)

    plt.scatter(x, y, label='Points')
    plt.plot(x_values, y_values, color='red', label='Approximate polynomial')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    return result


print(Least_squares_approx_wLR(x, y, n))
