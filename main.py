import math

import numpy
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

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


def AproxSr2(xw, yw, n):
    n+=1
    if n < 2:
        return ("źle")
    if len(xw) != len(yw):
        return ("źle")
    #zrobiłem wiaderka na piramidkę z liczb która zawsze się będzie układać
    #chodzi o te same liczby na przekątnych żeby się do macierzy wpisywały
    #od x^0 do x^n, komicznie jest gdzieniegdzie zaimplementowane ale to działająca wersja alfa
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
    # return Buckets
    M = np.zeros((n, n))

    def SumowanieX(lista, stopien):
        suma = 0
        for i in range(len(lista)):
            suma += math.pow(lista[i], stopien)
        return suma

    def SumowanieY(listaX, listaY, stopien):
        suma = 0
        for i in range(len(listaX)):
            if stopien == 0:
                suma += listaY[i]
            else:
                suma += listaY[i] * math.pow(listaX[i], stopien)
        return suma

    counter = 0
    for bucket in Buckets:
        for num in bucket:
                a = num % n
                b = num // n
                M[b][a] = SumowanieX(xw, counter)
        counter += 1
    counter = 0
    W = np.zeros((1, n))
    for i in range(n):
        W[0][i] = SumowanieY(xw, yw, counter)
        counter += 1

    result = numpy.matmul(numpy.linalg.inv(M), numpy.transpose(W))

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


print(AproxSr2(xw, yw, n))
