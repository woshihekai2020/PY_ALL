
import numpy as np

def integrate( F, x, y, xStop, h ):
    X = []
    Y = []
    X.append( x )
    Y.append( y )
    while x < xStop:
        h = min( h, xStop - x )
        y = y + h * F(x, y)
        x = x + h
        X.append( x )
        Y.append( y )
        return np.array(X), np.array( Y )
def printSoln( X, Y, freq ):
    def printHead( n ):
        print( "\n    x ", end= "  " )
        for i in range( n ):
            print( "    y[", i, "] ", end= " " )
        print()
    def printLine( x, y, n ):
        print( "{:13.4e}".format(x), end= " " )
        for i in range( n ):
            print( "{:13.4e}".format(y[i]), end= " " )
        print()
    m = len( Y )
    try: n = len( Y[0] )
    except TypeError: n = 1
    if freq == 0: freq = m
    printHead( n )
    for i in range(0, m, freq):
        printLine( X[i], Y[i], n )
    if i != m - 1: printLine( X[m - 1], Y[m - 1], n )

import matplotlib
matplotlib.use( 'TkAgg' )
import matplotlib.pyplot as plt
def ExpEulerInit():
    def F(x, y):
        F = np.zeros( 2 )
        F[0] = y[1]
        F[1] = -0.1 * y[1] - x
        return F
    x = 0.0
    xStop = 2.0
    y = np.array( [0.0, 1.0])
    h = 0.05
    X, Y = integrate( F, x, y, xStop, h )
    yExact = 100.0 * X - 5.0 * X ** 2 + 990.0 * (np.exp(-0.1 * X) - 1.0)
    plt.plot( X, Y[:, 0], 'o', X, yExact, '-')
    plt.grid( True )
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def integrate( F, x, y, xStop, h ):
    def run_kut4( F, x, y, h ):
        K0 = h * F(x, y)
        K1 = h * F(x + h/2.0, y + K0/ 2.0)
        K2 = h * F(x + h/2.0, y + K1/ 2.0)
        K3 = h * F(x + h, y + K2)
        return (K0 + 2.0 * K1 + 2.0 * K2 + K3) / 6.0
    X = []
    Y = []
    X.append( x )
    Y.append( y )
    while x < xStop:
        h = min( h, xStop - x )
        y = y + run_kut4(F, x, y, h)
        x = x + h
        X.append( x )
        Y.append( y )
    return np.array( X ), np.array( Y )
def ExpRunKut4():
    def F(x, y):
        F = np.zeros( 2 )
        F[0] = y[1]
        F[1] = -0.1 * y[1] - x
        return F
    x = 0.0
    xStop = 2.0
    y = np.array( [0.0, 1.0] )
    h = 0.2

    X, Y = integrate( F, x, y, xStop, h )
    yExact = 100.0 * X - 5.0 * X ** 2 + 990.0 * (np.exp(-0.1*X) - 1.0)
    plt.plot( X, Y[:, 0],'o', X, yExact, '-' )
    plt.legend(('Numerical', 'Exact'), loc= 0)
    plt.show()



if __name__ == "__main__":
    #ExpEulerInit()
    ExpRunKut4()