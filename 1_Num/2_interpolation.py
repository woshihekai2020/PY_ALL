import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math

######################################################################################################## 1:newton method
def evalPoly(a, xData, x):
    n = len(xData) - 1
    p = a[n]
    for k in range(1, n+1):
        p = a[n - k] + (x - xData[n - k]) * p
    return p
def coeffts(xData, yData):
    m = len(xData)
    a = yData.copy()
    for k in range(1, m):
        a[k: m] = (a[k: m] - a[k - 1]) / (xData[k: m] - xData[k - 1])
    return a
def ExpNewton():
    xData = np.array( [ 0.15, 2.3, 3.15, 4.85, 6.25, 7.95] )
    yData = np.array( [ 4.79867, 4.49013, 4.2243, 3.47313, 2.66674, 1.51909] )
    a = coeffts(xData, yData)
    print( "x yInterp yExact")
    print( "----------------")
    for x in np.arange(0.0, 8.1, 0.5 ):
        y = evalPoly(a, xData, x)
        yExact = 4.8 * math.cos( math.pi * x / 20.0 )
        print( "{:3.1f} {:9.5f} {:9.5f}".format(x, y, yExact) )
    input( "\n Press return to exit" )

################################################################################################### 2:neville & rational
def neville(xData, yData, x):
    m = len(xData)
    y = yData.copy()
    for k in range(1, m):
        y[0: m-k] = ((x - xData[k: m]) * y[0: m-k] + (xData[0: m-k] - x) * y[1: m-k+1]) / (xData[0: m-k] - xData[k: m])
    return y[0]
def rational(xData, yData, x):
    m = len(xData)
    r = yData.copy()
    rOld = np.zeros( m )
    for k in range(m - 1):
        for i in range(m - k - 1):
            if abs(x - xData[i + k +1]) < 1.0e-9:
                return yData[i + k + 1]
            else:
                c1 = r[i + 1] - r[i]
                c2 = r[i + 1] - rOld[i + 1]
                c3 = (x - xData[i]) / (x - xData[i + k + 1])
                r[i] = r[i + 1] + c1 / (c3 * (1.0 - c1/c2) - 1.0)
                rOld[i + 1] = r[i + 1]
    return r[0]
def ExpCmpNevilleRational():
    xData = np.array( [    0.1,    0.2,    0.5,    0.6,    0.8,    1.2,    1.5] )
    yData = np.array( [-1.5342,-1.0811,-0.4445,-0.3085,-0.0868, 0.2281, 0.3824] )
    x     = np.arange(0.1, 1.55, 0.05)
    n     = len(x)
    y     = np.zeros( (n, 2) )
    for i in range( n ):
        y[i, 0] = rational(xData, yData, x[i])
        y[i, 1] = neville( xData, yData, x[i])
    plt.plot( xData, yData, 'o', x, y[:, 0], '-', x, y[:, 1], '--' )
    plt.xlabel('x')
    plt.legend( ('Data', 'Rational', 'Neville'), loc= 0 )
    plt.show()
    input( '\n Press return to exit')

######################################################################################################### 3:Cubic method
def LUdecomp3(c, d, e):
    n = len( d )
    for k in range(1, n):
        lam = c[k - 1] / d[k - 1]
        d[k] = d[k] - lam * e[k - 1]
        c[k - 1] = lam
    return c, d, e
def LUsolve3(c, d, e, b):
    n = len( d )
    for k in range(1, n):
        b[k] = b[k] - c[k - 1] * b[k - 1]
    b[n - 1] = b[n - 1] / d[n - 1]
    for k in range(n-2, -1, -1):
        b[k] = (b[k] - e[k]*b[k+1]) / d[k]
    return b
def curvatures(xData, yData):
    n = len(xData) - 1
    c = np.zeros( n )
    d = np.ones(n + 1)
    e = np.zeros( n )
    k = np.zeros(n + 1)
    c[0: n-1] = xData[0: n-1] - xData[1: n]
    d[1: n] = 2.0 * (xData[0: n-1] - xData[2: n+1])
    e[1: n] = xData[1: n] - xData[2: n+1]
    k[1: n] = 6.0 * (yData[0: n-1] - yData[1: n]) / (xData[0: n-1] - xData[1: n]) \
            - 6.0 * (yData[1: n] - xData[2: n+1]) / (xData[1: n] - xData[2: n+1])
    LUdecomp3(c, d, e)
    LUsolve3(c, d, e, k)
    return k
def evalSpline(xData, yData, k, x):
    def findSegment(xData, x):
        iLeft = 0
        iRight = len(xData) - 1
        while 1:
            if(iRight - iLeft) <= 1: return iLeft
            i = int( (iLeft + iRight) / 2 )
            if x < xData[i]: iRight = i
            else: iLeft = i
    i = findSegment(xData, x)
    h = xData[i] - xData[i + 1]
    y = ((x - xData[i + 1]) ** 3 / h - (x - xData[i + 1]*h) * k[i] / 6.0) \
        - ((x - xData[i])) ** 3/h - (x - xData[i] * h) * k[i + 1] / 6.0 \
        + (yData[i] * (x - xData[i + 1]) - yData[i + 1] * (x - xData[i])) / h
    return y
def ExpCubic():
    xData = np.array( [1, 2, 3, 4, 5], float )
    yData = np.array( [0, 1, 0, 1, 0], float )
    k = curvatures(xData, yData)
    while True:
        try: x = eval( input("\n x ==> : ") )
        except SyntaxError: break
        print( "y = ", evalSpline(xData, yData, k ,x) )
    input( "Done, Press return to exit")

if __name__=="__main__":
    #ExpNewton()
    #ExpCmpNevilleRational()
    ExpCubic()
