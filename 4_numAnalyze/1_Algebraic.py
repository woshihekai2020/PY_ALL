
import numpy as np

def gaussElimin( a, b ):
    n = len( a )
    for k in range( 0, n-1 ):
        for i in range( k+1, n ):
            if a[i, k] != 0.0:
                lam = a[i, k] / a[k, k]
                a[i, k+1:n] = a[i, k+1:n] - lam * a[k, k+1:n]
                b[i] = b[i] - lam * b[k]
    for k in range( n-1, -1, -1 ):
        b[k] = (b[k] - np.dot(a[k, k+1:n], b[k+1: n])) / a[k, k]
    return b
def LU( a, b ):
    n = len( a )
    for k in range(0, n-1):
        for i in range(k+1, n):
            if a[i, k] != 0.0:
                lam = a[i, k] / a[k, k]
                a[i, k+1:n] = a[i, k+1:n] - lam * a[k, k+1:n]
                a[i, k] = lam
    for k in range( 1, n ):
        b[k] = b[k] - np.dot(a[k, 0:k], b[0: k])
    b[n - 1] = b[n - 1] / a[n-1, n-1]
    for k in range( n-2, -1, -1 ):
        b[k] = (b[k] - np.dot(a[k, k+1:n], b[k+1: n])) / a[k, k]
    return b
def EXPgauss():
    def vandermode( v ):
        n = len( v )
        a = np.zeros( (n, n) )
        for j in range( n ):
            a[:, j] = v ** (n - j - 1)
        return a
    v = np.array( [1.0, 1.2, 1.4, 1.6, 1.8, 2.0] )
    b = np.array( [0.0, 1.0, 0.0, 1.0, 0.0, 1.0] )
    a = vandermode( v )
    aOrig = a.copy()
    bOrig = b.copy()
    aLU   = a.copy()
    bLU   = b.copy()
    x   = gaussElimin(a, b)
    xLU = LU( aLU, bLU )
    det = np.prod( np.diagonal(a) )
    print( '\ndet = ', det )
    print( 'x = \n', x )
    print( '\nxLU = \n', xLU )
    print( '\nCheck result: [a]{x} - b = \n', np.dot(aOrig, x) - bOrig )
    print( '\nCheck result: [a]{xLU}-b = \n', np.dot(aOrig, xLU) - bOrig )

import math
def choleski( a, b ):
    n = len( a )
    for k in range( n ):
        try:
            a[k, k] = math.sqrt( a[k, k] - np.dot(a[k, 0:k], a[k, 0:k]) )
        except ValueError:
            print( 'matrix is not positive definite' )
        for i in range( k+1, n ):
            a[i, k] = (a[i, k] - np.dot(a[i, 0:k], a[k, 0:k])) / a[k, k]
    for k in range( 1, n ): a[0:k, k] = 0.0
    #solve
    n = len( b )
    for k in range( n ):
        b[k] = (b[k] - np.dot(a[k, 0:k], b[0: k])) / a[k, k]
    for k in range( n-1, -1, -1 ):
        b[k] = (b[k] - np.dot(a[k+1:n, k], b[k+1: n])) / a[k, k]
    return b
def EXPcholeski():
    a = np.array( [[ 1.44,-0.36, 5.52, 0.0], \
                   [-0.36,10.33,-7.78, 0.0], \
                   [ 5.52,-7.78,28.40, 9.0], \
                   [ 0.0,  0.0 , 9.0 ,61.0]])
    b = np.array( [ 0.04, -2.15, 0.0, 0.88])
    aOrig = a.copy()
    x = choleski(a, b)
    print( 'x = ', x )
    print( '\nCheck: A * x = \n', np.dot(aOrig, x) )
def LU3( c, d, e, b ):
    n = len( d )
    for k in range( 1, n ):
        lam  = c[k - 1] / d[k - 1]
        d[k] = d[k] - lam * e[k - 1]
        c[k - 1] = lam
    for k in range( 1, n ):
        b[k] = b[k]  - c[k - 1] * b[k - 1]
    b[n - 1] = b[n - 1] / d[n - 1]
    for k in range( n-2, -1, -1 ):
        b[k] = (b[k] - e[k] * b[k + 1]) / d[k]
    return b
def EXPlu3():
    d = np.ones( (5) ) * 2.0
    c = np.ones( (4) ) * (-1.0)
    b = np.array( [ 5.0,-5.0, 4.0,-5.0, 5.0 ] )
    e = c.copy()
    x = LU3( c, d, e, b )
    print( '\nx = \n', x )

def swapRows( v, i, j ):
    if len(v.shape) == 1:
        v[i], v[j] = v[j], v[i]
    else:
        v[[i,j], :] = v[[j, i], :]
def swapCols( v, i, j ):
    v[:, [i, j]] = v[:, [j, i]]
def gaussPoivot( a, b, tol= 1.0e-12 ):
    n = len( b )
    s = np.zeros( n )
    for i in range( n ):
        s[i] = max( np.abs(a[i, :]) )
    for k in range( 0, n-1 ):
        p = np.argmax( np.abs(a[k:n, k])/s[k: n]) + k
        if abs(a[p, k]) < tol: print("matrix is singular")
        if p != k:
            swapRows( b, k, p )
            swapRows( s, k, p )
            swapRows( a, k, p )
        for i in range( k+1, n ):
            if a[i, k] != 0.0:
                lam = a[i, k] / a[k, k]
                a[i, k+1:n] = a[i, k+1:n] - lam * a[k, k+1:n]
                b[i] = b[i] - lam * b[k]
    if abs( a[n-1, n-1] ) < tol: print('matirx is singular')

    b[n - 1] = b[n - 1] / a[n-1, n-1]
    for k in range( n-2, -1, -1 ):
        b[k] = (b[k] - np.dot(a[k, k+1:n], b[k+1: n])) / a[k, k]
    return b
def LUPivot( a, b, tol= 1.0e-9 ):
    n = len( a )
    seq = np.array( range(n) )

    s = np.zeros( (n) )
    for i in range( n ):
        s[i] = max( abs(a[i, :]) )
    for k in range(0, n-1):
        p = np.argmax( np.abs(a[k:n, k]) / s[k: n]) + k
        if abs( a[p, k] ) < tol: print('matrix is singular')
        if p != k :
            swapRows( s, k, p )
            swapRows( a, k, p )
            swapRows( seq, k, p )
        for i in range( k+1, n ):
            if a[i, k] != 0.0:
                lam = a[i, k] / a[k, k]
                a[i, k+1:n] = a[i, k+1:n] - lam * a[k, k+1:n]
                a[i, k] = lam
    x = b.copy()
    for i in range( n ):
        x[i] = b[ seq[i] ]
    for k in range( 1, n ):
        x[k] = x[k] - np.dot( a[k, 0:k], x[0: k] )
    x[n - 1] = x[n - 1] / a[n-1, n-1]
    for k in range( n-2, -1, -1 ):
        x[k] = (x[k] - np.dot(a[k, k+1:n], x[k+1: n])) / a[k, k]
    return x
def EXPcmp():
    def vandermode( v ):
        n = len( v )
        a = np.zeros( (n, n) )
        for j in range( n ):
            a[:, j] = v ** (n - j - 1)
        return a
    v = np.array( [1.0, 1.2, 1.4, 1.6, 1.8, 2.0] )
    b = np.array( [0.0, 1.0, 0.0, 1.0, 0.0, 1.0] )
    a = vandermode( v )
    aOrig = a.copy()
    bOrig = b.copy()
    aLU   = a.copy()
    bLU   = b.copy()
    x     = gaussPoivot( a, b )
    xLU   = LUPivot( a, b )
    det = np.prod( np.diagonal(a) )
    print( '\ndet = ', det )
    print( 'x = \n', x )
    print( '\nxLU = \n', xLU )
    print( '\nCheck pivot result: [a]{x} - b = \n', np.dot(aOrig, x) - bOrig )
    print( '\nCheck pivot result: [a]{xLU}-b = \n', np.dot(aOrig, xLU) - bOrig )


def gaussSeidel( iterEqs, x, tol= 1.0e-9 ):
    omega = 1.0
    k = 10
    p = 1
    for i in range( 1, 501 ):
        xOld = x.copy()
        x = iterEqs( x, omega )
        dx = math.sqrt( np.dot(x-xOld, x-xOld) )
        if dx < tol: return x, i, omega
        if i == k: dx1 = dx
        if i == k + p:
            dx2 = dx
            omege = 2.0 / (1.0 + math.sqrt(1.0 - (dx2/dx1) ** (1.0/p)))
    print( 'Gauss-Seidel failed to converge' )

def conjGrad( Av, x, b, tol= 1.0e-9 ):
    n = len( b )
    r = b - Av( x )
    s = r.copy()
    for i in range( n ):
        u = Av( s )
        alpha = np.dot( s, r ) / np.dot( s, u )
        x = x + alpha * s
        r = b - Av( s )
        if( math.sqrt(np.dot(r, r)) ) < tol:
            break
        else:
            beta = -np.dot( r, u )/ np.dot( s, u )
            s = r + beta * s
    return x, i






if __name__ == "__main__":
    #EXPgauss()
    #EXPcholeski()
    #EXPlu3()
    EXPcmp()

