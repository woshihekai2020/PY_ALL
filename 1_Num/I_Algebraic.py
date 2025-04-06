import numpy as np
import math

####################################################################################################### 1:direct methods
def gaussElimin( a, b ):
    n = len( a )
    for k in range(0, n-1):
        for i in range(k+1, n):
            if a[i, k] != 0.0:
                lam = a[i, k] / a[k, k]
                a[i, k+1:n] = a[i, k+1:n] - lam * a[k, k+1:n]
                b[i] = b[i] - lam * b[k]


    for k in range(n-1, -1, -1):
        b[k] = (b[k] - np.dot(a[k, k+1:n], b[k+1:n])) / a[k, k]
    return b
def LU( a, b ):
    n = len( a )
    for k in range(0, n-1):
        for i in range(k+1, n):
            if a[i, k] != 0.0:
                lam = a[i, k] / a[k, k]
                a[i, k+1:n] = a[i, k+1:n] - lam * a[k, k+1:n]
                a[i, k] = lam
    for k in range(1, n):
        b[k] = b[k] - np.dot(a[k, 0:k], b[0: k])
    b[n - 1] = b[n - 1] / a[n-1, n-1]
    for k in range(n-2, -1, -1):
        b[k] = (b[k] - np.dot(a[k, k+1:n], b[k+1: n])) / a[k, k]
    return b
def choleski(a, b):
    n = len( a )
    for k in range( n ):
        try:
            a[k, k] = math.sqrt(a[k, k] - np.dot(a[k, 0:k], a[k, 0:k]))
        except ValueError:
            print( 'matrix is not positive definite' )
            return
        for i in range( k+1, n ):
            a[i, k] = (a[i, k] - np.dot(a[i, 0:k], a[k, 0:k])) / a[k, k]
    for k in range(1, n):
        a[0:k, k] = 0.0

    n = len( b )
    for k in range( n ):
        b[k] = (b[k] - np.dot(a[k, 0:k], b[0: k])) / a[k, k]
    for k in range( n-1, -1, -1 ):
        b[k] = (b[k] - np.dot(a[k+1:n, k], b[k+1: n])) / a[k, k]
    return b

def LU3(c, d, e, b):
    n = len( d )
    for k in range(1, n ):
        lam = c[k - 1] / d[k - 1]
        d[k] = d[k] - lam * e[k - 1]
        c[k - 1] = lam
    for k in range(1, n):
        b[k] = b[k] - c[k - 1] * b[k - 1]
    b[n - 1] = b[n - 1] / d[n - 1]
    for k in range(n-2, -1, -1):
        b[k] = (b[k] - e[k] * b[k + 1]) / d[k]
    return b
def swapRows(v, i, j):
    if len(v.shape) == 1:
        v[i], v[j] = v[j], v[i]
    else:
        v[[i, j], :] = v[[j, i], :]
def swapCols(v, i, j):
    v[:, [i, j]] = v[:, [j, i]]
def gaussPivot(a, b, tol= 1.0e-12):
    n = len( b )
    s = np.zeros( n )
    for i in range( n ):
        s[i] = max( np.abs(a[i, :]) )
    for k in range(0, n-1):
        p = np.argmax(np.abs(a[k:n, k]) / s[k: n]) + k
        if abs(a[p, k]) < tol: print("matrix is singular")
        if p != k:
            swapRows(b, k, p)
            swapRows(s, k, p)
            swapRows(a, k, p)
    for i in range(k+1, n):
        if a[i, k] != 0.0:
            lam = a[i, k] / a[k, k]
            a[i, k+1:n] = a[i, k+1:n] - lam * a[k, k+1:n]
            b[i] = b[i] - lam * b[k]
    if abs( a[n-1, n-1] ) < tol: print("matrix is singular")

    b[n - 1] = b[n - 1] / a[n-1, n-1]
    for k in range(n-2, -1, -1):
        b[k] = (b[k] - np.dot(a[k, k+1:n], b[k+1: n])) / a[k, k]
    return b
def LUPivot(a, b, tol=1.0e-12):
    n = len( a )
    seq = np.array( range(n) )

    s = np.zeros( (n) )
    for i in range( n ):
        s[i] = max( abs(a[i, :]) )
    for k in range(0, n-1):
        p = np.argmax( np.abs(a[k:n, k])/ s[k: n] ) + k
        if abs( a[p, k] ) < tol:  print("matrix is singular")
        if p != k:
            swapRows(s, k, p)
            swapRows(a, k, p)
            swapRows(seq, k, p)
        for i in range(k+1, n):
            if a[i, k] != 0.0:
                lam = a[i, k] / a[k, k]
                a[i, k+1:n] = a[i, k+1:n] - lam * a[k, k+1:n]
                a[i, k] = lam
    x = b.copy()
    for i in range(1, n):
        x[i] = b[ seq[i] ]
    for k in range(1, n ):
        x[k] = x[k] - np.dot(a[k, 0:K], x[0: k])
    x[n - 1] = x[n - 1] / a[n-1, n-1]
    for k in range(n-2, -1, -1):
        x[k] = (x[k] - np.dot(a[k, k+1:n], x[k+1: n])) / a[k, k]
    return x

#################################################################################################### 2:iterative methods
def gaussSeidel( iterEqs, x, tol= 1.0e-9 ):
    omega = 1.0
    k = 10
    p = 1
    for i in range(1, 501):
        xOld = x.copy()
        x = iterEqs(x, omega)
        dx = math.sqrt( np.dot(x-xOld, x-xOld) )
        if dx < tol: return x, i, omega
        if i == k: dx1 = dx
        if i == k + p:
            dx2 = dx
            omega = 2.0 / (1.0 + math.sqrt(1.0 - (dx2/dx1)**(1.0/p)))
    print('Gauss-Seidel failed to converge')
def conjGrad(Av, x, b, tol= 1.0e-9):
    n = len(b)
    r = b - Av(x)
    s = r.copy()
    for i in range(n):
        u = Av(s)
        alpha = np.dot(s, r)/np.dot(s, u)
        x = x + alpha * s
        r = b - Av(x)
        if( math.sqrt(np.dot(r, r))) < tol:
            break
        else:
            beta = -np.dot(r, u) / np.dot(s, u)
            s = r + beta * s
    return x, i

################################################################################################################## 3:EXP
def ExpCmp():
    def vandermode( v ):
        n = len( v )
        a = np.zeros((n, n))
        for j in range( n ):
            a[:, j] = v ** (n - j - 1)
        return a
    v = np.array( [1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    b = np.array( [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    a = vandermode( v )
    aOrig   = a.copy()
    bOrig   = b.copy()
    aLU     = a.copy()
    bLU     = b.copy()
    x       = gaussElimin( a, b )
    xLU     = LU(aLU, bLU)
    det     = np.prod( np.diagonal(a) )
    detLU   = np.prod( np.diagonal(aLU))

    print( '\ndet = ', det )
    print( 'x = \n', x )
    print('\ndet_LU = ', detLU)
    print('x_LU = \n', xLU)
def ExpCholeski():
    a = np.array( [ [ 1.44,-0.36, 5.52, 0.00], \
                    [-0.36,10.33,-7.78, 0.00], \
                    [ 5.52,-7.78,28.40, 9.00], \
                    [ 0.00, 0.00, 9.00,61.00]] )
    b = np.array(   [ 0.04,-2.15, 0.00, 0.88]  )
    aOrig = a.copy()
    x = choleski(a, b)
    print( "solve choleski: x = \n", x )
    print( "\n check: A * X = \n", np.dot(aOrig, x) )

def ExpLU3():
    d = np.ones( (5) ) * 2.0
    c = np.ones( (4) ) * (-1.0)
    b = np.array( [ 5.0,-5.0, 4.0,-5.0, 5.0 ] )
    e = c.copy()
    x = LU3(c, d, e, b )
    print( '\nx = \n', x )
#not run
def ExpMatInvLU():
    def matInv( a ):
        n = len( a[0] )
        aInv = np.identity( n )
        for i in range( n ):
            aInv[:, i] = LU(a, aInv[:, i])
        return aInv
    a = np.array([[ 0.6,-0.4, 1.0],
                  [-0.3, 0.2, 0.5],
                  [ 0.6,-1.0, 0.5]])
    aOrig = a.copy()
    aInv = matInv(a)
    print( "\n aInv = \n", aInv )
    print( "\n check: a * aInv = \n", np.dot(aOrig, aInv))

def ExpGaussSeidel():
    def iterEqs(x, omega):
        n = len(x)
        x[0] = omega * (x[1] - x[n - 1]) / 2.0 + (1.0 - omega) * x[0]
        for i in range(1, n-1):
            x[i] =omega * (x[i - 1] + x[i + 1]) / 2.0 + (1.0 - omega) * x[i]
        x[n - 1] = omega * (1.0 - x[0] + x[n - 2]) / 2.0 + (1.0 - omega) * x[n - 1]
        return x
    n = eval( input("number of equations ==> " ) )
    x = np.zeros( n )
    x, numIter, omega = gaussSeidel( iterEqs, x )
    print( "\n number of iterations = ", numIter )
    print( "\n Relaxation factor = ", omega )
    print( "\n the solution is :\n", x )
def ExpConjGrad():
    def Ax(v):
        n = len(v)
        Ax = np.zeros( n )
        Ax[0] = 2.0 * v[0] - v[1] + v[n - 1]
        Ax[1: n-1] = -v[0: n-2] + 2.0 * v[1: n-1] - v[2: n]
        Ax[n - 1] = -v[n - 2] + 2.0 * v[n - 1] + v[0]
        return Ax
    n = eval( input("number of equations ==> " ) )
    b = np.zeros( n )
    b[n - 1] = 1.0
    x = np.zeros( n )
    x, numIter = conjGrad(Ax, x, b)
    print( "\n The solution is : \n", x )
    print( "\n Number of iterations = \n", numIter )


if __name__ == "__main__":
    #ExpCmp()           #campare Gauss and LU
    #ExpCholeski()

    #ExpLU3()
    #ExpMatInvLU()

    #ExpGaussSeidel()
    ExpConjGrad()
