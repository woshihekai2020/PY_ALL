
import math

def trapezoid( f, a, b, Iold, k ):
    if k == 1: Inew = (f(a) + f(b)) * (b - 1) / 2.0
    else:
        n = 2 ** (k - 2)
        h = (b - a) / n
        x = a + h / 2.0
        sum = 0.0
        for i in range( n ):
            sum = sum + f( x )
            x = x + h
        Inew = (Iold + h * sum) / 2.0

    return Inew;
def ExpTrapezoid():
    def f(x): return math.sqrt( x ) * math.cos( x )
    Iold = 0.0
    for k in range( 1, 21 ):
        Inew = trapezoid( f, 0.0, math.pi, Iold, k )
        if( k > 1 ) and (abs(Inew - Iold)) < 1.0e-6: break;
        Iold = Inew
    print( "Integral = ", Inew )
    print( "nPanels = ", 2 ** (k - 1) )



import numpy as np

def romberg( f, a, b, tol= 1.0e-6 ):
    def richardson( r, k ):
        for j in range( k-1, 0, -1 ):
            const = 4.0 ** (k - j)
            r[j]  = (const * r[j + 1] - r[j]) / (const - 1.0)
        return r
    r = np.zeros( 21 )
    r[1] = trapezoid( f, a, b, 0.0, 1 )
    r_old = r[1]
    for k in range( 2, 21 ):
        r[k] = trapezoid( f, a, b, r[k - 1], k )
        r = richardson( r, k )
        if abs( r[1] - r_old ) < tol * max( abs(r[1]), 1.0 ):
            return r[1], 2 ** (k - 1)
        r_old = r[1]
    print( "Romberg quadrature did not converge")
def ExpRomberg():
    def f(x): return 2.0 * (x ** 2) * math.cos( x ** 2 )

    I, n = romberg( f, 0, math.sqrt(math.pi) )
    print( " Integral = ", I )
    print( " numEvals = ", n )




def gaussNodes( m, tol= 1.0e-9 ):
    def legendre( t, m ):
        p0 = 1.0
        p1 = t
        for k in range( 1, m ):
            p  = ( (2.0 * k + 1.0) * t * p1 - k * p0) / (1.0 + k)
            p0 = p1
            p1 = p
        dp = m * (p0 - t * p1) / (1.0 - t ** 2 )
        return p, dp
    A = np.zeros( m )
    x = np.zeros( m )
    nRoots = int( (m + 1) / 2 )
    for i in range( nRoots ):
        t = math.cos( math.pi * (i + 0.75) / (m + 0.5) )
        for j in range( 30 ):
            p, dp = legendre( t, m )
            dt = -p / dp; t = t + dt
            if abs( dt ) < tol:
                x[i] = t; x[m - i - 1] = -t
                A[i] = 2.0/(1.0 - t ** 2) / (dp ** 2)
                A[m - i - 1] = A[ i ]
                break
    return x, A
def gaussQuad( f, a, b, m ):
    c1 = (b + a) / 2.0
    c2 = (b - a) / 2.0
    x, A = gaussNodes( m )
    sum = 0.0
    for i in range( len(x) ):
        sum = sum + A[i] * f(c1 + c2 * x[i])
    return c2 * sum
def ExpGaussQuad():
    def f(x): return (math.sin(x)/x) ** 2

    a = 0.0; b = math.pi
    Iexact = 1.41815
    for m in range( 2, 12 ):
        I = gaussQuad( f, a, b, m )
        if abs( I - Iexact ) < 0.0001:
            print("Number of nodes = ", m )
            print("Integrad = " , gaussQuad(f, a, b, m))
            break




if __name__ == "__main__":
    #ExpTrapezoid()
    #ExpRomberg()  #ERROR: not run
    ExpGaussQuad()

