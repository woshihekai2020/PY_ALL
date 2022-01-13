
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from numpy import sign
import math

def rootsearch( f, a, b, dx ):
    x1 = a
    f1 = f( a )
    x2 = a + dx
    f2 = f( x2 )
    while sign(f1) == sign(f2):
        if x1 >= b: return None, None
        x1 = x2
        f1 = f2
        x2 = x1 + dx
        f2 = f( x2 )
    else:
        return x1, x2
def ExpRootsearch( ):
    def f(x): return x**3 - 10.0 * x ** 2 + 5.0

    x1 = 0.0
    x2 = 1.0
    for i in range( 4 ):
        dx = (x2 - x1) / 10.0
        x1, x2 = rootsearch( f, x1, x2, dx )
    x = ( x1 + x2 ) / 2.0
    print( " x = : ", "{:6.4f}".format(x) )
    input( "Press return to exit" )


def bisection( f, x1, x2, switch= 1, tol= 1.0e-9 ):
    f1 = f( x1 )
    if f1 == 0.0: return x1
    f2 = f( x2 )
    if f2 == 0.0: return x2
    if sign( f1 ) == sign( f2 ):
        print( "Root is not bracketed" )
    n = int( math.ceil( math.log( abs(x2 - x1)/tol)/math.log(2.0)))

    for i in range( n ):
        x3 = 0.5 * (x1 + x2)
        f3 = f( x3 )
        if( switch == 1 ) and (abs(f3) > abs(f1)) and(abs(f3) > abs(f2)):
            return None
        if f3 == 0.0: return x3
        if sign( f2 ) != sign(f3): x1 = x3; f1 = f3
        else: x2 = x3; f2 = f3
    return (x1 + x2)/2.0
def ExpBisection():
    def f(x): return x**3 - 10.0 * x ** 2 + 5.0

    x = bisection( f, 0.0, 1.0, tol= 1.0e-9 )
    print( 'x = ', "{:6.4f}".format(x) )
    input( "Press return to exit" )


def CmpRootBisection():
    def f(x): return x - math.tan( x )
    a, b, dx = (0.0, 20.0, 0.01)
    print( "The root are :" )
    while True:
        x1, x2 = rootsearch( f, a, b, dx )
        if x1 != None:
            a = x2
            root = bisection(f, x1, x2, 1 )
            if root != None: print( root )
        else:
            print( "\n Done" )
            break
    input( "Press return to exit " )

def ridder( f, a, b, tol= 1.0e-9 ):
    fa = f( a )
    if fa == 0.0: return a
    fb = f( b )
    if fb == 0: return b
    #if sign( fa ) != sign( fb ):
    for i in range( 30 ):
        c = 0.5 * ( a + b )
        fc = f( c )
        s = math.sqrt( fc ** 2 - fa * fb )
        if s == 0.0: return None
        dx = ( c  - a ) * fc / s
        if( fa - fb ) < 0.0: dx = -dx
        x = c + dx
        fx = f( x )
        xOld = x #add hekai 0703
        if i > 0 :
            if abs( x - xOld ) < tol * max( abs(x), 1.0 ): return x
        #xOld = x
        if sign( fc ) == sign( fx ):
            if sign( fa ) != sign( fx ):
                b = x
                fb = fx
            else:
                a = c; b = x; fa = fc; fb = fx
        return None
    print( "too many iterations" )

if __name__ =="__main__" :
    #ExpRootsearch()
    #ExpBisection()
    CmpRootBisection()