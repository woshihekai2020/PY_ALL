
import numpy as np
import math

def jacobi( a, tol= 1.0e-8 ):
    def threshold( a ):
        n = len( a )
        sum = 0.0
        for i in range( n - 1 ):
            for j in range( i+1, n ):
                sum = sum + abs( a[i, j] )
        return 0.5 * sum / n / (n - 1)