
import cv2
import numpy as np
from math import sin, cos, pi
import random

def _Gray( img ):
    x, y, z = np.shape( img )
    gray = np.zeros( [x, y], "uint8" )
    for i in range( x ):
        for j in range( y ):
            gray[i][j] = np.dot( np.array(img[i][j], dtype= "float"), [.114, .587, .299] )
    return gray

def _Sift( img ):
    r, c = np.shape( img )
    corners = [ [int(i[0][0]), int(i[0][1])] for i in cv2.goodFeaturesToTrack(img, 233, 0.01, 10) ]
    img = cv2.GaussianBlur( img, (5, 5), 1, 1 )
    img = np.array( img, dtype= "float" )
    def _Grad( img ):
        x, y = r, c
        kernel = np.array([
            [[-1, 0, 1], [-1, 0,-1], [-1, 0, 1]],
            [[-1,-1,-1], [ 0, 0, 0], [ 1, 1, 1]]], dtype= "float" ) / 6
        gx = cv2.filter2D( img, -1, np.array( kernel[1] ))
        gy = cv2.filter2D( img, -1, np.array( kernel[0] ))
        gradient = np.zeros( [x, y], "float")
        angle = np.zeros( [x, y], 'float')
        for i in range( x ):
            for j in range( y ):
                gradient[i][j] = ((gx[i][j]) ** 2 + (gy[i][j]) ** 2 ) ** 0.5
                angle[i][j]    = np.math.atan2( gy[i][j], gx[i][j] )
        return gradient, angle
    gradient, angle = _Grad( img )
    bins = ( r + c ) // 80
    length = len( corners )

    def _Vote():
        direct = []
        for corner in corners:
            y, x = corner
            voting = [0 for i in range(37)]
            for i in range( max(x - bins, 0), min(x + bins + 1, r)):
                for j in range( max(x - bins, 0), min(y + bins + 1, c )):
                    k = int(( angle[i][j] + pi) / (pi/18) + 1)
                    if k >= 37:
                        k = 36
                    voting[k] += gradient[i][j]

            p = 1
            for i in range(2, 37):
                if voting[i] > voting[p]:
                    p = i
        return direct
    direct = _Vote()

    def _Feature( pos, theta ):
        def _theta( x, y ):
            if( x < 0 or x >= r ) or ( y < 0 or y >= c ):
                return 0
            dif = angle[x][y] - theta
            return dif if dif > 0 else dif + 2 * pi
        def _DB_linear( x, y ):
            xx, yy = int( x ), int( y )
            dy1, dy2 = y - yy, yy + 1 - y
            dx1, dx2 = x - xx, xx + 1 - x
            val = _theta( xx, yy ) * dx2 * dy2 + _theta(xx + 1, yy) * dx1 *dy2 \
                + _theta( xx, yy + 1 ) * dx1, dy2 + _theta( xx + 1, yy + 1 ) * dx1 * dy1
            return val
        y0, x0 = pos
        H = np.array( [ cos(theta), sin(theta)] )
        V = np.array( [-sin(theta), cos(theta)] )

        val = []



if __name__ == "__main__":
    tgt0 = cv2.imread( r"", 1 )
    imgset0 = [ cv2.imread(r"") for i in range( 1, 6 ) ]

    r0, c0, a0 = np.shape( tgt0 )
    times = 1.0

    resized_tgt0 = cv2.resize( tgt0, (int(r0 * times), int(c0 * times)) )
    tgt = _Gray( resized_tgt0 )
    imgset = [_Gray( imgset0[i] ) for i in range( len(imgset0) )]

    ff = []
    cc = []
    ll = []
    ft, ct, lt = _Sift( tgt )



