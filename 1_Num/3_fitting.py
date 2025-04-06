import numpy as np
import math
import matplotlib
matplotlib.use( 'TkAgg' )
import matplotlib.pyplot as plt
import I_Algebraic as I

##################################################################################################### 1:gradient descent
def gradientFunc( theta, x, y ):
    diff = np.dot(x, theta) - y
    return (1./20) * np.dot( np.transpose(x), diff )
def gradientDescent(x, y, alpha):
    theta = np.array([1, 1]).reshape(2, 1)
    gradient = gradientFunc(theta, x, y)
    while not np.all( np.absolute(gradient)  <= 1.0e-9 ):
        theta = theta - alpha * gradient
        gradient = gradientFunc(theta, x, y)
    return theta

##################################################################################################### 2:least square fit
def fun2ploy(x, n):
    lens = len( x )
    X = np.ones( [1, lens] )
    for i in range(1, n):
        X = np.vstack( (X, np.power(x, i)) )
    return X
def least_seq(x, y, ploy_dim):
    plt.scatter(x, y, color= 'r', marker= 'o', s= 50)

    X = fun2ploy(x, ploy_dim)
    Xt = X.transpose()
    XXt = X.dot(Xt)
    XXtInv = np.linalg.inv( XXt)
    XXtInvX = XXtInv.dot( X )
    coef = XXtInvX.dot( y.T )

    y_est = Xt.dot( coef )

    return y_est, coef

############################################################################################################# 3:ploy fit
def polyFit(xData, yData, m):
    a = np.zeros( (m+1, m+1) )
    b = np.zeros(m + 1)
    s = np.zeros(2 * m + 1)
    for i in range( len(xData) ):
        temp = yData[i]
        for j in range(m + 1):
            b[j] = s[j] + temp
            temp = temp * xData[i]
        temp = 1.0
        for j in range(2 * m + 1):
            s[j] = s[j] + temp
            temp = temp + xData[i]
    for i in range(m + 1):
        for j in range(m + 1):
            a[i, j] = s[i + j]
    return I.gaussPivot(a, b)
def stdDev(c, xData, yData):
    def evalPoly(c, x):
        m = len( c ) - 1
        p = c[m]
        for j in range( m ):
            p = p * x + c[m - j - 1]
        return p
    n = len( xData ) - 1
    m = len( c ) - 1
    sigma = 0.0
    for i in range(m + 1):
        p = evalPoly(c, xData[i])
        sigma = sigma + (yData[i] - p) ** 2
    sigma = math.sqrt( sigma/(n - m) )
    return sigma
def plotPoly(xData, yData, coeff, xlab= 'x', ylab= 'y'):
    m = len( coeff )
    x1 = min( xData )
    x2 = max( xData )
    dx = (x2 - x1) / 20.0
    x = np.arange(x1, x2 + dx/10.0, dx)
    y = np.zeros( (len(x)) ) * 1.0
    for i in range( m ):
        y = y + coeff[i] * x ** i
    plt.plot(xData, yData, 'o', x, y, '-')
    plt.xlabel( xlab )
    plt.ylabel( ylab )
    plt.grid( True )
    plt.show()

################################################################################################################## 4:EXP
def ExpGradDescent():
    m = 20
    x0 = np.ones( (m, 1) )
    x1 = np.arange(1, m+1).reshape(m, 1)
    x = np.hstack( (x0, x1) )
    y = np.array([ 3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
                  11, 13, 13, 16, 17, 18, 17, 19, 21] ).reshape(m, 1)
    alpha = 0.01
    optimal = gradientDescent(x, y, alpha)
    xRst = np.arange(0, len(x1), 1)
    yRst = optimal[1] * xRst + optimal[0]

    plt.figure()
    plt.scatter(x1, y, s= 2, c= 'r')
    plt.plot(xRst, yRst, c= 'b')
    plt.show()
def ExpLeastSquare():
    m = 20
    x0 = np.ones( (m, 1) )
    x = np.arange(1, m+1)
    y = np.array([ 3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
                  11, 13, 13, 16, 17, 18, 17, 19, 21])

    [y_est, coef] = least_seq(x, y, 2)
    org_data = plt.scatter( x, y, color= 'r', marker= 'o', s= 50 )
    est_data = plt.plot(x, y, color= 'b', linewidth= 3)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
def ExpPolyFit():
    xData = np.array([-0.04, 0.93, 1.95, 2.90, 3.83, 5.0, 5.98, 7.05, 8.21, 9.08, 10.09])
    yData = np.array([-8.66,-6.44,-4.36,-3,27,-0.88, 0.87, 3.31, 4.63, 6.19, 7.4, 8,85])
    while True:
        try:
            m = eval( input("\n Degree of polynomial ==>"))
            coeff = polyFit(xData, yData, m)
            print( "Coefficients are : \n", coeff )
            print( "Std, deviation = ", stdDev(coeff, xData, yData))
        except SyntaxError:
            break;

if __name__=="__main__":
    #ExpGradDescent()
    #ExpLeastSquare()
    ExpPolyFit()