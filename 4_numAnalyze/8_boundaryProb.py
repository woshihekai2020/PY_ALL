
import numpy as np


def linInterp( f, x1, x2 ):
    f1 = f(x1)
    f2 = f(x2)
    return  x2 - f2 * (x2 - x1) / (f2 - f1)