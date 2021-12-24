
#https://www.cnblogs.com/21207-iHome/p/6038853.html

import numpy as np

def best_fit_transform( A, B ):
    #calculates the least-squares best-fit transform between corresponding 3D points
    assert len( A ) == len( B )

    centroid_A = np.mean( A, axis= 0 )
    centroid_B = np.mean( B, axis= 0 )
    AA = A - centroid_A
    BB = B - centroid_B

    W = np.dot( BB.T, AA )
    U, s, VT = np.linalg.svd( W )
    R = np.dot( U, VT )

    if np.linalg.det( R ) < 0:
        VT[2, :] *= -1
        R = np.dot( U, VT )

    t = centroid_B.T - np.dot( R, centroid_A.T )

    T = np.identity( 4 )
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T, R, t

def nearest_neighbor( src, dst ):
    indecies  = np.zeros( src.shape[0], dtype= np.int )
    distances = np.zeros( src.shape[0] )

    for i, s, in enumerate( src ):
        min_dist = np.inf
        for j, d in enumerate( dst ):
            dist = np.linalg.norm( s - d )
            if dist < min_dist:
                min_dist = dist
                indecies[i] = j
                distances[ i ] = dist
    return distances, indecies

def icp( A, B, init_pose= None, max_iterations= 50, tolerance= 0.001 ):
    src = np.ones( (4, A.shape[0]) )
    dst = np.ones( (4, B.shape[0]) )
    src[0:3, :] = np.copy( A.T )
    dst[0:3, :] = np.copy( B.T )

    if init_pose is not None:
        src = np.dot( init_pose, src )

    prev_error = 0

    for i in range( max_iterations ):
        distances, indices = nearest_neighbor( src[0:3, :].T, dst[0:3, :].T )
        T, _, _ = best_fit_transform( src[0:3, :].T, dst[0:3, indices].T )

        src = np.dot( T, src )

        mean_error = np.sum( distances ) / distances.size
        if abs( prev_error - mean_error ) < tolerance:
            break
        prev_error = mean_error

    T, _, _ = best_fit_transform( A, src[0:3, :].T )

    return T, distances

if __name__ == "__main__":
    A = np.random.randint( 0, 10.1, (20, 3) )

    rotz = lambda  theta: np.array([[np.cos(theta),-np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [            0,             0, 1]])

    trans = np.array( [2.12, -0.2, 1.3] )
    B = A.dot( rotz( np.pi/4 ).T + trans )

    T, distances = icp( A, B )

    np.set_printoptions( precision= 3, suppress= True )
    print( T )
