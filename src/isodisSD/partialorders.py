import numpy as np
import scipy.stats
import pandas as pd
from scipy.stats import norm
pd.options.mode.chained_assignment = None

from _isodisSD import new_func2_sd, new_func_list_sd, indx_norm_sd
#, new_func_mat_sd, new_func_mat_list_sd, new_func_list_sd, new_func_list_mat_sd

def comp_ord(X):
    """Componentwise partial order on rows of array x.

    Compares the columns x[j, j] of a matrix x in the componentwise
    order.


    Parameters
    ----------
    x : np.array
        Two-dimensional array with at least two columns.
    Returns
    -------
    paths : np.array
        Two-column array, containing in the first coolumns the indices
        of the rows of x that are smaller in the componentwise order,
        and in the second column the indices of the corresponding
        greater rows.
    col_order : np.array
        Array of the same dimension of x, containing in each column the
        order of the column in x.
    """
    X = np.asarray(X)
    if X.ndim != 2 | X.shape[0] < 2:
        raise ValueError("X should have at least two rows")
    Xt = X.transpose()
    m = Xt.shape[1]
    d = Xt.shape[0]
    colOrder = np.argsort(Xt, axis=1)
    ranks = np.apply_along_axis(scipy.stats.rankdata, 1, Xt, method='max')
    smaller = []
    greater = []
    for k in range(m):
        nonzeros = np.full((m), False)
        nonzeros[colOrder[0,0:ranks[0,k]]] = True
    
        for l in range(1,d):
            if ranks[l,k]<m:
                nonzeros[colOrder[l,ranks[l,k]:m]] = False
        nonzeros = np.where(nonzeros)[0]
        n_nonzeros = nonzeros.shape[0]
        smaller.extend(nonzeros)
        greater.extend([k]*n_nonzeros)
    paths = np.vstack([smaller, greater]) 
    return paths, colOrder.transpose()


def tr_reduc(paths, n):
    """Transitive reduction of path matrix.

    Transforms transitive reduction of a directed acyclic graph.

    Parameters
    ----------
    x : np.array
        Two-dimensional array containing the indices of the smaller
        vertices in the first row and the indices of the
        greater vertices in the second row.
    Returns
    -------

    """
    edges = np.full((n, n), False)
    edges[paths[:,0], paths[:,1]] = True
    np.fill_diagonal(edges, False)
    for k in range(n):
        edges[np.ix_(edges[:, k], edges[k])] = False
    edges = np.array(np.nonzero(edges))
    return edges


def neighbor_points(x, X, order_X):
    """    
    Neighbor points with respect to componentwise order
    
    Parameters
    ----------
    x : np.array
        Two-dimensional array 
    X : Two dimensional array with at least to columns
    order_X : output of function compOrd(X)

    Returns
    -------
    list given for each x[i,] the indices 
    of smaller and greater neighbor points within the rows of X

    """
    X = np.asarray(X)
    x = np.asarray(x)                
    col_order = order_X[1]

    nx = x.shape[0]
    k = x.shape[1]
    n = X.shape[0]
    ranks_left = np.zeros((nx,k))
    ranks_right = np.zeros((nx,k))
    for j in range(k):
        ranks_left[:,j] = np.searchsorted(a = X[:,j], v = x[:,j], sorter = col_order[:,j])
        ranks_right[:,j] = np.searchsorted(a = X[:,j], v = x[:,j], side = "right", sorter = col_order[:,j])

    x_geq_X = np.full((n, nx), False)
    x_leq_X = np.full((n, nx), True)
    for i in range(nx):
        if ranks_right[i,0] > 0:
            x_geq_X[col_order[0:int(ranks_right[i,0]),0],i] = True
        if ranks_left[i,0] > 0:
            x_leq_X[col_order[0:int(ranks_left[i,0]),0],i] = False
        for j in range(1, k):
            if ranks_right[i,j] < n:
                x_geq_X[col_order[int(ranks_right[i,j]):n,j],i] = False
            if ranks_left[i,j] > 0:
                x_leq_X[col_order[0:int(ranks_left[i,j]),j],i] = False
    paths = np.full((n,n), False)
    paths[order_X[0][0], order_X[0][1]] = True
    np.fill_diagonal(paths, False)  

    for i in range(n):
        x_leq_X[np.ix_(paths[i,:], x_leq_X[i, :])] = False
        x_geq_X[np.ix_(paths[:, i], x_geq_X[i, :])] = False

    smaller = []
    greater = []

    for i in range(nx):
        smaller.append(x_geq_X[:,i].nonzero()[0])
        greater.append(x_leq_X[:,i].nonzero()[0])
        
    return smaller, greater




def ecdf_crps(y, grid, X):
    dim_n = len(y)
    
    def get_cdf(i):
        return X[i,]
    def get_grid(i):
        return grid
    def modify_points(points):
        return np.hstack([points[0], np.diff(points)])
    def crps0(y, p, w, x):
        return 2*np.sum(w*(np.array((y<x))-p+0.5*w)*np.array(x-y))
    
    p = list(map(get_cdf, np.arange(dim_n)))
    x = list(map(get_grid, np.arange(dim_n)))
    w = list(map(modify_points, p))

    return(list(map(crps0, y, p, w, x))) 


def crps_gaussian_limit(y, mu, scale, inta, intb):
    y = y-mu
    lower = inta / scale
    upper = intb / scale
    out_l1 = out_u1 = out_l2 = 0
    out_u2 = 1
    z = y.copy()
    ###
    p_l = norm.cdf(lower)
    out_l1 = -lower * p_l**2 - 2 * norm.pdf(lower) * p_l
    #out_l1[lower == -Inf] <- 0
    out_l2 = norm.cdf(lower, scale = np.sqrt(0.5))
    z = np.maximum(lower, z)
    p_u = 1-norm.cdf(upper)
    out_u1 = upper * p_u**2 - 2 * norm.pdf(upper) * p_u
    #out_u1[upper == Inf] <- 0
    out_u2 = norm.cdf(upper, scale = np.sqrt(0.5))
    z = np.minimum(upper, z)
    b = out_u2 - out_l2
    out_z = z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z)
    
    out = out_z + out_l1 + out_u1 - b / np.sqrt(np.pi)
    out[(lower > upper) | (lower == upper)] = np.nan
    out[lower == upper] = 0
    ###
    return out + np.absolute(y - z)

def neighborPoints(x, X, M):
    x = np.sort(x, axis=1)
    X = np.array(X)

    mX = X.shape[0]
    mx = x.shape[0]

    list_indx = new_func2_sd(X, x)
    #return list_indx

    smaller_indx = list_indx[0]
    greater_indx = list_indx[1]

    smaller = list()
    greater = list()

    for i in range(mx):

        ix1 = np.where(smaller_indx[i] == 1)[0]
        ix2 = np.where(greater_indx[i] == 1)[0]

        Msmallerreduced = M[np.ix_(ix1, ix1)]
        smaller.append(ix1[np.where(np.sum(Msmallerreduced, axis=1) == 1)[0]])

        Mgreaterreduced = M[np.ix_(ix2, ix2)]
        greater.append(ix2[np.where(np.sum(Mgreaterreduced, axis=0) == 1)[0]])

    return smaller, greater


def neighborPoints3(x, X, gridx, gridX, M):
    if isinstance(gridx, np.ndarray) and isinstance(x, np.ndarray):
        mx = x.shape[0]
        if isinstance(X, np.ndarray):
            raise ValueError("not yet implemented for input type of 'data' and/or 'grid'")
            #list_indx = new_func_mat_sd(X, x, gridx, gridX)
        else:
            raise ValueError("not yet implemented for input type of 'data' and/or 'grid'")
            #list_indx = new_func_mat_list_sd(X, x, gridx, gridX)
    elif isinstance(x, list) and isinstance(gridx, list):
        mx = len(x)
        if isinstance(X, list):
            list_indx = new_func_list_sd(X, x, gridx, gridX)
        else:
            raise ValueError("not yet implemented for input type of 'data' and/or 'grid'")
            #list_indx = new_func_list_mat_sd(X, x, gridx, gridX)
    else:
        raise ValueError("Wrong input type for 'data' and/or 'grid'")

    smaller_indx = list_indx[0]
    greater_indx = list_indx[1]

    smaller = list()
    greater = list()

    for i in range(mx):

        ix1 = np.where(smaller_indx[i] == 1)[0]
        ix2 = np.where(greater_indx[i] == 1)[0]

        Msmallerreduced = M[np.ix_(ix1, ix1)]
        smaller.append(ix1[np.where(np.sum(Msmallerreduced, axis=1) == 1)[0]])

        Mgreaterreduced = M[np.ix_(ix2, ix2)]
        greater.append(ix2[np.where(np.sum(Mgreaterreduced, axis=0) == 1)[0]])

    return smaller, greater


def neighborPoints_norm(data, X, M):
    mx = data.shape[0]

    list_indx = indx_norm_sd(X, data)

    smaller_indx = list_indx[0]
    greater_indx = list_indx[1]

    smaller = list()
    greater = list()

    for i in range(mx):

        ix1 = np.where(smaller_indx[i] == 1)[0]
        ix2 = np.where(greater_indx[i] == 1)[0]

        Msmallerreduced = M[np.ix_(ix1, ix1)]
        smaller.append(ix1[np.where(np.sum(Msmallerreduced, axis=1) == 1)[0]])

        Mgreaterreduced = M[np.ix_(ix2, ix2)]
        greater.append(ix2[np.where(np.sum(Mgreaterreduced, axis=0) == 1)[0]])

    return smaller, greater




