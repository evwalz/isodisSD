import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
import scipy.sparse as sparse
from tqdm import tqdm
import dc_stat_think as dcst
from collections import defaultdict
import osqp
#from .pava import pavaCorrect, pavaCorrect2
from .partialorders import comp_ord, tr_reduc, neighbor_points, ecdf_crps, neighborPoints, neighborPoints3, crps_gaussian_limit, neighborPoints_norm
import random
import bisect
import properscoring as ps
from _isodisSD import isocdf_seq, ecdf_comp_class_sd, ecdf_list_comp_class_sd, compOrd_cpp, normal_comp_sd, normal_comp_sd_ab, pavaCorrect_c

class predictions_idr(object):

    def __init__(self, ecdf, points, lower, upper):
        self.ecdf = ecdf
        self.points = points
        self.lower = lower
        self.upper = upper


class idrsdpredict(object):
    
    def __init__(self, predictions, incomparables):
        self.predictions = predictions
        self.incomparables = incomparables 
        
    def qscore (self, quantiles, y) :
        """
        Quantile score of IDR quantile predictions

        Parameters
        ----------
        quantiles : np array
            vector of quantiles
        y : np array
            one dimensional array of observation of the same length as predictions


        Returns
        -------
        matrix of quantile scores

        """
        if y.ndim > 1:
                raise ValueError("y must be a 1-D array")
        quantiles = np.asarray(quantiles)
        y = np.asarray(y)
        predicted = self.qpred(quantiles = quantiles)
        ly = y.size
        if ly != 1 and ly != predicted.shape[0]:
            raise ValueError("y must have length 1 or same length as predictions")
        qsvals = np.transpose(np.transpose(predicted) - y)
        qsvals2 = np.where(qsvals > 0,1,0) - quantiles
        return(2 * qsvals * qsvals2)
    
    def bscore (self, thresholds, y) :
        """
        Brier score of forecast probabilities for exceeding given thresholds

        Parameters
        ----------
        thresholds : np array
            vector of thresholds
        y : np array
            one dimensional array of observation


        Returns
        -------
        matrix of brier scores

        """
        y = np.asarray(y)
        
        if y.ndim > 1:
                raise ValueError("y must be a 1-D array")
        thresholds = np.asarray(thresholds)
        
        predicted = np.asarray(self.cdf(thresholds = thresholds))
        ly = y.size
        if ly != 1 and ly != predicted.shape[0]:
            raise ValueError("y must have length 1 or same lentgh as predictions")
        if ly == 1:
            if thresholds.size == 1:
                if y <= thresholds:
                    predicted = predicted -1
            else:
                sel_column = y <= thresholds    
                predicted[:, sel_column] = predicted[:, sel_column] - 1
        else:
            #predicted = np.subtract(predicted , np.vstack([y[k] <= thresholds for k in range(ly)]))
            
            if thresholds.size == 1:
                predicted = predicted-(y <= thresholds)
            else:
                lt = thresholds.size
                predicted = np.subtract(predicted , np.transpose(np.vstack([y <= thresholds[k] for k in range(lt)])))
        return(predicted**2)

    def pit (self, y, randomize = True, seed = None) :
        """
        Probability integral transform (PIT) of IDR

        Parameters
        ----------
        y : np array
            one dimensional array of observation
        randomize : boolean, optional
            PIT values should be randomized at discontinuity points of predictive CDF.
            The default is True.
        seed : number, optional
            seed argument for random number generator. The default is None.

        Returns
        -------
        One dimensional array of PIT values

        """
        if y.ndim > 1:
            raise ValueError("y must be a 1-D array")
        y = np.asarray(y)
        ly = y.size
        predictions = self.predictions
    
        if ly != len(predictions):
            raise ValueError("y must have same length as predictions")
    
        def pit0 (data, y):
            return(interp1d(x = np.hstack([np.min(data.points), data.points]), y = np.hstack([0,data.ecdf]), kind='previous', fill_value="extrapolate")(y))
    
        pitVals = np.array(list(map(pit0, predictions, list(y))))
        if randomize:
            sel = [x.ecdf.shape[0] for x in predictions]
            sel = np.where(np.array(sel) > 1)[0]
            if not any(sel):
                eps = 1
            else :
                preds_sel = [predictions[i] for i in sel]
                eps = np.min([np.min(np.diff(x.points)) for x in preds_sel])
            lowerPitVals = np.array(list(map(pit0, predictions, y-eps*0.5)))
            if seed is not None:
                random.seed(seed)
            sel = lowerPitVals < pitVals
            if any(sel):
                pitVals[sel] = np.random.uniform(low = lowerPitVals[sel], high = pitVals[sel], size = np.sum(sel)) 
        return(pitVals)
    
    def cdf (self, thresholds):
        """
        Cumulative distribution function (CDF) of IDR predictions

        Parameters
        ----------
        idrsdpredict : object from class idrsdpredict
        thresholds : np.array
            1-D array of thresholds at which CDF will be evaluated

        Returns
        -------
        list of probabilities giving the evaluated CDFs at given thresholds
        """
        predictions = self.predictions
        thresholds = np.asarray(thresholds)
    
        if thresholds.ndim > 1:
            raise ValueError("thresholds must be a 1-D array")
    
        if np.isnan(np.sum(thresholds)) == True:
            raise ValueError("thresholds contains nan values")
    
        def cdf0 (data):
        # f2 = interp1d(x, y, kind='next')
            return(interp1d(x = np.hstack([np.min(data.points),data.points]), y = np.hstack([0,data.ecdf]), kind='previous', fill_value="extrapolate")(thresholds))
    
        return(np.vstack(list(map(cdf0, predictions))).squeeze())

    def plot (self, index = 0, bounds = True, col_cdf = 'black', col_bounds = 'blue'):
        """
        Plot IDR predictions    

        Parameters
        ----------
        predictions : list
        index : integer value, optional
            index of prediction for which a plot is desired. The default is 0.
        bounds : boolean, optional
            whether bounds should be plotted. The default is True.
        col_cdf : color code, optional
            color of predictive CDF. The default is 'black'.
        col_bounds : color code, optional
            color of bounds. The default is 'blue'.

        Returns
        -------
        None.

        """
        predictions = self.predictions
        data = predictions[index]
        stepfun = interp1d(x = np.hstack([np.min(data.points),data.points]), y = np.hstack([0,data.ecdf]), kind='previous', fill_value="extrapolate")
        xnew = np.linspace(np.min(data.points), np.max(data.points), num=1001, endpoint=True)
        plt.plot(np.hstack([np.min(data.points),xnew]), np.hstack([0,stepfun(xnew)]), color=col_cdf)
        plt.axhline(y = 0, linestyle = ':', color = 'grey')
        plt.axhline(y = 1, linestyle = ':', color = 'grey')
        #if bounds and "upper" in data:
        if bounds and len(data.upper) > 0:
            if any(data.lower > 0 ):
                stepfun2 = interp1d(x = np.hstack([np.min(data.points),data.points]), y = np.hstack([0,data.lower]), kind='previous', fill_value="extrapolate")
                plt.plot(np.hstack([np.min(data.points),xnew]), np.hstack([0,stepfun2(xnew)]), color = col_bounds, linestyle = ':')
            else:    
                plt.hlines(y = 0, xmin = np.min(xnew), xmax = np.max(xnew), color = col_bounds, linestyle = ':')
            if any(data.upper < 1):
                stepfun3 = interp1d(x = np.hstack([np.min(data.points),data.points]), y = np.hstack([0,data.upper]), kind='previous', fill_value="extrapolate")
                plt.plot(np.hstack([np.min(data.points),xnew]), np.hstack([0,stepfun3(xnew)]), color = col_bounds, linestyle = ':')
            else:
                plt.hlines(y = 1,  xmin = np.min(xnew), xmax = np.max(xnew), color = col_bounds, linestyle = ':')
        plt.title("IDR predictive CDF")
        plt.xlabel("Thresholds")
        plt.ylabel("CDF")
    
    def crps (self, obs):
        """
        Computes the continuous rank probability score (CRPS) of IDR

        Parameters
        ----------
        predictions : list
        obs : np.array
            1-D array of observations

        Returns
        -------
        A list of CRPS values 

        """
        predictions = self.predictions
        if type(predictions) is not list:
                raise ValueError("predictions must be a list")
    
        y = np.array(obs)
    
        if y.ndim > 1:
            raise ValueError("obs must be a 1-D array")
    
        if np.isnan(np.sum(y)) == True:
            raise ValueError("obs contains nan values")
    
        if y.size != 1 and len(y) != len(predictions):
            raise ValueError("obs must have length 1 or the same length as predictions")
    
        def get_points(predictions):
            return np.array(predictions.points)
        def get_cdf(predictions):
            return np.array(predictions.ecdf)
        def modify_points(points):
            return np.hstack([points[0], np.diff(points)])
        def crps0(y, p, w, x):
            return 2*np.sum(w*(np.array((y<x))-p+0.5*w)*np.array(x-y))

        x = list(map(get_points, predictions))
        p = list(map(get_cdf, predictions))
        w = list(map(modify_points, p))
    
        return(list(map(crps0, y, p, w, x)))
    
    def qpred (self, quantiles):
        """
        Evaluate quantile function of IDR predictions 

        Parameters
        ----------
        quantiles : quantiles
            numeric vector of quantiles


        Returns
        -------
        list of forecast for desired quantiles

        """
        predictions = self.predictions
        quantiles = np.array(quantiles)
        if np.min(quantiles) < 0 or np.max(quantiles) > 1:
            raise ValueError("quantiles must be a numeric vector with entries in [0,1]")
    
        def q0 (data):
            return(interp1d(x = np.hstack([data.ecdf, np.max(data.ecdf)]), y =np.hstack([data.points,data.points[-1]]) ,kind='next', fill_value="extrapolate")(quantiles))

        return(np.vstack(list(map(q0, predictions))).squeeze())
    
        

class idrcal:
        def __init__(self, ecdf, thresholds, indices, X, y, constr, input_type, grid, org_crps):
            self.ecdf = ecdf
            self.thresholds = thresholds 
            self.indices = indices
            self.X = X
            self.y = y
            self.constr = constr
            self.grid = grid
            self.input_type = input_type
            self.org_crps = org_crps

            
        def predict(self, data=None, digits = 3):
            """
            Prediction based on IDR model fit

            Parameters
            ----------
            idr_object : object from class idrobject
            data : pd.DataFrame, optional
                containing variables with which to predict. The default is None.
            digits : integer value, optional
                digits number of decimal places for predictive CDF. 
                The default is 3.

            Returns
            -------
            object of class idrpredict.
            predictions : Object of class predictions_idr:
                points : where predictie CDF has jumps
                cdf : estimated CDF evaluated at points
                lower : bounds for estimated CDF (out-of-sample predictions)
                upper : bounds for estimated CDF (out-of-sample predictions)
            incomparables : gives the indices of all predictions for which the 
                climatological forecast is returned because the forecast variables are not 
                comparable to the training data. None if not available.

            """
    
            cdf = self.ecdf.copy()
            thresholds = self.thresholds.copy()
            order_indices = []
            preds = []
            if data is None:
                indices = self.indices
                for i in range(indices.shape[0]):
                    edf = np.round(cdf[i,:], digits)
                    sel = np.hstack([edf[0] > 0, np.diff(edf) > 0])
                    #dat = {'points': thresholds[sel], 'cdf': edf[sel]}
                    #tmp = pd.DataFrame(dat, columns = ['points', 'cdf'])
                    tmp = predictions_idr(ecdf = edf[sel], points = thresholds[sel], lower = [], upper = [])
                    for j in indices[i]:
                        order_indices.append(j) 
                        preds.append(tmp)
                preds_rearanged = [preds[k] for k in np.argsort(order_indices)]
                idr_predictions = idrsdpredict(predictions = preds_rearanged, incomparables = None) 
                return(idr_predictions)
    
            
            input_type = self.input_type
            X = self.X.copy()

            if (input_type == 'idr'):
                if type(data).__name__ == 'idrpredict':
                    predictions = data.predictions
                    if type(predictions) is not list:
                        raise ValueError("predictions must be a list")
                    def get_points(predictions):
                        return np.array(predictions.points)
                    def get_cdf(predictions):
                        return np.array(predictions.ecdf)
                    grid = list(map(get_points, predictions))
                    data = list(map(get_cdf, predictions))
                else:
                    raise ValueError("wrong 'data' for type 'idr'")

            if (input_type == 'ecdf' or input_type == 'idr'):
                grid_base = self.grid.copy()
                M = self.constr[0]
                #raise ValueError("ecdf and idr not yet implemented")
                smaller, greater = neighborPoints3(data, X, grid, grid_base, M)
            
            elif (input_type == 'ensemble'):
                M = self.constr[0]
                col_check = all(elem in data.columns for elem in X.columns)
                if col_check == False:
                    raise ValueError("some variables of idr fit are missing in data")
                smaller, greater = neighborPoints(data, X, M)
                #return neighborPoints(data, X, M)


            elif (input_type == 'normal' or input_type == 'normal_ab'):
                M = self.constr[0]
                smaller, greater = neighborPoints_norm(data, X, M)
                
            #elif (input_type == 'dis'):
            #    raise ValueError("dis not yet implemented")
            
            else:
                raise ValueError("wrong input_type")
            
            incomparables =  np.array(list(map(len, smaller)))+np.array(list(map(len, greater))) == 0

            if any(incomparables):
                y = self.y
                edf = np.round(dcst.ecdf_formal(thresholds, y.explode()), digits)
                sel = edf > 0
                edf = edf[sel]
                points = thresholds[sel]
                upr = np.where(edf == 1)[0]
                if upr < len(edf)-1:
                    points = np.delete(points, np.arange(upr, len(edf)))
                    edf = np.delete(edf, np.arange(upr, len(edf)))                    
                #dat = {'points':points, 'lower':edf, 'cdf':edf, 'upper':edf}
                #tmp = pd.DataFrame(dat, columns = ['points', 'lower', 'cdf', 'upper'])
                tmp = predictions_idr(ecdf = edf, points = points, lower = edf, upper = edf)
                for i in np.where(incomparables == True)[0]:
                    preds.append(tmp)
                    order_indices.append(i)

            for i in np.where(incomparables == False)[0]:
                if smaller[i].size>0 and greater[i].size == 0:
                    upper = np.round(np.amin(cdf[smaller[i].astype(int),:], axis=0), digits)
                    sel = np.hstack([upper[0] != 0, np.diff(upper) != 0])
                    upper = upper[sel] 
                    lower = np.zeros(len(upper))
                    estimCDF = upper
                elif smaller[i].size == 0 and greater[i].size > 0:
                    lower = np.round(np.amax(cdf[greater[i].astype(int),:], axis=0), digits)
                    sel = np.hstack([lower[0] != 0, np.diff(lower) != 0]) 
                    lower = lower[sel] 
                    upper = np.ones(len(lower))
                    estimCDF = lower
                else: 
                    lower = np.round(np.amax(cdf[greater[i].astype(int),:], axis=0), digits)
                    upper = np.round(np.amin(cdf[smaller[i].astype(int),:], axis=0), digits)
                    sel = np.hstack([lower[0] != 0, np.diff(lower) != 0]) + np.hstack([upper[0] != 0, np.diff(upper) != 0])
                    lower = lower[sel]
                    upper = upper[sel]
                    estimCDF = np.round(0.5*(lower+upper), digits)
              
                #dat = {'points': thresholds[sel], 'lower': lower, 'cdf': estimCDF, 'upper': upper}
                #tmp = pd.DataFrame(dat, columns = ['points', 'lower', 'cdf', 'upper'])
                tmp = predictions_idr(ecdf = estimCDF, points = thresholds[sel], lower = lower, upper = upper)
                order_indices.append(i)
                preds.append(tmp)

            preds_rearanged = [preds[k] for k in np.argsort(order_indices)]
            idr_predictions = idrsdpredict(predictions = preds_rearanged, incomparables = np.where(incomparables))  
    #return preds_rearanged             
            return idr_predictions  



def prepareData(X, groups, orders):
    """
    Prepare data fir IDR modeling with given orders    
    
    Parameters
    ----------
    X : pd.DataFrame of covariates
    groups : dictionary 
        assigning column names of X to groups
    orders : dictionary 
        assigning groups to type of ordering


    Returns
    -------
    X : pd.DataFrame 

    """
    res = defaultdict(list) 
    for key, val in sorted(groups.items()): 
        res[val].append(key) 
    for key, val in res.items():
        if len(val)>1:
            if orders[str(int(key))] == "comp":
                continue
            tmp = -np.sort(-X[val], axis=1)
            if orders[str(int(key))] == "sd":
                X[val] = tmp 
            else:
                X[val] = np.cumsum(tmp, axis=1)
    return X

def idrsd (y, X = None,grid = None, dis_func = None, input_type = 'ensemble' ,inta = None, intb = None, org_crps = False, verbose = False, max_iter = 10000, eps_rel = 0.00001, eps_abs = 0.00001, progress = True, *args):
    """
    Fits isotonic distirbutional regression (IDR) to a training dataset. 

    Parameters
    ----------
    y : np.array
        one dimensional array (response variable)
    X : distributional forecast specified by 'input_type'
    input_type : input_type of input_type data X
    grid : threshold values of ECDFs in X, only required if X is matrix of ECDFs.
    dis_func : specifies parametric forecast distribution. Note that parameters of distribution must be provided 
    inta : lower bound for grid computation of isodisSD
    intb : upper bound for grid computation of isodisSD
    (OSQP Solver Setting)
    verbose : boolean
        print output of OSQP solver. The default is False
    max_iter : maximum number of iterations
    eps_rel : relative tolerance
    epc_abs : absolute tolerance


    Returns
    -------
    An object of class idrobject containing the components:
        X: data frame of distinct covariate combinations used for fit
        y: list of responses for given covariate combination
        cdf: estimated CDF
        thresholds: where CDF is evaluated
        groups: groups used for estimation
        orders: orders used for estimation
        indices: indices of covariates in original dataset
        constraints: in multivariate IDR is None, 
        otherwise the order constraints for optimization
    """
    #if not isinstance(X, pd.DataFrame):
    #   raise ValueError("X must be a pandas data frame")
    
    y = np.asarray(y)
    
    if y.ndim > 1:
        raise ValueError('idr only handles 1-D arrays of observations')
    
    if np.isnan(np.sum(y)) == True:
        raise ValueError("y contains nan values")
	 
    
    #M =  all(elem in groups.keys() for elem in X.columns)  
    #if M == False:
    #    raise ValueError("some variables must be used in groups and in X")
    
    thresholds = np.sort(np.unique(y))
    nThr = len(thresholds)
    
    if nThr == 1:
        raise ValueError("y must contain more than 1 distinct value")
    
    original_crps = None
    
    if input_type == 'idr':
        if type(X).__name__ == 'idrpredict':
            if (org_crps):
                original_crps = np.mean(X.crps(y))
            predictions = X.predictions
            if type(predictions) is not list:
                raise ValueError("predictions must be a list")
            def get_points(predictions):
                return np.array(predictions.points)
            def get_cdf(predictions):
                return np.array(predictions.ecdf)
            grid = list(map(get_points, predictions))
            X = list(map(get_cdf, predictions))
        else:
            raise ValueError('wrong X for input_type idr')

    if input_type == 'ensemble':
        if isinstance(X, pd.DataFrame) == False:
            raise ValueError("data must be a pandas data frame")
    
        if X.shape[0] <= 1:
            raise ValueError('X must have more than 1 row')

        if X.isnull().values.any() == True:
            raise ValueError("X contains nan values")
        
        if y.size != X.shape[0]:
            raise ValueError("length of y must match number of rows in X")
        
        #nVar = X.shape[1]
        #oldNames = X.columns 

        X_np = X.to_numpy()

        if org_crps:
            original_crps = np.mean(ps.crps_ensemble(y, X_np))

        Xp = pd.DataFrame(np.sort(X_np, axis=1))
        nVar = Xp.shape[1]
        oldNames = Xp.columns 
        Xp['y'] = y
        Xp['ind'] = np.arange(len(y))
        tt = list(oldNames)
        X_grouped = Xp.groupby(tt).agg({'y':list, 'ind':list})
        X_grouped = X_grouped.sort_values(by=list(oldNames)[::-1])
        X_grouped = X_grouped.reset_index()
        cpY = X_grouped["y"]
        indices = X_grouped["ind"]
        Xp = X_grouped[tt]
        lenlist = np.vectorize(len)
        weights = lenlist(indices)
        N = Xp.shape[0]
        # implement function compOrd_cpp
        M = compOrd_cpp(Xp.to_numpy())
        M = M.astype('int')
        paths = np.column_stack(np.where(M == 1))
        constr = list()
        constr.append(M)
        constr.append(paths)
        

    elif input_type == 'normal':
        # Check if X has two columns
        if X.shape[1] != 2:
            raise ValueError("'X' must contain two columns with mean and std")

        # Check if X contains numeric variables
        if not all(np.issubdtype(col.dtype, np.number) for col in X.T):
            raise ValueError("'X' must contain numeric variables")

        # Check if X and y contain NAs
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("'X' and 'y' must not contain NAs")

        # Check if length(y) and nrow(X) match
        if len(y) != X.shape[0]:
            raise ValueError("length(y) and nrow(X) must match")

        # Check if sigma is positive
        if np.any(X[:, 1] < 0):
            raise ValueError("sigma must be positive")
        
        if org_crps:
            original_crps = np.mean(ps.crps_gaussian(y, X[:,0], X[:,1]))

        Xp = pd.DataFrame(data = {'mu':X[:,0], 'sg':X[:, 1]})
        oldNames = Xp.columns 
        tt = list(oldNames)
        Xp['y'] = y
        Xp['ind'] = np.arange(len(y))
        class_X_grouped = Xp.groupby(tt).agg({'y':list, 'ind':list})
        class_X_grouped = class_X_grouped.reset_index()
        cpY = class_X_grouped["y"]
        indices = class_X_grouped["ind"]
        lenlist = np.vectorize(len)
        weights = lenlist(indices)
        Xp = class_X_grouped[tt].to_numpy()
        M = normal_comp_sd(Xp)
        M = M.astype('int')
        paths = np.column_stack(np.where(M == 1))
        constr = list()
        constr.append(M)
        constr.append(paths)
        nVar = 2
        N = Xp.shape[0]
            
    elif input_type == 'normal_ab':
        # Check if X has two columns
        if X.shape[1] != 2:
            raise ValueError("'X' must contain two columns with mean and std")

        # Check if X contains numeric variables
        if not np.all(np.apply_along_axis(lambda col: np.issubdtype(col.dtype, np.number), 0, X)):
            raise ValueError("'X' must contain numeric variables")

        # Check if X and y contain NAs
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("'X' and 'y' must not contain NAs")

        # Check if length(y) and nrow(X) match
        if X.shape[0] != len(y):
            raise ValueError("length(y) and nrow(X) must match")

        # Check if sigma is positive
        if not np.all(X[:, 1] >= 0):
            raise ValueError("sigma bust be positive")

        # Check if inta and intb are defined
        if inta is None or intb is None:
            raise ValueError("define a and b")

        # Check if inta is smaller than intb
        if inta >= intb:
            raise ValueError("a must be smaller than b")
        

        if org_crps:
            original_crps = np.mean(crps_gaussian_limit(y, X[:,0], X[:,1], inta, intb))

        Xp = pd.DataFrame(data = {'mu':X[:,0], 'sg':X[:, 1]})
        oldNames = Xp.columns 
        tt = list(oldNames)
        Xp['y'] = y
        Xp['ind'] = np.arange(len(y))
        class_X_grouped = Xp.groupby(tt).agg({'y':list, 'ind':list})
        class_X_grouped = class_X_grouped.reset_index()
        cpY = class_X_grouped["y"]
        indices = class_X_grouped["ind"]
        lenlist = np.vectorize(len)
        weights = lenlist(indices)
        Xp = class_X_grouped[tt].to_numpy()
        M = normal_comp_sd_ab(Xp, inta, intb)
        M = M.astype('int')
        paths = np.column_stack(np.where(M == 1))
        constr = list()
        constr.append(M)
        constr.append(paths)
        nVar = 2
        N = Xp.shape[0]

    elif input_type == 'ecdf' or input_type == 'idr':
        # include checks
        if org_crps:
            if input_type == 'ecdf':
                original_crps = np.mean(ecdf_crps(y, grid, X))

        if type(X) == list:
            M_class = ecdf_list_comp_class_sd(X, grid)
        else:
            
            M_class = ecdf_comp_class_sd(X, grid)

        
        M = M_class[0]
        class_X = M_class[1]
        #print(class_X)
        nVar = len(class_X)
        
        
        class_X = pd.DataFrame(data = {'cy' : class_X})
        
        class_X['y'] = y
        class_X['ind'] = np.arange(len(y))
        
        class_X_grouped = class_X.groupby('cy').agg({'y':list, 'ind':list})
        class_X_grouped = class_X_grouped.reset_index()
        cpY = class_X_grouped["y"]
        indices = class_X_grouped["ind"]
        #print(indices)
        
        vec_indx = np.asarray([x[0] for x in indices])
        #return M_class
        
        M = M[vec_indx, :][:, vec_indx]
        M = M.astype('int')
        paths = np.column_stack(np.where(M == 1))
        constr = list()
        constr.append(M)
        constr.append(paths)
        lenlist = np.vectorize(len)
        weights = lenlist(indices)
        N = len(cpY)
        # reorder X:
        if type(X) == list:
            Xp = [X[i] for i in vec_indx]
            grid_cp = grid.copy()
            grid = [grid_cp[i] for i in vec_indx]
            
        else:
            Xp = X[vec_indx,]

    #elif input_type == 'dis':
    #    raise ValueError('input_type dis not yet implemented')    

    else:
        raise ValueError('invalid value for input_type')
        

    
    cdf = np.zeros((N,nThr-1))
    A = tr_reduc(constr[1], N)
    nConstr = A.shape[1]
    l = np.zeros(nConstr) 
    A = sparse.csc_matrix((np.repeat([1,-1],nConstr), (np.tile(np.arange(nConstr),2),A.flatten())), shape=(nConstr, N))
    P = sparse.csc_matrix((weights, (np.arange(N),np.arange(N))))
    i = 0
    I = nThr -1
    #conv =  np.full(I, False, dtype=bool) 
    q = -weights*np.array(cpY.apply(lambda x: np.mean(np.array(x) <= thresholds[i]))) 
    m = osqp.OSQP()
    m.setup(P=P, q=q, A=A, l=l, verbose = verbose, max_iter = max_iter, eps_rel = eps_rel, eps_abs = eps_abs) 
    sol = m.solve()
    pmax = np.where(sol.x>0,sol.x,0)
    cdf[:,0] = np.where(pmax<1, pmax, 1) 
    if I > 1:
        if progress:
            for i in tqdm(range(1,I)):
                m.warm_start(x = cdf[:, i - 1])
                q = -weights*np.array(cpY.apply(lambda x: np.mean(np.array(x) <= thresholds[i]))) 
                m.update(q = q)
                sol = m.solve()
                pmax = np.where(sol.x>0,sol.x,0)
                cdf[:, i] =np.where(pmax<1, pmax, 1)
        else:
            for i in range(1,I):
                m.warm_start(x = cdf[:, i - 1])
                q = -weights*np.array(cpY.apply(lambda x: np.mean(np.array(x) <= thresholds[i]))) 
                m.update(q = q)
                sol = m.solve()
                pmax = np.where(sol.x>0,sol.x,0)
                cdf[:, i] =np.where(pmax<1, pmax, 1)
                

    
    if nVar > 1:
        
        #cdf_cl = pavaCorrect2(cdf)
        #cdf2 = np.ones((N,nThr))
        #cdf2[:,:-1] = cdf_cl
        
        cdf = pavaCorrect_c(cdf)
        cdf1 = np.ones((N,nThr))
        cdf1[:,:-1] = cdf
    
    idr_object = idrcal(ecdf = cdf1, thresholds = thresholds, indices = indices, X = Xp, y = cpY, constr = constr, grid = grid, input_type = input_type, org_crps = original_crps)
    return(idr_object)


def crps_unc(obs):
    obs_ensemble = np.tile(obs, (len(obs), 1))
    return np.mean(ps.crps_ensemble(obs, obs_ensemble))

def crps_ecdf(y, grid, ecdf):
    x = [grid] * len(y)
    p = [ecdf[i, :] for i in range(ecdf.shape[0])]
    w = [[p_i[0]] + list(np.diff(p_i)) for p_i in p]

    def crps0(y, p, w, x):
        return 2 * np.sum(w * ((y < x) - p + 0.5 * w) * (x - y))

    return [crps0(y_i, p_i, w_i, x_i) for y_i, p_i, w_i, x_i in zip(y, p, w, x)]  


def isodeco_crps(y, X = None,grid = None, dis_func = None, input_type = 'ensemble' ,inta = None, intb = None, org_crps = True, verbose = False, max_iter = 10000, eps_rel = 0.00001, eps_abs = 0.00001, progress = True, *args):
    cali_idr =  idrsd(y = y, X = X,grid = grid, dis_func = dis_func, input_type = input_type ,inta = inta, intb = intb, org_crps = True, verbose = verbose, max_iter = max_iter , eps_rel = eps_rel, eps_abs = eps_abs, progress = progress, *args)

    cali_preds = cali_idr.predict()
    cali_crps = np.mean(cali_preds.crps(y))

    uncertainty = crps_unc(y)

    crps_original = cali_idr.org_crps

    result = {
    'MCB': crps_original - cali_crps,
    'DSC': uncertainty - cali_crps,
    'UNC': uncertainty,
    'CRPS': crps_original
    }   
    return result