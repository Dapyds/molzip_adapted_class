"""
This code has been adapted from https://github.com/daenuprobst/molzip.git
"""
import gzip
import numpy as np
import pandas as pd # called for easy numpy string encoding
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import datamol as dm # called for easy joblib parallelisation

def regression(val_smiles, train_smiles, train_val, k=10):
    "Perform the regression"
    # compress first the training data to save some computational time
    ctrain_smiles = [gzip.compress(smile.encode()) for smile in train_smiles]
    pred_l = []
    for smile in val_smiles:
        pred_l.append(regression_one(smile, train_smiles, ctrain_smiles, train_val, k=k))
    return np.array(pred_l)


def projection(val_smiles, train_smiles=None, n_comp=2):
    "Perform a PCA on using the distance matrix"
    if train_smiles is None:
        train_smiles = val_smiles
    # compress first the training data to save some computational time
    ctrain_smiles = [gzip.compress(smile.encode()) for smile in train_smiles]
    dist_mat = []
    for smile in val_smiles:
        dist_mat.append(regression_one(smile, train_smiles, ctrain_smiles, [], k=10, proj=True))
    dist_mat = np.array(dist_mat)
    pca = PCA(n_components=n_comp)
    return pca.fit_transform(dist_mat)


def regression_one(x1, X_train, cX_train, y_train, k=1, proj=False):
    """
    Compute the distance of x1 with the training data

    x1 = target smiles
    X_train = list of training smiles
    cX_train = list of compressed training smiles
    y_train = list of values to fit
    """
    y_train = np.array(y_train)
    # size of the compressed input smile x1
    Cx1 = len(gzip.compress(x1.encode()))
    distance_from_x1 = []

    # compute distances from the training data
    for x2, Cx2 in zip(X_train, cX_train):
        Cx2 = len(gzip.compress(x2.encode()))
        Cx1x2 = len(gzip.compress((x1 + x2).encode()))
        ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
        distance_from_x1.append(ncd)

    if proj:
        return distance_from_x1
    else:
        distance_from_x1 = np.array(distance_from_x1)

        # find the training smiles that are the closest to the input x1
        sorted_idx = np.argsort(distance_from_x1)
        top_k_values = y_train[sorted_idx[:k]]  # take their values
        top_k_dists = distance_from_x1[sorted_idx[:k]]  # save their disatnce

        # now compute a distance average
        sum_dist = top_k_dists.sum()

        return np.matmul(top_k_dists, top_k_values) * (1./sum_dist)


def calc_ncd(c_trainxtest, ctrain, ctest):
    """Numpy version of Normalized Compression Distance computation.

    Args:
        c_trainxtest (2D ndarray): length of compressed concatenated SMILES
        ctrain (1D dnarray): length of compressed training SMILES
        ctest (1D ndarray): length of compressed testing SMILES

    Returns:
        ndarray: Normalized Compression Distance matrix
    """
    m_min = np.minimum(ctest[:, np.newaxis], ctrain[np.newaxis, :])
    m_max = np.maximum(ctest[:, np.newaxis], ctrain[np.newaxis, :])
    return (c_trainxtest - m_min) / m_max


def chunks_generator(to_chunk, chunk_size):
    """Split array-like object into chunks of given length (memory handling & parallelization).

    Args:
        to_chunk (array-like): array to split into chunks
        chunk_size (int): length of each chunk

    Returns:
        generator: generator object to iterate through
    """
    seq = range(0, len(to_chunk), chunk_size)
    return (to_chunk[i:i + chunk_size] for i in seq)


class MolZipRegressor(BaseEstimator, RegressorMixin):
    """Regressor class usable with scikit-learn tools.

    Args:
        BaseEstimator (class): Base scikit-learn estimator
        RegressorMixin (class): Base scikit-learn regressor
    """
    def __init__(self, k=1, n_jobs=1):
        """initialisation of the regressor

        Args:
            k (int, optional): k neighboors to use for prediction. Defaults to 1.
            n_jobs (int, optional): number of job to run in parallel. Defaults to 1.
        """
        self._estimator_type = "regressor" # Scikit-learn variable
        self.is_fitted_ = False # Scikit-learn variable
        self.k = k
        self.n_jobs = n_jobs
        self.compress = np.vectorize(gzip.compress) # for numpy gzip
        self.len = np.vectorize(len)
        pass
    
    
    def fit(self, X, y):
        """Fitting function (just compress the SMILES and store it).

        Args:
            X (1D array-like): list of the SMILES
            y (1D array-like): list of the activities

        Returns:
            MolZipRegressor: self object
        """
        X, y = check_X_y( # Scikit-learn data integrity check
            X, y, accept_sparse=True,
            dtype=None, ensure_2d=False
        )
        # Encode, gzip, and get length, vectorized:
        self.X_ = pd.Series(X.reshape(-1)).str.encode('ascii').values
        self.cX = self.len(self.compress(self.X_))
        self.y_ = np.array(y)
        self.is_fitted_ = True # scikit-learn variable
        return self
    
    
    def _predict(self, X):
        """Main prediction function.

        Args:
            X (1D array-like): SMILES of the molecules to predict

        Returns:
            1D array-like: regression prediction values
        """
        X = pd.Series(X.reshape(-1)).str.encode('ascii').values # encoding
        ctest = self.len(self.compress(X)) # compression
        trainxtest = X[:, None] + self.X_ # concatenate train & test
        c_trainxtest = self.len(self.compress(trainxtest)) # compress train & test
        ncd = calc_ncd(c_trainxtest, self.cX, ctest) # compute distances
        
        # find the training smiles that are the closest to the input ones
        sorted_idx = np.argsort(ncd)
        top_k_values = self.y_[sorted_idx[:, :self.k]]  # take their values
        top_k_dists = np.take_along_axis(ncd, sorted_idx[:, :self.k], axis=1)  # save their disatnce
        
        # now compute a distance average
        sum_dist = top_k_dists.sum(axis=1)
        return np.sum(top_k_dists * top_k_values, axis=1) * (1./sum_dist)
    
    
    def predict(self, X):
        """Paralellized & memory lighted prediction.

        Args:
            X (1D array-like): SMILES of the molecules to predict

        Returns:
            1D array-like: regression prediction values
        """
        X = check_array( # sckit-learn data integrity check
            X, accept_sparse=True, dtype=None, ensure_2d=False
        )
        check_is_fitted(self, 'is_fitted_')
        
        return np.concatenate(dm.parallelized( # parallelized prediction
            self._predict, chunks_generator(X, 100), n_jobs=self.n_jobs
        ))
    
    
    def get_params(self, deep=True):
        """Scikit-learn parameters handling function"""
        return dict(k=self.k)
    
    
    def set_params(self, **parameters):
        """Scikit-learn parameters handling function"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
