import numpy as np


def get_covariance_matrix(A):
    """Compute the covariance matrix. 

    Requires A to be normalized
    """
    return A.T @ A / A.shape[0]

class PCA():
    """ Principal component analysis (PCA)

    Step 1: Center around mean
    Step 2: Normalize 
    Step 3: Apply SVD (randomized truncated SVD Halko et al. 2009)
    Step 4: Choose Eigenvectors, Thomas P. Minka:
    Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604
    Step 5: Profit
    """
    def __init__(self, 
            n_components = None
            ):
        self.n_components = n_components

    def fit(self, a):
        assert(len(a) == self.n_components)
        
