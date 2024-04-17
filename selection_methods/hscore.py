import time
import numpy as np

class HScore(object):
    def __init__(self, args):
        self.args = args

    def hscore(self, features: np.ndarray, labels: np.ndarray):
        r"""
        H-score in `An Information-theoretic Approach to Transferability in Task Transfer Learning (ICIP 2019) 
        <http://yangli-feasibility.com/home/media/icip-19.pdf>`_.
        
        The H-Score :math:`\mathcal{H}` can be described as:

        .. math::
            \mathcal{H}=\operatorname{tr}\left(\operatorname{cov}(f)^{-1} \operatorname{cov}\left(\mathbb{E}[f \mid y]\right)\right)
        
        where :math:`f` is the features extracted by the model to be ranked, :math:`y` is the groud-truth label vector

        Args:
            features (np.ndarray):features extracted by pre-trained model.
            labels (np.ndarray):  groud-truth labels.

        Shape:
            - features: (N, F), with number of samples N and feature dimension F.
            - labels: (N, ) elements in [0, :math:`C_t`), with target class number :math:`C_t`.
            - score: scalar.
        """
        f = features
        y = labels

        covf = np.cov(f, rowvar=False)
        C = int(y.max() + 1)
        g = np.zeros_like(f)

        for i in range(C):
            Ef_i = np.mean(f[y == i, :], axis=0)
            g[y == i] = Ef_i

        covg = np.cov(g, rowvar=False)
        score = np.trace(np.dot(np.linalg.pinv(covf, rcond=1e-15), covg))

        return score
    
    def score(self, features, labels):

        start_time = time.time()
        score = self.hscore(features, labels)
        end_time = time.time()

        return score, end_time - start_time
