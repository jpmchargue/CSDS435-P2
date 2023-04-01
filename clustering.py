"""
An abstract class representing a clustering algorithm.
"""
from abc import ABC, abstractmethod
import numpy as np

class Clustering(ABC):
    @abstractmethod
    def cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Generate cluster labels for a set of points using this algorithm.

        Args:
            distance_matrix (np.ndarray): an L x L symmetrical matrix of 
                distances between L points.

        Returns:
            np.ndarray: an array of integers, where every index i represents the ID of
                the cluster containing the ith point in the distance matrix.
        """
        pass