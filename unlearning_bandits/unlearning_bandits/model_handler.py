from abc import ABC, abstractmethod

class BaseModelHandler(ABC):
    """
    Abstract Base Class for a pluggable machine learning model.

    This class defines the interface for the core operations that the bandit
    agent can choose as its "arms": fast updates, fast deletions,
    and full model retraining[cite: 50].
    """

    @abstractmethod
    def fast_insert(self, data: dict) -> float:
        """
        Performs a fast, incremental update of the model.
        Returns the prediction loss after the update.
        """
        pass

    @abstractmethod
    def fast_delete(self, data: dict) -> float:
        """
        Performs a fast, efficient unlearning operation.
        Returns the prediction loss after the deletion.
        """
        pass

    @abstractmethod
    def full_retrain(self) -> float:
        """
        Performs a full, costly retraining of the model from scratch.
        Returns the prediction loss after the retraining.
        """
        pass

    @abstractmethod
    def predict(self, data: dict) -> float:
        """
        Calculates the instantaneous prediction loss for a given data point[cite: 53].
        """
        pass

import numpy as np
