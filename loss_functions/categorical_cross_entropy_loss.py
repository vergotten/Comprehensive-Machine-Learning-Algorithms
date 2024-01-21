import torch
import torch.nn as nn
import numpy as np

class CategoricalCrossEntropy_Loss:
    """
    A class used to represent a Categorical Cross Entropy Loss

    ...

    Methods
    -------
    categorical_cross_entropy_manual(y_pred, y_true)
        Calculates the Categorical Cross Entropy loss manually.
    categorical_cross_entropy_pytorch(y_pred, y_true)
        Calculates the Categorical Cross Entropy loss using PyTorch's built-in function.
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the CategoricalCrossEntropy_Loss object.
        """
        pass

    @staticmethod
    def categorical_cross_entropy_manual(y_pred, y_true):
        """
        Calculates the Categorical Cross Entropy loss manually.

        Parameters:
            y_pred (torch.Tensor): Predicted values, a tensor of shape (N, C) where N - number of samples, C - number of channels.
            y_true (torch.Tensor): True values, a tensor of shape (N,) where N - number of samples.

        Returns:
            float: Calculated Categorical Cross Entropy loss.
        """
        def softmax(x):
            e_x = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0]) # Subtract max for numerical stability
            return e_x / torch.sum(e_x, dim=1, keepdim=True)

        N = y_pred.shape[0]
        y_pred = softmax(y_pred)
        y_pred = y_pred.clamp(min=1e-7, max=1-1e-7)  # Add a smoothing factor for numerical stability
        log_likelyhood = -torch.log(y_pred[torch.arange(N).long(), y_true.long()])
        loss = torch.sum(log_likelyhood) / N
        return loss

    @staticmethod
    def categorical_cross_entropy_pytorch(y_pred, y_true):
        """
        Calculates the Categorical Cross Entropy loss using PyTorch's built-in function.

        Parameters:
            y_pred (torch.Tensor): Predicted values, a tensor of shape (N, C) where N - number of samples, C - number of channels.
            y_true (torch.Tensor): True values, a tensor of shape (N,) where N - number of samples.

        Returns:
            float: Calculated Categorical Cross Entropy loss.
        """
        loss = nn.CrossEntropyLoss()
        return loss(y_pred, y_true)


if __name__ == "__main__":
    def categorical_cross_entropy_test(N = 10, C = 3):
        """
        Tests the CategoricalCrossEntropy_Loss class.

        Parameters:
            N (int): Number of samples.
            C (int): Number of channels.
        """
        y_true = torch.randint(C, size=(N,)); print(f"y_true: {y_true}")
        y_pred = torch.randn(N, C, dtype=torch.float32); print(f"y_pred: {y_pred}")

        categorical_cross_entropy_manual = CategoricalCrossEntropy_Loss.categorical_cross_entropy_manual(y_pred, y_true)
        print(f"categorical_cross_entropy_manual: {categorical_cross_entropy_manual}")

        categorical_cross_entropy_pytorch = CategoricalCrossEntropy_Loss.categorical_cross_entropy_pytorch(y_pred, y_true)
        print(f"categorical_cross_entropy_pytorch: {categorical_cross_entropy_pytorch}")

    categorical_cross_entropy_test()
