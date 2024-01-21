import torch
import torch.nn as nn
import numpy as np

class MSE_Loss:
    """
    A class used to represent an MSE Loss

    ...

    Methods
    -------
    mse_manual(y_pred, y_true)
        Calculates the Mean Squared Error loss manually.
    mse_pytorch(y_pred, y_true)
        Calculates the Mean Squared Error loss using PyTorch's built-in function.
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the MSE_Loss object.
        """
        pass

    @staticmethod
    def mse_manual(y_pred, y_true):
        """
        Calculates the Mean Squared Error loss manually.

        Parameters:
            y_pred (torch.Tensor): Predicted values, a tensor of shape (N, C) where N - number of samples, C - number of channels.
            y_true (torch.Tensor): True values, a tensor of shape (N, C) where N - number of samples, C - number of channels.

        Returns:
            float: Calculated MSE loss.
        """
        N = y_pred.numel() # N = y_pred.shape[0]
        mse = torch.sum((y_true-y_pred)**2) / N
        return mse

    @staticmethod
    def mse_pytorch(y_pred, y_true):
        """
        Calculates the Mean Squared Error loss using PyTorch's built-in function.

        Parameters:
            y_pred (torch.Tensor): Predicted values, a tensor of shape (N, C) where N - number of samples, C - number of channels.
            y_true (torch.Tensor): True values, a tensor of shape (N, C) where N - number of samples, C - number of channels.

        Returns:
            float: Calculated MSE loss.
        """
        loss = nn.MSELoss()
        return loss(y_pred, y_true)


if __name__ == "__main__":
    def mse_loss_test(N = 10, C = 3):
        """
        Tests the MSE_Loss class.

        Parameters:
            N (int): Number of samples.
            C (int): Number of channels.
        """
        y_true = torch.rand(N, C, dtype=torch.float32); print(f"y_true: {y_true}") # or np.random.rand()
        y_pred = torch.rand(N, C, dtype=torch.float32); print(f"y_pred: {y_pred}")

        mse_manual = MSE_Loss.mse_manual(y_pred, y_true); print(f"mse_manual: {mse_manual}")
        mse_pytorch = MSE_Loss.mse_pytorch(y_pred, y_true); print(f"mse_pytorch: {mse_pytorch}")


    mse_loss_test()
