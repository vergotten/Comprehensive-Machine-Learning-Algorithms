import torch
import torch.nn as nn
import numpy as np


class MSE_Loss():
    def __init__(self):
        pass

    def mse_manual(y_pred, y_true):
        """
        Parameters:
            y_pred: a numpy array of shape (N,) where N is number of samples.
            y_true: a numpy arrays of shape (N, C) where N - number of samples, C - number of channels.  

        Returns:
            float: calculated mse loss. 
        """
        N = y_pred.numel() # N = y_pred.shape[0]
        mse = torch.sum((y_true-y_pred)**2) / N
        return mse
        
    def mse_pytorch(y_pred, y_true):
        loss = nn.MSELoss()
        return loss(y_pred, y_true)


if __name__ == "__main__":
    def mse_loss_test():
        N = 10; C = 3
        y_true = torch.rand(N, C, dtype=torch.float32); print(f"y_true: {y_true}") # or np.random.rand()
        y_pred = torch.rand(N, C, dtype=torch.float32); print(f"y_pred: {y_pred}")

        mse_manual = MSE_Loss.mse_manual(y_pred, y_true); print(f"mse_manual: {mse_manual}")
        mse_pytorch = MSE_Loss.mse_pytorch(y_pred, y_true); print(f"mse_pytorch: {mse_pytorch}")

    mse_loss_test()