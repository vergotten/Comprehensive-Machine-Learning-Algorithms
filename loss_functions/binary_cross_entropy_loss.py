import torch
import torch.nn as nn

class BinaryCrossEntropy_Loss:
    """
    This class provides two methods to calculate Binary Cross-Entropy Loss.
    """
    def __init__(self):
        pass

    @staticmethod
    def binary_cross_entropy_manual(y_pred, y_true):
        """
        Calculate Binary Cross-Entropy Loss manually.

        Parameters:
        y_pred (torch.Tensor): The predicted probabilities.
        y_true (torch.Tensor): The actual labels.

        Returns:
        torch.Tensor: The calculated Binary Cross-Entropy loss.
        """
        def sigmoid(x):
            return 1 / (1 + torch.exp(-x))

        N = y_pred.shape[0]
        y_pred = sigmoid(y_pred)
        loss = -1 / N * torch.sum(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        return loss

    @staticmethod
    def binary_cross_entropy_pytorch(y_pred, y_true):
        """
        Calculate Binary Cross-Entropy Loss using PyTorch's built-in function.

        Parameters:
        y_pred (torch.Tensor): The predicted probabilities.
        y_true (torch.Tensor): The actual labels.

        Returns:
        torch.Tensor: The calculated Binary Cross-Entropy loss.
        """
        y_pred = torch.sigmoid(y_pred)
        loss_fn = nn.BCELoss()
        loss = loss_fn(y_pred, y_true)
        return loss

if __name__ == "__main__":
    def binary_cross_entropy_test(N=10, C=2):
        """
        Test the BinaryCrossEntropy_Loss class with some random data.

        Parameters:
        N (int): The number of samples.
        C (int): The number of classes.
        """
        y_true = torch.randint(C, size=(N,))  # generate vector (N,) with 0's and 1's
        y_true = y_true.float()
        print(f"y_true: {y_true}")
        y_pred = torch.randn(N, 1).squeeze()  # generate (N, C) matrix with normal distribution values [0:1]
        print(f"y_pred: {y_pred}")

        binary_cross_entropy_manual = BinaryCrossEntropy_Loss.binary_cross_entropy_manual(y_pred, y_true)
        print(f"binary_cross_entropy_manual: {binary_cross_entropy_manual}")

        binary_cross_entropy_pytorch = BinaryCrossEntropy_Loss.binary_cross_entropy_pytorch(y_pred, y_true)
        print(f"binary_cross_entropy_pytorch: {binary_cross_entropy_pytorch}")

    binary_cross_entropy_test()
