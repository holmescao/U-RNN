import numpy as np
import torch


class SaveBestModel:
    """
    Implements early stopping and model saving based on improvement in validation loss.

    Attributes:
        patience: Number of epochs to wait for improvement before stopping.
        verbose: If True, prints messages regarding validation loss improvement.
    """

    def __init__(self, patience=7, verbose=False):
        """
        Initializes the SaveBestModel with specified patience and verbosity.

        Parameters:
        - patience: Number of epochs to wait after the last improvement in validation loss.
        - verbose: If set to True, enables printing of log messages.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, optimizer, epoch, save_path):
        """
        Evaluates the model's performance and decides whether to save a checkpoint or stop early.

        Parameters:
        - val_loss: The validation loss for the current epoch.
        - model: The model being trained.
        - optimizer: The optimizer used for training.
        - epoch: The current training epoch.
        - save_path: Directory where the model checkpoints are saved.
        """
        model_dict = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model_dict, epoch, save_path)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model_dict, epoch, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, save_path):
        """
        Saves the model checkpoint if validation loss has decreased.

        Parameters:
        - val_loss: The new validation loss to compare against the minimum.
        - model_dict: A dictionary containing the model's and optimizer's state.
        - epoch: The current epoch number.
        - save_path: The path to save the checkpoint to.
        """
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.9f} --> {val_loss:.9f}).  Saving model ...')
        torch.save(model, save_path + "/" +
                   "checkpoint_{}_{:.9f}.pth.tar".format(epoch, val_loss))
        self.val_loss_min = val_loss
