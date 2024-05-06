import argparse
import os


def ArgumentParsers(
    exp_root="./exp",
    timestamp="timestamp",
):
    """
    Creates and configures an argument parser for command-line options, setting default values for experiment 
    directory and timestamp.

    Parameters:
    - exp_root: Default path where the experiments are stored.
    - timestamp: Default timestamp value used for naming files or directories within the experiment.

    Returns:
    - argparse.Namespace: Parsed command-line arguments with default values included if not overridden by user input.
    """
    description = "demo."
    parser = argparse.ArgumentParser(description)

    # Setup experiment configurations and paths
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int, default=-1,
                        help="Local rank of the process on the node, unique per node.")
    parser.add_argument("--exp_name", default="ConvLSTM-model optimize",
                        type=str, help="Name of the experiment for reference and logging.")
    parser.add_argument("--description", default=description,
                        type=str, help="Detailed description of the experiment.")
    parser.add_argument("--exp_root", default=exp_root, type=str,
                        help="Root directory for all experiment related outputs.")
    parser.add_argument("--exp_dir", default=os.path.join(exp_root, timestamp),
                        type=str, help="Specific directory for storing experiment outputs.")
    parser.add_argument("--data_root", default="./data/urbanflood24",
                        type=str, help="Root directory where dataset is stored.")

    # Model configuration flags
    parser.add_argument("--clstm", action="store_true",
                        help="Flag to use ConvLSTM as the base cell for the model.")
    parser.add_argument("--use_checkpoint", action="store_true",
                        help="Enable the use of checkpointing to save model state intermittently.")

    # Training configuration
    parser.add_argument("--seq_num", default=28, type=int,
                        help="Window size per iteration during training.")
    parser.add_argument("--window_size", default=360, type=int,
                        help="Size of the window for each sample during training.")
    parser.add_argument("--train_event", action="store_false",
                        help="Whether to train using full events or only rain processes.")
    parser.add_argument("--all_seq_train", action="store_true",
                        help="Flag to use all sequences in one forward pass.")
    parser.add_argument("--full_window_size", action="store_true",
                        help="Set sequence number equal to window size per backward pass.")
    parser.add_argument("--blocks", default=3, type=int,
                        help="Number of stages in the network model.")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="Batch size for training.")
    parser.add_argument("--random_seed", default=42, type=int,
                        help="Seed for random number generation to ensure reproducibility.")
    parser.add_argument("--num_workers", default=20, type=int,
                        help="Number of worker threads for loading data.")

    # Hardware configuration
    parser.add_argument("--device", default="0", type=str,
                        help="GPU device IDs to use.")

    # Learning rate configuration
    parser.add_argument("--lr", default=1e-2, type=float,
                        help="Initial learning rate.")
    parser.add_argument("--schedule_name", default="WarmUpCosineAnneal", type=str,
                        help="Learning rate scheduler type: ReduceLROnPlateau or WarmUpCosineAnneal.")
    parser.add_argument("--warm_up_iter", default=10, type=float,
                        help="Number of iterations for learning rate warm-up phase.")
    parser.add_argument("--patience", default=10, type=float,
                        help="Patience for learning rate reduction under ReduceLROnPlateau.")
    parser.add_argument("--factor", default=0.9, type=float,
                        help="Factor by which the learning rate will be reduced. New_lr = lr * factor.")

    # Model training and loss configuration
    parser.add_argument("--epochs", default=1000, type=int,
                        help="Total number of epochs to train.")
    parser.add_argument("--loss_name", default="FocalBCE_and_WMSE", type=str,
                        help="Name of the loss function to use for model training.")
    parser.add_argument("--reduction", default="mean", type=str,
                        help="Method to reduce loss: mean or sum.")

    # Directories for saving outputs
    parser.add_argument("--save_loss_dir", default="%s/%s/save_train_loss" %
                        (exp_root, timestamp), type=str, help="Directory to save training loss outputs.")
    parser.add_argument("--save_res_data_dir", default="%s/%s/save_res_data" %
                        (exp_root, timestamp), type=str, help="Directory to save result data from the model.")
    parser.add_argument("--save_model_dir", default="%s/%s/save_model" % (exp_root,
                        timestamp), type=str, help="Directory to save trained model checkpoints.")
    parser.add_argument("--save_dir_flood_maps", default="%s/%s/flood_maps" % (exp_root,
                        timestamp), type=str, help="Directory to save flood maps generated during testing.")
    parser.add_argument("--save_fig_dir", default="%s/%s/figs/" % (exp_root,
                        timestamp), type=str, help="Directory to save figures and plots.")

    # Other configurations
    parser.add_argument("--flood_thres", type=float, default=0.15 * 1000,
                        help="Threshold for defining flooding in the model's predictions.")
    parser.add_argument("--model_params", default="See wandb",
                        help="Model parameters as logged and stored in wandb.")
    parser.add_argument("--upload", action="store_true",
                        help="Whether to upload results to wandb after training.")
    parser.add_argument("--wind_random", action="store_false",
                        help="Whether to select training windows randomly.")
    parser.add_argument("--test", action="store_false",
                        help="Whether to perform testing after training.")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed precision training to accelerate training and reduce memory usage.")
    parser.add_argument("--timestamp", default=timestamp,
                        help="Timestamp to uniquely identify this run.")
    parser.add_argument("--resume", action="store_false",
                        help="Whether to resume training from the last saved checkpoint.")
    parser.add_argument("--cls_thred", type=float, default=0.5,
                        help="Classification threshold for determining positive class predictions.")
    parser.add_argument("--trt", action="store_true",
                        help="Whether to use TensorRT for inference optimization.")
    parser.add_argument("--trt_model_dir", type=str, default="%s/%s/tensorrt/" %
                        (exp_root, timestamp), help="TensorRT model path.")

    args = parser.parse_args()
    # Print all arguments in a well-formatted manner
    args_dict = vars(args)  # Convert the Namespace to a dictionary
    for arg, value in args_dict.items():
        print(f"{arg}: {value}")

    return args
