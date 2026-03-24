import argparse
import os
import yaml


# Default config file paths (relative to this file's directory = code root)
_CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_TRAIN_CFG   = os.path.join(_CODE_ROOT, "configs", "defaults", "training.yaml")
_DEFAULT_DATA_CFG    = os.path.join(_CODE_ROOT, "configs", "defaults", "data.yaml")
_DEFAULT_NETWORK_CFG = os.path.join(_CODE_ROOT, "configs", "network.yaml")

# Experiment config directory (for per-scenario YAMLs)
_DEFAULT_EXP_CFG_DIR = os.path.join(_CODE_ROOT, "configs")


def _load_yaml(path):
    """Load a YAML file and return its contents as a flat dict. Returns {} on missing file."""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _str2bool(v):
    """Allow boolean CLI args to be passed as --flag true/false."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "1"):
        return True
    if v.lower() in ("no", "false", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v!r}")


def ArgumentParsers(
    exp_root="../exp",
    timestamp="timestamp",
):
    """
    Build the argument parser for U-RNN.

    Defaults are loaded from three YAML files:
        configs/training.yaml  — optimizer, SWP paradigm, loss, epochs
        configs/data.yaml      — dataset path, resolution, normalization
        configs/network.yaml   — model architecture (loaded separately in main.py)

    Any argument can be overridden on the command line.  To use a custom YAML,
    pass --train_config / --data_config / --net_config <path>.

    Returns
    -------
    argparse.Namespace
    """
    # ── Step 1: minimal pre-parse to discover which YAML files to load ────────
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--train_config",   default=_DEFAULT_TRAIN_CFG)
    pre.add_argument("--data_config",    default=_DEFAULT_DATA_CFG)
    pre.add_argument("--net_config",     default=None)
    pre.add_argument("--exp_config",     default=None,
                     help="Self-contained experiment YAML (configs/full.yaml or configs/lite.yaml). "
                          "When provided, training.yaml and data.yaml are NOT loaded — "
                          "the exp_config is the sole source of defaults.")
    pre_args, _ = pre.parse_known_args()

    # ── Step 2: load YAML defaults ─────────────────────────────────────────────
    # Two modes:
    #   exp_config provided → standalone mode: only exp_config is loaded.
    #     Use configs/full.yaml or configs/lite.yaml (each is self-contained).
    #   exp_config absent   → developer mode: training.yaml + data.yaml merged.
    yaml_defaults = {}
    if pre_args.exp_config:
        yaml_defaults.update(_load_yaml(pre_args.exp_config))
    else:
        yaml_defaults.update(_load_yaml(pre_args.train_config))
        yaml_defaults.update(_load_yaml(pre_args.data_config))

    # ── Step 3: full parser ────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="U-RNN: High-resolution spatiotemporal nowcasting of urban flooding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file paths
    parser.add_argument("--train_config", default=pre_args.train_config, type=str,
                        help="Path to training YAML config. Defaults to configs/training.yaml.")
    parser.add_argument("--data_config",  default=pre_args.data_config,  type=str,
                        help="Path to data YAML config. Defaults to configs/data.yaml.")
    parser.add_argument("--net_config",   default=pre_args.net_config,   type=str,
                        help="Path to network architecture YAML. Defaults to configs/network.yaml.")
    parser.add_argument("--exp_config",   default=pre_args.exp_config,   type=str,
                        help="Per-experiment YAML (highest priority). "
                             "See configs/experiments/ for ready-made scenarios: "
                             "inference.yaml, lite_training.yaml, full_training.yaml.")

    # Distributed training (set automatically by torch.distributed.launch / torchrun)
    parser.add_argument("--local-rank", type=int, default=-1,
                        help="Local rank for DDP. Set automatically by torchrun; -1 means single-GPU.")

    # Experiment metadata
    parser.add_argument("--exp_name", default="U-RNN", type=str,
                        help="Experiment name used for wandb logging.")
    parser.add_argument("--exp_root", default=exp_root, type=str,
                        help="Root directory for all experiment outputs.")
    parser.add_argument("--exp_dir",  default=os.path.join(exp_root, timestamp), type=str,
                        help="Specific output directory for this run.")
    parser.add_argument("--timestamp", default=timestamp,
                        help="Unique run identifier. Auto-generated at training start.")

    # ── Dataset ─────────────────────────────────────────────────────────────────
    parser.add_argument("--data_root", type=str,
                        help="Root directory of the dataset.")
    parser.add_argument("--input_height", type=int,
                        help="Spatial height of the input grid (pixels). Must match dataset.")
    parser.add_argument("--input_width",  type=int,
                        help="Spatial width of the input grid (pixels). Must match dataset.")
    parser.add_argument("--historical_nums", type=int,
                        help="Number of past rainfall timesteps used as input features. "
                             "Input channels = historical_nums * 2 + 3.")
    parser.add_argument("--duration", type=int,
                        help="Time steps per sample used for padding/truncation in data loading.")
    parser.add_argument("--location", type=str, default="",
                        help="If set, restrict training/testing to this single location name "
                             "(e.g. 'location1'). Empty string (default) uses all locations.")

    # Normalization constants
    parser.add_argument("--flood_max",       type=float, help="Max flood depth (mm) for normalization.")
    parser.add_argument("--rain_max",        type=float, help="Max rainfall intensity (mm/min).")
    parser.add_argument("--cumsum_rain_max", type=float, help="Max cumulative rainfall (mm).")
    parser.add_argument("--flood_thres",     type=float, help="Flood/dry classification threshold (mm).")

    # Visualization
    parser.add_argument("--viz_time_points", nargs="+", type=int,
                        help="Timestep indices for spatial snapshots in test output.")

    # ── Training ─────────────────────────────────────────────────────────────────
    parser.add_argument("--epochs",      type=int,   help="Total training epochs.")
    parser.add_argument("--batch_size",  type=int,   help="Training batch size per GPU.")
    parser.add_argument("--random_seed", type=int,   help="Global random seed.")
    parser.add_argument("--num_workers", type=int,   help="Data loader worker threads.")

    # SWP paradigm
    parser.add_argument("--seq_num",          type=int,         help="Time steps per backward pass.")
    parser.add_argument("--window_size",      type=int,         help="Sequence window length per sample.")
    parser.add_argument("--train_event",      type=_str2bool,   help="Use full event sequence for training.")
    parser.add_argument("--all_seq_train",    type=_str2bool,   help="Use entire event length as window.")
    parser.add_argument("--full_window_size", type=_str2bool,   help="Set seq_num = window_size.")
    parser.add_argument("--wind_random",      type=_str2bool,   help="Shuffle SWP windows within each sample.")

    # Pre-warming
    parser.add_argument("--prewarming", type=_str2bool,
                        help="Enable SWP pre-warming (paper-accurate but slower). "
                             "When False (default), each window starts from the previous "
                             "window's final hidden states, which is ~equally accurate "
                             "but significantly faster.")

    # Optimizer & LR scheduler
    parser.add_argument("--lr",            type=float, help="Initial / peak learning rate.")
    parser.add_argument("--lr_min",        type=float, help="Minimum LR (WarmUpCosineAnneal).")
    parser.add_argument("--schedule_name", type=str,   help="LR scheduler: WarmUpCosineAnneal | ReduceLROnPlateau.")
    parser.add_argument("--warm_up_iter",  type=float, help="Warm-up epochs.")
    parser.add_argument("--patience",      type=float, help="LR patience (ReduceLROnPlateau).")
    parser.add_argument("--factor",        type=float, help="LR reduction factor (ReduceLROnPlateau).")
    parser.add_argument("--grad_clip",     type=float, default=1.0,
                        help="Max L2 norm for gradient clipping (torch.nn.utils.clip_grad_norm_). "
                             "Set to 0 to disable. Default: 1.0.")

    # Loss & thresholds
    parser.add_argument("--loss_name",  type=str,   help="Loss function name.")
    parser.add_argument("--reduction",  type=str,   help="Loss reduction: mean | sum.")
    parser.add_argument("--cls_thred",  type=float, help="Classification probability threshold.")

    # ── Model flags ────────────────────────────────────────────────────────────
    parser.add_argument("--clstm",          action="store_true", help="Use ConvLSTM cells instead of ConvGRU.")
    parser.add_argument("--use_checkpoint", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--blocks",         type=int, default=3,  help="Number of encoder-decoder stages.")

    # ── Hardware & runtime ─────────────────────────────────────────────────────
    parser.add_argument("--device",  default="0", type=str, help="GPU device ID(s) to use.")
    parser.add_argument("--amp",     action="store_true",   help="Enable mixed-precision training (FP16).")
    parser.add_argument("--resume",  type=_str2bool, help="Resume from the last checkpoint.")
    parser.add_argument("--pretrain_path", type=str, default="",
                        help="Path to a pre-trained checkpoint (.pth.tar) for fine-tuning. "
                             "Loads weights only (no optimizer state); training starts from epoch 0. "
                             "Layers with mismatched shapes are skipped automatically.")
    parser.add_argument("--test",    type=_str2bool, help="Run test/inference after training.")
    parser.add_argument("--upload",  action="store_true",   help="Upload results to wandb.")

    # TensorRT inference
    parser.add_argument("--trt",          action="store_true", help="Use TensorRT for inference.")
    parser.add_argument("--trt_model_dir", type=str,
                        default="%s/%s/tensorrt/" % (exp_root, timestamp),
                        help="Directory for TensorRT engine files.")

    # Dataset event lists (optional; default: built-in train.txt / test.txt)
    parser.add_argument("--train_list_file", type=str, default=None,
                        help="Path to custom training event list .txt file. "
                             "One event name per line. Default: built-in train.txt.")
    parser.add_argument("--test_list_file",  type=str, default=None,
                        help="Path to custom testing event list .txt file. "
                             "One event name per line. Default: built-in test.txt.")

    # Output directories (auto-derived from exp_root + timestamp)
    parser.add_argument("--save_loss_dir",       default="%s/%s/save_train_loss" % (exp_root, timestamp), type=str)
    parser.add_argument("--save_res_data_dir",   default="%s/%s/save_res_data"   % (exp_root, timestamp), type=str)
    parser.add_argument("--save_model_dir",      default="%s/%s/save_model"      % (exp_root, timestamp), type=str)
    parser.add_argument("--save_dir_flood_maps", default="%s/%s/flood_maps"      % (exp_root, timestamp), type=str)
    parser.add_argument("--save_fig_dir",        default="%s/%s/figs/"           % (exp_root, timestamp), type=str)
    parser.add_argument("--save_metric_dir",     default="%s/%s/metrics/"        % (exp_root, timestamp), type=str)

    # Misc (kept for backwards-compat)
    parser.add_argument("--model_params", default="See wandb",
                        help="Placeholder shown in wandb run config.")
    parser.add_argument("--description",  default="U-RNN training run.", type=str)

    # ── Apply YAML defaults (CLI args override these) ──────────────────────────
    parser.set_defaults(**yaml_defaults)

    args = parser.parse_args()

    # Re-derive all timestamp-based directory paths from the resolved args.timestamp.
    # This ensures that passing --timestamp <X> on the CLI correctly propagates to
    # save_model_dir, save_fig_dir, etc. (which are constructed as format strings
    # at parser-build time and therefore do not automatically reflect CLI overrides).
    args.exp_dir          = os.path.join(args.exp_root, args.timestamp)
    args.save_model_dir   = os.path.join(args.exp_dir, "save_model")
    args.save_res_data_dir = os.path.join(args.exp_dir, "save_res_data")
    args.save_loss_dir    = os.path.join(args.exp_dir, "save_train_loss")
    args.save_dir_flood_maps = os.path.join(args.exp_dir, "flood_maps")
    args.save_fig_dir     = os.path.join(args.exp_dir, "figs/")
    args.save_metric_dir  = os.path.join(args.exp_dir, "metrics/")
    args.trt_model_dir    = os.path.join(args.exp_dir, "tensorrt/")

    # Print all resolved arguments
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    return args
