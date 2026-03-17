"""
Dynamic2DFlood.py — Dataset and preprocessing for the UrbanFlood24 benchmark.

The UrbanFlood24 dataset (Cao et al., J. Hydrology 2025) contains 56 rainfall-
induced urban flood events across four highly urbanised catchments in China
and the UK.  Each event provides:

  • flood.npy    — ground-truth water depth (T, H, W) in metres (MIKE+ output)
  • rainfall.npy — rainfall intensity (T,) or (T, H, W) in mm / time-step
  • absolute_DEM.npy  — terrain surface model (H, W) in metres
  • impervious.npy    — impervious surface fraction (H, W) in [0, 1]
  • manhole.npy       — drainage inlet mask (H, W) in {0, 1}

Rainfall format:
  • Scalar rainfall  (T,)    — spatially uniform; stored as (T, 1, 1, 1) after
                               _prepare_input so DataLoader produces (B, T, 1, 1, 1).
  • Spatial rainfall (T,H,W) — spatially heterogeneous (design storms, radar);
                               stored as (T, 1, H, W) after _prepare_input so
                               DataLoader produces (B, T, 1, H, W).
  Both formats are handled transparently by get_past_rainfall.

Input feature tensor (assembled in ``preprocess_inputs``):
    Shape: (B, 1, C, H, W)   where C = historical_nums * 2 + 3

    Channels (in order):
      [0 : W]         — W = historical_nums past instantaneous rainfall
                        fields (mm/min), normalised by rain_max.
      [W : 2W]        — W past cumulative rainfall fields (mm, running sum
                        since event start), normalised by cumsum_rain_max.
                        Cumulative rainfall captures the total water volume
                        that has already been absorbed / ponded, which drives
                        flood depth better than the instantaneous rate alone.
      [2W]            — DEM (terrain elevation, mm), normalised per-sample
                        by the local spatial min/max.
      [2W+1]          — Impervious surface fraction, normalised to [0,1].
                        Controls how much rainfall becomes surface runoff.
      [2W+2]          — Drainage manhole mask, normalised to [0,1].
                        Marks locations connected to the underground pipe network.

    Default: W = 30 → C = 30*2+3 = 63 channels (1-min resolution, 30-min lookback).
    Lightweight: W = 3  → C = 3*2+3 = 9  channels (10-min resolution, 30-min lookback).
"""

import torch.nn.functional as F
import numpy as np
import os
import torch
import torch.utils.data as data


class Dynamic2DFlood(data.Dataset):
    """PyTorch Dataset for the UrbanFlood24 benchmark.

    Each sample corresponds to one rainfall event × one catchment location.
    The dataset returns raw (un-normalised) tensors; normalisation is applied
    on-the-fly in ``preprocess_inputs`` during training/inference so that the
    per-sample DEM normalisation can use the spatial min/max of each event.

    Parameters
    ----------
    data_root : str
        Root directory of the dataset (parent of ``train/`` and ``test/``).
    split : str
        ``"train"`` or ``"test"``.
    """

    def __init__(self, data_root, split, event_list_file=None, duration=360, location=""):
        """
        Parameters
        ----------
        data_root : str
            Root directory of the dataset (parent of ``train/`` and ``test/``).
        split : str
            ``"train"`` or ``"test"``.
        event_list_file : str or None
            Path to a custom event list ``.txt`` file (one event name per line).
            When ``None`` (default), uses the built-in ``train.txt`` / ``test.txt``
            next to this module.  Pass a custom path for non-UrbanFlood24 datasets
            (e.g. LarNO Futian or UKEA).
        duration : int
            Number of timesteps to pad/truncate each event to.  Must match the
            ``window_size`` / ``duration`` specified in the experiment YAML.
            Default: 360 (UrbanFlood24 full, 1-min resolution, 6-hour events).
            Use 36 for UKEA (10-min, 6h), 72 for Futian (5-min, 6h), 36 for lite.
        location : str
            If non-empty, restrict to this single location name (e.g. ``"location1"``).
            Default ``""`` uses all locations found in the flood directory.
        """
        super(Dynamic2DFlood, self).__init__()
        self.data_root = data_root
        self.duration  = duration
        self.data_dir = os.path.join(
            data_root, "train" if "train" in split else "test")
        self.geo_root = os.path.join(self.data_dir, "geodata")
        self.flood_root = os.path.join(self.data_dir, "flood")

        # Locations sorted: numerically if name contains digits (e.g. location1, location16),
        # otherwise lexicographically (e.g. region1, ukea).
        def _loc_sort_key(x):
            digits = ''.join(filter(str.isdigit, x))
            return (0, int(digits)) if digits else (1, x)

        all_locations = sorted(os.listdir(self.flood_root), key=_loc_sort_key)
        if location:
            if location not in all_locations:
                raise ValueError(f"Location '{location}' not found in {self.flood_root}. "
                                 f"Available: {all_locations}")
            self.locations = [location]
        else:
            self.locations = all_locations
        self.locations_dir = [os.path.join(self.flood_root, loc)
                              for loc in self.locations]
        self.event_names = self._load_event_names(split, event_list_file)

        # Total samples = num_events × num_locations
        self.num_samples = len(self.event_names) * len(self.locations)
        print(f"Loaded Dynamic2DFlood {split} with {self.num_samples} samples "
              f"(locations: {len(self.locations)}, events: {len(self.event_names)})")

    def _load_event_names(self, split, event_list_file=None):
        """Read the event list from a ``.txt`` file (one event name per line).

        Parameters
        ----------
        split : str
            ``"train"`` or ``"test"`` — used to resolve the default filename.
        event_list_file : str or None
            Explicit path to a custom event list file.  When ``None``, defaults
            to ``train.txt`` or ``test.txt`` in the same directory as this module.
        """
        if event_list_file is not None:
            filename = event_list_file
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filename = f'{script_dir}/{split}.txt'
        with open(filename, 'r') as file:
            event_names = [line.strip() for line in file if line.strip()]
        return event_names

    def _load_event(self, index):
        """Return (event_data_dict, event_dir_path) for a flat dataset index."""
        event_id = index // len(self.locations)
        loc_id = index % len(self.locations)
        event_dir = os.path.join(
            self.flood_root, self.locations[loc_id],
            self.event_names[event_id])
        event_data = self._load_event_data(
            event_dir, self.geo_root, self.locations[loc_id])
        return event_data, event_dir

    def _load_event_data(self, event_dir, geo_root, location):
        """Load all ``.npy`` files for one event.

        Reads:
          • ``event_dir/flood.npy``    — water depth (T, H, W) in metres
          • ``event_dir/rainfall.npy`` — rainfall (T,) or (T, H, W) in mm/step
          • ``geo_root/<location>/absolute_DEM.npy``
          • ``geo_root/<location>/impervious.npy``
          • ``geo_root/<location>/manhole.npy``
        """
        event_data = {}
        # Flood and rainfall
        for attr_file in os.listdir(event_dir):
            if not attr_file.endswith(".jpg"):
                attr_name, _ = os.path.splitext(attr_file)
                event_data[attr_name] = np.load(
                    os.path.join(event_dir, attr_file), allow_pickle=True)

        # Static geodata (terrain, imperviousness, drainage)
        geo_dir = os.path.join(geo_root, location)
        for attr_file in os.listdir(geo_dir):
            attr_file_path = os.path.join(geo_dir, attr_file)
            if not os.path.isdir(attr_file_path):
                attr_name, _ = os.path.splitext(attr_file)
                event_data[attr_name] = np.load(attr_file_path, allow_pickle=True)

        return event_data

    def _prepare_input(self, event_data, event_dir, duration=360):
        """Convert raw numpy arrays to model-ready tensors.

        Flood depths in the dataset are stored in metres; we convert to mm
        (× 1000) to match the normalisation constants (flood_max = 5000 mm).
        Rainfall is already in mm / time-step — no unit conversion needed.

        Supports both **scalar** rainfall (T,) and **spatial** rainfall (T, H, W):
          • Scalar  → stored as (T, 1, 1, 1)  ; after DataLoader: (B, T, 1, 1, 1)
          • Spatial → stored as (T, 1, H, W)  ; after DataLoader: (B, T, 1, H, W)
        ``get_past_rainfall`` detects the format and handles both transparently.

        Returns a dict of un-normalised tensors with shapes:
          absolute_DEM       : (1, 1, H, W)   — converted to mm
          max_DEM, min_DEM   : scalar          — for per-sample normalisation
          impervious         : (1, 1, H, W)
          manhole            : (1, 1, H, W)
          rainfall           : (T, 1, 1, 1)   — scalar; OR (T, 1, H, W) — spatial
          cumsum_rainfall    : (T, 1, 1, 1)   — scalar; OR (T, 1, H, W) — spatial
        """
        # Convert DEM from metres to millimetres (matches flood_max units)
        absolute_DEM = torch.from_numpy(event_data["absolute_DEM"]).float() * 1000
        impervious   = torch.from_numpy(event_data["impervious"]).float()
        manhole      = torch.from_numpy(event_data["manhole"]).float()
        rainfall     = torch.from_numpy(event_data["rainfall"]).float()

        # Ensure rainfall sequence length matches duration (zero-pad if shorter)
        if rainfall.shape[0] < duration:
            padding_length = duration - rainfall.shape[0]
            if rainfall.ndim == 1:  # scalar (T,)
                rainfall = F.pad(rainfall, (0, padding_length), 'constant', 0)
            else:  # spatial (T, H, W) — pad along time dimension (dim 0)
                rainfall = F.pad(rainfall, (0, 0, 0, 0, 0, padding_length), 'constant', 0)

        # Running cumulative rainfall from event start (captures total water input)
        cumsum_rainfall = torch.cumsum(rainfall, dim=0)

        # Add batch and channel dimensions for broadcasting
        absolute_DEM    = absolute_DEM.unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        impervious      = impervious.unsqueeze(0).unsqueeze(0)      # (1,1,H,W)
        manhole         = manhole.unsqueeze(0).unsqueeze(0)         # (1,1,H,W)

        # Reshape rainfall to (T, 1, ?, ?) where ? is 1 (scalar) or H,W (spatial)
        if rainfall.ndim == 1:  # scalar (T,) → (T, 1, 1, 1)
            rainfall        = rainfall.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            cumsum_rainfall = cumsum_rainfall.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        else:  # spatial (T, H, W) → (T, 1, H, W)
            rainfall        = rainfall.unsqueeze(1)
            cumsum_rainfall = cumsum_rainfall.unsqueeze(1)

        return {
            "absolute_DEM":   absolute_DEM,
            "max_DEM":        absolute_DEM.max(),
            "min_DEM":        absolute_DEM.min(),
            "impervious":     impervious,
            "manhole":        manhole,
            "rainfall":       rainfall,
            "cumsum_rainfall": cumsum_rainfall,
        }

    def _prepare_target(self, event_data, duration=360):
        """Convert flood ground truth to mm tensors, truncated to ``duration``.

        Ground-truth water depths are from MIKE+ 1D-2D coupled simulations and
        are stored in metres.  We convert to mm to match the normalisation
        constant ``flood_max = 5000 mm``.
        """
        flood = (torch.from_numpy(event_data["flood"]).float()[:duration]
                 * 1000)  # metres → millimetres
        # Squeeze channel dim if stored as (T, 1, H, W) → (T, H, W)
        if flood.ndim == 4 and flood.shape[1] == 1:
            flood = flood.squeeze(1)
        return flood  # shape: (T, H, W)

    def __getitem__(self, index):
        event_data, event_dir = self._load_event(index)
        input_vars  = self._prepare_input(event_data, event_dir, duration=self.duration)
        target_vars = self._prepare_target(event_data, duration=self.duration)
        return [input_vars, target_vars, event_dir]

    def __len__(self):
        return self.num_samples


# ─── Feature preprocessing ────────────────────────────────────────────────────

def preprocess_inputs(t, inputs, device, nums=30,
                      rain_max=6.0, cumsum_rain_max=250.0):
    """Assemble and normalise the multi-channel input tensor for timestep *t*.

    Constructs the C = historical_nums*2 + 3 channel input tensor by:
      1. Extracting the ``nums`` most recent rainfall intensity fields
         (zero-padded at the start if t < nums).
      2. Extracting the corresponding cumulative rainfall fields.
      3. Appending the three static spatial features (DEM, impervious, manhole).

    All channels are normalised to [0, 1] via Min-Max scaling.

    Parameters
    ----------
    t : int
        Current timestep index (0-based) within the sequence window.
    inputs : dict
        Raw (un-normalised) input tensors from ``Dynamic2DFlood.__getitem__``.
    device : torch.device
        Target device.
    nums : int
        Lookback window length in timesteps (= ``historical_nums``).
        Determines the number of rainfall/cumsum channels.
    rain_max : float
        Max instantaneous rainfall intensity (mm/min) for normalisation.
    cumsum_rain_max : float
        Max cumulative rainfall (mm) for normalisation.

    Returns
    -------
    Tensor, shape (B, 1, C, H, W)
        C = nums*2 + 3 (rainfall + cumsum + DEM + impervious + manhole).
    """
    # Static spatial features — DEM normalised per-sample; others fixed-range
    norm_DEM        = MinMaxScaler(inputs["absolute_DEM"],
                                   inputs["max_DEM"][0], inputs["min_DEM"][0])
    norm_impervious = MinMaxScaler(inputs["impervious"], 0.95, 0.05)
    norm_manhole    = MinMaxScaler(inputs["manhole"], 1, 0)

    H, W = inputs["absolute_DEM"].shape[-2:]

    # Dynamic rainfall features: ``nums`` past timesteps, broadcast to (H, W)
    rainfall        = get_past_rainfall(inputs["rainfall"], t, nums, H, W)
    cumsum_rainfall = get_past_rainfall(inputs["cumsum_rainfall"], t, nums, H, W)
    norm_rainfall        = MinMaxScaler(rainfall, rain_max, 0)
    norm_cumsum_rainfall = MinMaxScaler(cumsum_rainfall, cumsum_rain_max, 0)

    # Concatenate in order: [rainfall(nums), cumsum(nums), DEM, impervious, manhole]
    # Total channels C = nums*2 + 3
    processed_inputs = torch.cat(
        [norm_rainfall, norm_cumsum_rainfall,
         norm_DEM, norm_impervious, norm_manhole],
        dim=2,
    ).to(device=device, dtype=torch.float32)

    return processed_inputs  # (B, 1, C, H, W)


def get_past_rainfall(rainfall, t, nums, H, W):
    """Extract the ``nums`` most recent rainfall fields ending at timestep *t*.

    Supports both scalar and spatial rainfall tensors:
      • Scalar  rainfall: input shape (B, T, 1, 1, 1) — broadcast to (H, W)
      • Spatial rainfall: input shape (B, T, 1, H, W) — used as-is

    If t < nums (early timesteps), the earlier channels are zero-padded to
    represent the pre-event dry period.

    Parameters
    ----------
    rainfall : Tensor
        Shape (B, T, 1, h, w) where h=w=1 for scalar or h=H, w=W for spatial.
    t : int
        Current timestep (0-based).
    nums : int
        Number of historical steps to retrieve.
    H, W : int
        Target spatial dimensions for the output tensor.

    Returns
    -------
    Tensor, shape (B, 1, nums, H, W)
    """
    B, S, C, h, w = rainfall.shape
    start_idx = max(0, t - nums + 1)
    end_idx   = min(t + 1, S)

    # Zero-padded output: early timesteps get zeros (pre-event / no data)
    extracted_rainfall = torch.zeros((B, 1, nums, H, W), device=rainfall.device)
    actual_num_steps   = end_idx - start_idx

    # Extract: rainfall[:, start:end, 0, ...] → (B, n_steps, h, w)
    extracted_data = rainfall[:, start_idx:end_idx, 0, ...]   # (B, n, h, w)
    extracted_data = extracted_data.unsqueeze(1)               # (B, 1, n, h, w)

    if h == 1 and w == 1:
        # Scalar rainfall — broadcast spatial singleton dimensions to full grid
        extracted_data = extracted_data.expand(-1, 1, -1, H, W)

    extracted_rainfall[:, :, nums - actual_num_steps:, ...] = extracted_data

    return extracted_rainfall  # (B, 1, nums, H, W)


def MinMaxScaler(data, max, min):
    """Normalise *data* to [0, 1] using fixed min-max bounds.

    Used with calibrated constants (e.g. rain_max=6.0 mm/min,
    flood_max=5000 mm) so that the normalisation is consistent across all
    events and generalises to unseen event magnitudes.
    """
    return (data - min) / (max - min)


def r_MinMaxScaler(data, max, min):
    """Invert MinMaxScaler: recover physical values from normalised predictions.

    Used in post-processing to convert model outputs (in [0, 1]) back to
    flood depths in mm for evaluation and visualisation.
    """
    return data * (max - min) + min
