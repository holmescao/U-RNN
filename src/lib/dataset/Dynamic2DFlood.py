import torch.nn.functional as F
import numpy as np
import os
import torch
import torch.utils.data as data


class Dynamic2DFlood(data.Dataset):
    """
    A dataset class for loading 2D flood event data for either training or testing.
    """

    def __init__(self, data_root, split):
        """
        Initializes the dataset by setting up directories and loading event names.

        Parameters:
        - data_root: The root directory of the data.
        - split: The type of data split ('train' or 'test').
        """
        super(Dynamic2DFlood, self).__init__()
        self.data_root = data_root
        self.data_dir = os.path.join(
            data_root, "train" if "train" in split else "test")
        self.geo_root = os.path.join(self.data_dir, "geodata")
        self.flood_root = os.path.join(self.data_dir, "flood")

        self.locations = sorted(os.listdir(self.flood_root), key=lambda x: int(
            ''.join(filter(str.isdigit, x))))
        self.locations_dir = [os.path.join(
            self.flood_root, loc) for loc in self.locations]
        self.event_names = self._load_event_names(split)

        self.num_samples = len(self.event_names) * len(self.locations)
        print(
            f"Loaded Dynamic2DFlood {split} with {self.num_samples} samples (locations: {len(self.locations)}, events: {len(self.event_names)})")

    def _load_event_names(self, split):
        """
        Loads event names from a text file based on the split.

        Parameters:
        - split: The data split ('train' or 'test') to determine which events to load.

        Returns:
        - List of event names.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = f'{script_dir}/{split}.txt'
        with open(filename, 'r') as file:
            event_names = [line.strip() for line in file]
        return event_names

    def _load_event(self, index):
        """
        Loads event data given an index.

        Parameters:
        - index: Index of the event to load.

        Returns:
        - Tuple containing the event data and the directory path of the event.
        """
        event_id = index // len(self.locations)
        loc_id = index % len(self.locations)
        event_dir = os.path.join(
            self.flood_root, self.locations[loc_id], self.event_names[event_id])
        event_data = self._load_event_data(
            event_dir, self.geo_root, self.locations[loc_id])

        return event_data, event_dir

    def _load_event_data(self, event_dir, geo_root, location):
        """
        Helper method to load data for a given event.

        Parameters:
        - event_dir: Directory for the specific event's data.
        - geo_root: Root directory for geographic data.
        - location: Specific location identifier.

        Returns:
        - Dictionary containing loaded data attributes.
        """
        event_data = {}
        # Load flood and rainfall data
        for attr_file in os.listdir(event_dir):
            if not attr_file.endswith(".jpg"):
                attr_name, _ = os.path.splitext(attr_file)
                attr_file_path = os.path.join(event_dir, attr_file)
                event_data[attr_name] = np.load(
                    attr_file_path, allow_pickle=True)

        # Load geographic data such as DEM, impervious surfaces, and manholes
        geo_dir = os.path.join(geo_root, location)
        for attr_file in os.listdir(geo_dir):
            if not os.path.isdir(attr_file_path := os.path.join(geo_dir, attr_file)):
                attr_name, _ = os.path.splitext(attr_file)
                event_data[attr_name] = np.load(
                    attr_file_path, allow_pickle=True)

        return event_data

    def _prepare_input(self, event_data, event_dir, duration=360):
        """
        Prepares the input data tensors for a specific event, adjusting dimensions and units.

        Parameters:
        - event_data: Dictionary containing data attributes for the event.
        - event_dir: Directory path of the event.
        - duration: Expected duration in time frames for padding.

        Returns:
        - Dictionary of input variables with adjusted dimensions and units.
        """
        # Extract and convert data into tensors
        # Convert from meters to millimeters
        absolute_DEM = torch.from_numpy(
            event_data["absolute_DEM"]).float() * 1000
        impervious = torch.from_numpy(event_data["impervious"]).float()
        manhole = torch.from_numpy(event_data["manhole"]).float()
        rainfall = torch.from_numpy(event_data["rainfall"]).float()

        # Padding rainfall data to ensure consistency in sequence length
        if len(rainfall) < duration:
            padding_length = duration - len(rainfall)
            rainfall = F.pad(rainfall, (0, padding_length), 'constant', 0)

        # Cumulative rainfall calculation
        cumsum_rainfall = torch.cumsum(rainfall, dim=0)

        # Reshape data for model input
        absolute_DEM = absolute_DEM.unsqueeze(0).unsqueeze(
            0)  # Add batch and channel dimensions
        impervious = impervious.unsqueeze(0).unsqueeze(0)
        manhole = manhole.unsqueeze(0).unsqueeze(0)
        rainfall = rainfall.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        cumsum_rainfall = cumsum_rainfall.unsqueeze(
            1).unsqueeze(1).unsqueeze(1)

        return {
            "absolute_DEM": absolute_DEM,
            "max_DEM": absolute_DEM.max(),
            "min_DEM": absolute_DEM.min(),
            "impervious": impervious,
            "manhole": manhole,
            "rainfall": rainfall,
            "cumsum_rainfall": cumsum_rainfall
        }

    def _prepare_target(self, event_data, duration=360):
        """
        Prepares the target flood data for model training/testing.

        Parameters:
        - event_data: Dictionary containing flood data attributes.
        - duration: Expected duration in time frames for truncation.

        Returns:
        - Torch tensor of flood data truncated to the specified duration and converted to millimeters.
        """
        # Extract and convert flood data into a tensor, truncate if longer than duration
        flood = torch.from_numpy(event_data["flood"]).float(
        )[:duration] * 1000  # Convert from meters to millimeters

        return flood

    def __getitem__(self, index):
        """
        Get dataset item, formatted for model input.

        Parameters:
        - index: Index of the event data to load.

        Returns:
        - Tuple containing input variables, target variables, and the event directory path.
        """
        # Load event data based on index
        event_data, event_dir = self._load_event(index)
        # Prepare input and target variables
        input_vars = self._prepare_input(event_data, event_dir)
        target_vars = self._prepare_target(event_data)

        return [input_vars, target_vars, event_dir]

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
        - Integer count of total samples.
        """
        return self.num_samples


def preprocess_inputs(t, inputs, device, nums=30):
    """
    Normalize inputs and prepare them for model processing.

    Parameters:
    - t: Current timestep for data extraction.
    - inputs: Dictionary of input tensors.
    - device: Device to which tensors should be moved.
    - nums: Number of historical timesteps to consider.

    Returns:
    - Tensor of concatenated normalized inputs.
    """
    # Extract and normalize input data tensors
    absolute_DEM = MinMaxScaler(
        inputs["absolute_DEM"], inputs["max_DEM"][0], inputs["min_DEM"][0])
    impervious = MinMaxScaler(inputs["impervious"], 0.95, 0.05)
    manhole = MinMaxScaler(inputs["manhole"], 1, 0)

    # Retrieve and normalize past rainfall data
    H, W = inputs["absolute_DEM"].shape[-2:]
    rainfall = get_past_rainfall(inputs["rainfall"], t, nums, H, W)
    cumsum_rainfall = get_past_rainfall(
        inputs["cumsum_rainfall"], t, nums, H, W)
    # Max intensity for a 500-year event
    norm_rainfall = MinMaxScaler(rainfall, 6, 0)
    # Max total rainfall for a 500-year event
    norm_cumsum_rainfall = MinMaxScaler(cumsum_rainfall, 250, 0)

    # Concatenate all processed inputs along the channel dimension and move to specified device
    processed_inputs = torch.cat(
        [norm_rainfall, norm_cumsum_rainfall, absolute_DEM, impervious, manhole],
        dim=2,
    ).to(device=device, dtype=torch.float32)

    return processed_inputs


def get_past_rainfall(rainfall, t, nums, H, W):
    """
    Extracts a slice of past rainfall data for given parameters.

    Parameters:
    - rainfall: Tensor containing rainfall data.
    - t: Current time index.
    - nums: Number of timesteps to retrieve.
    - H: Height of the data.
    - W: Width of the data.

    Returns:
    - Tensor of extracted rainfall data.
    """
    B, S, C, _, _ = rainfall.shape
    start_idx = max(0, t - nums + 1)
    end_idx = min(t + 1, S)

    extracted_rainfall = torch.zeros(
        (B, 1, nums, H, W), device=rainfall.device)
    actual_num_steps = end_idx - start_idx
    extracted_data = rainfall[:, start_idx:end_idx,
                              0, ...].unsqueeze(1).expand(-1, 1, -1, H, W)
    extracted_rainfall[:, :, nums - actual_num_steps:, ...] = extracted_data

    return extracted_rainfall


def MinMaxScaler(data, max, min):
    """
    Normalizes data using the Min-Max scaling technique.

    Parameters:
    - data: The data tensor to normalize.
    - max: The maximum value for scaling.
    - min: The minimum value for scaling.

    Returns:
    - Normalized data tensor.
    """
    return (data - min) / (max - min)


def r_MinMaxScaler(data, max, min):
    """
    Reverses Min-Max scaling to original values based on the provided maximum and minimum values used in scaling.

    Parameters:
    - data: Normalized data tensor.
    - max: The maximum value used in the original normalization.
    - min: The minimum value used in the original normalization.

    Returns:
    - Original data tensor.
    """
    return data * (max - min) + min
