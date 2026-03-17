#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import os

import tensorrt as trt
from .common_runtime import *

try:
    # Sometimes python does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


def GiB(val):
    """
    Converts the given value in gigabytes to bytes.

    Parameters:
    - val (int or float): Value in gigabytes to convert.

    Returns:
    - int: The value in bytes.
    """
    return val * (1 << 30)


def add_help(description):
    """
    Creates an argument parser with a given description and parses known arguments. This function is used to quickly
    generate help menus for scripts, utilizing the argparse library to display defaults alongside each option.

    Parameters:
    - description: Description text for the help menu of the command-line interface.

    Returns:
    - Namespace: A namespace populated with values from known command line arguments.
    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args, _ = parser.parse_known_args()
    return args


def find_sample_data(description="Runs a TensorRT Python sample", subfolder="", find_files=[], err_msg=""):
    """
    Parses sample arguments.

    Args:
        description: Description of the sample.
        subfolder: The subfolder containing data relevant to this sample
        find_files: A list of filenames to find. Each filename will be replaced with an absolute path.

    Returns:
        str: Path of data directory.
    """

    # Standard command-line arguments for all samples.
    kDEFAULT_DATA_ROOT = os.path.join(os.sep, "usr", "src", "tensorrt", "data")
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-d",
        "--datadir",
        help="Location of the TensorRT sample data directory, and any additional data directories.",
        action="append",
        default=[kDEFAULT_DATA_ROOT],
    )
    args, _ = parser.parse_known_args()

    def get_data_path(data_dir):
        # If the subfolder exists, append it to the path, otherwise use the provided path as-is.
        data_path = os.path.join(data_dir, subfolder)
        if not os.path.exists(data_path):
            if data_dir != kDEFAULT_DATA_ROOT:
                print("WARNING: " + data_path +
                      " does not exist. Trying " + data_dir + " instead.")
            data_path = data_dir
        # Make sure data directory exists.
        if not (os.path.exists(data_path)) and data_dir != kDEFAULT_DATA_ROOT:
            print(
                "WARNING: {:} does not exist. Please provide the correct data path with the -d option.".format(
                    data_path
                )
            )
        return data_path

    data_paths = [get_data_path(data_dir) for data_dir in args.datadir]
    return data_paths, locate_files(data_paths, find_files, err_msg)


def locate_files(data_paths, filenames, err_msg=""):
    """
    Locates the specified files within a list of data directories, returning their paths.
    Only the first occurrence of each file is returned based on the order of data directories provided.

    Parameters:
    - data_paths: List of directories to search for the files.
    - filenames: List of filenames to locate within the provided directories.
    - err_msg: Optional error message to include in the exception if a file is not found.

    Returns:
    - List[str]: A list containing the absolute paths of the files.

    Raises:
    - FileNotFoundError: If any file cannot be located in the provided directories.
    """
    found_files = [None] * len(filenames)
    for data_path in data_paths:
        # Find all requested files.
        for index, (found, filename) in enumerate(zip(found_files, filenames)):
            if not found:
                file_path = os.path.abspath(os.path.join(data_path, filename))
                if os.path.exists(file_path):
                    found_files[index] = file_path

    # Check that all files were found
    for f, filename in zip(found_files, filenames):
        if not f or not os.path.exists(f):
            raise FileNotFoundError(
                "Could not find {:}. Searched in data paths: {:}\n{:}".format(
                    filename, data_paths, err_msg)
            )
    return found_files


def setup_timing_cache(config: trt.IBuilderConfig, timing_cache_path: os.PathLike):
    """
    Sets up a timing cache from a specified file for a TensorRT builder configuration.
    If the cache file exists, it is loaded; otherwise, a new cache is created.

    Parameters:
    - config: The TensorRT builder configuration object.
    - timing_cache_path: Path to the file containing the serialized timing cache.
    """
    buffer = b""
    if os.path.exists(timing_cache_path):
        with open(timing_cache_path, mode="rb") as timing_cache_file:
            buffer = timing_cache_file.read()
    timing_cache: trt.ITimingCache = config.create_timing_cache(buffer)
    config.set_timing_cache(timing_cache, True)


def save_timing_cache(config: trt.IBuilderConfig, timing_cache_path: os.PathLike):
    """
    Saves the timing cache from a TensorRT builder configuration to a file.

    Parameters:
    - config: The TensorRT builder configuration object.
    - timing_cache_path: Path to the file where the timing cache will be saved.
    """
    timing_cache: trt.ITimingCache = config.get_timing_cache()
    with open(timing_cache_path, 'wb') as timing_cache_file:
        timing_cache_file.write(memoryview(timing_cache.serialize()))
