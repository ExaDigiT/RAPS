"""
Module for utility functions.

This module contains various utility functions used for different tasks such as converting time formats,
generating random numbers, summarizing and expanding ranges, determining job states, and creating binary arrays.

"""

from datetime import timedelta

import hashlib
import math
import numpy as np
import pandas as pd
import random
import sys
import uuid
import json


def sum_values(values):
    return sum(x[1] for x in values) if values else 0


def min_value(values):
    return min(x[1] for x in values) if values else 0


def max_value(values):
    return max(x[1] for x in values) if values else 0


def convert_seconds(seconds):
    """Convert seconds to time format: 3661s -> 01:01"""
    td = timedelta(seconds=seconds)
    h, m, _ = str(td).split(':')
    return f"{h}:{m}"


def truncated_normalvariate(mu, sigma, lower, upper):
    """
    Generate a random number from a truncated normal distribution.

    Parameters
    ----------
    mu : float
        Mean of the distribution.
    sigma : float
        Standard deviation of the distribution.
    lower : float
        Lower bound of the truncated distribution.
    upper : float
        Upper bound of the truncated distribution.

    Returns
    -------
    float
        Random number from the truncated normal distribution.
    """
    while True:
        number = random.normalvariate(mu, sigma)
        if lower < number < upper:
            return number


def linear_to_3d_index(linear_index, shape):
    """
    Convert linear index to 3D index.

    Parameters
    ----------
    linear_index : int
        Linear index.
    shape : tuple
        Shape of the 3D array.

    Returns
    -------
    tuple
        3D index corresponding to the linear index.
    """
    return np.unravel_index(linear_index, shape)


def create_binary_array(N, fraction_ones):
    """
    Create a binary array with a specified number of ones.

    Parameters
    ----------
    N : int
        Length of the binary array.
    fraction_ones : float
        Fraction of ones in the array.

    Returns
    -------
    np.ndarray
        Binary array.
    """
    num_ones = int(N * fraction_ones)
    num_zeros = N - num_ones
    array = np.array([1] * num_ones + [0] * num_zeros)
    np.random.shuffle(array)
    return np.packbits(array)


def get_bit_from_packed(packed_array, index):
    """
    Get the bit value at a specific index from a packed array.

    Parameters
    ----------
    packed_array : np.ndarray
        Packed binary array.
    index : int
        Index of the bit to retrieve.

    Returns
    -------
    int
        Bit value (0 or 1) at the specified index.
    """
    byte_index = index // 8
    bit_position = index % 8
    byte = packed_array[byte_index]
    bitmask = 1 << (7 - bit_position)
    bit_value = (byte & bitmask) >> (7 - bit_position)
    return bit_value


def summarize_ranges(nums):
    """
    Summarize a list of numbers into ranges.

    Parameters
    ----------
    nums : list
        List of numbers.

    Returns
    -------
    list
        List of summarized ranges.
    """
    if not nums:
        return []

    ranges = []
    start = nums[0]
    end = nums[0]

    for num in nums[1:]:
        if num == end + 1:
            end = num
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = num

    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ranges


def expand_ranges(range_str):
    """
    Expand summarized ranges into a list of numbers.

    Parameters
    ----------
    range_str : list
        List of summarized ranges.

    Returns
    -------
    list
        List of expanded numbers.
    """
    nums = []
    for r in range_str:
        if '-' in r:
            start, end = r.split('-')
            nums.extend(range(int(start), int(end) + 1))
        else:
            nums.append(int(r))

    return nums


def determine_state(probs):
    """
    Determine a state based on probability distribution.

    Parameters
    ----------
    probs : dict
        Dictionary containing states as keys and their probabilities as values.

    Returns
    -------
    str
        State selected based on the probability distribution.
    """
    rand_num = random.uniform(0, 1)
    cumulative_prob = 0
    for state, prob in probs.items():
        cumulative_prob += prob
        if rand_num <= cumulative_prob:
            return state


def power_to_utilization(power, pmin, pmax):
    """
    Convert power to utilization based on minimum and maximum power values.

    Parameters
    ----------
    power : float
        Power value.
    pmin : float
        Minimum power value.
    pmax : float
        Maximum power value.

    Returns
    -------
    float
        Utilization value.
    """
    return (power - pmin) / (pmax - pmin)


def create_binary_array_numpy(max_time, trace_quanta, util):
    """
    Create a binary array using NumPy.

    Parameters
    ----------
    max_time : int
        Maximum time.
    trace_quanta : int
        Trace quanta.
    util : array_like
        Utilization values.

    Returns
    -------
    np.ndarray
        Binary array.
    """
    num_quanta = max_time // trace_quanta
    util_filled = np.nan_to_num(util, nan=0)  # Replace NaN with 0
    traces = np.zeros((len(util), num_quanta), dtype=int)
    for i, util in enumerate(util_filled):
        traces[i, :int(util * num_quanta / 100)] = 1
    return traces

def extract_data_csv(fileName, skiprows, header):
    """ Read passed csv file path
        @ In, filename, dataframe, facility telemetry data
        @ In, skiprows, int, number of rows to be skipped
        @ In, header, list, header of output dataframe
        @ Out, df, dataframe, read file returned as a dataframe
    """
    df = pd.read_csv(fileName, skiprows=skiprows, header=header)
    df = df.rename(columns={df.columns[0]: 'time'})
    df = df.dropna()
    return df

def resampledf(df, time_resampled):
    """ Match key and return idx
        @ In, None
        @ Out, CDU_names, list, list of CDU names
    """
    df.set_index('time',inplace =True)
    df = df.reindex(df.index.union(time_resampled)).interpolate('values').loc[time_resampled]
    df = df.reset_index()
    return df

def output_dict(d, title='', output_file=sys.stdout):
    """
    Write dictionary contents to a file.

    Parameters
    ----------
    d : dict
        Dictionary to be written.
    title : str, optional
        Title to be written before the dictionary contents.
    output_file : file object, optional
        Output file object. Default is sys.stdout.
    """
    with output_file as file:
        file.write(title + '\n')
        for key, value in d.items():
            file.write(f"{key}: {value}\n")

def create_casename(prefix=''):
    """
    Generate a unique case name.

    Parameters
    ----------
    prefix : str, optional
        Prefix to be added to the case name.

    Returns
    -------
    str
        Unique case name.
    """
    return prefix + str(uuid.uuid4())[:7]


def next_arrival(lambda_rate):
    if not hasattr(next_arrival, 'next_time'):
        # Initialize the first time it's called
        next_arrival.next_time = 0
    else:
        next_arrival.next_time += \
            -math.log(1.0 - random.random()) / lambda_rate
    return next_arrival.next_time


def convert_to_seconds(time_str):
    # Define the conversion factors
    time_factors = {
        'd': 86400,  # 1 day = 86400 seconds
        'h': 3600,   # 1 hour = 3600 seconds
        'm': 60,     # 1 minute = 60 seconds
        's': 1       # 1 second = 1 second
    }

    # Check if the input string ends with a unit or is purely numeric
    if time_str[-1].isdigit():
        return int(time_str)  # Directly return the number if it's purely numeric

    # Extract the numeric part and the time unit
    num = int(time_str[:-1])
    unit = time_str[-1]

    # Convert to seconds using the conversion factors
    if unit in time_factors:
        return num * time_factors[unit]
    else:
        raise ValueError(f"Unknown time unit: {unit}")


def encrypt(name):
    """Encrypts a given name using SHA-256 and returns the hexadecimal digest."""
    encoded_name = name.encode()
    hash_object = hashlib.sha256(encoded_name)
    return hash_object.hexdigest()


def write_dict_to_file(dictionary, file_path):
    """Function to write dictionary to a text file"""
    with open(file_path, 'w') as file:
        for key, value in dictionary.items():
            if isinstance(value, dict):
                file.write(f"{key}: {{\n")
                for subkey, subvalue in value.items():
                    file.write(f"  {subkey}: {subvalue}\n")
                file.write("}\n")
            else:
                file.write(f"{key}: {value}\n")


def toJSON(obj):
    """Function to dump a json string from object"""
    return json.dumps(
        obj,
        default=lambda o:o.__dict__,
        sort_keys=True,
        indent=4)


def get_utilization(trace, time_quanta_index):
    """Retrieve utilization value for a given trace at a specific time quanta index."""
    if isinstance(trace, (list, np.ndarray)):
        return trace[time_quanta_index]
    elif isinstance(trace, (int, float)):
        return float(trace)
    else:
        raise TypeError(f"Invalid type for utilization: {type(trace)}.")
