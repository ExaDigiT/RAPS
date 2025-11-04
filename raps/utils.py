"""
Module for utility functions.

This module contains various utility functions used for different tasks such as converting time formats,
generating random numbers, summarizing and expanding ranges, determining job states, and creating binary arrays.

"""

from datetime import datetime, timedelta, timezone, date
from collections.abc import Iterable
from enum import Enum
import os
import hashlib
import math
import re
import numpy as np
import pandas as pd
import random
import sys
import uuid
import json
import argparse
from pathlib import Path
from typing import Annotated as A, TypeVar, TypeAlias, Protocol
from pydantic import (
    BaseModel, TypeAdapter, AfterValidator, BeforeValidator, ConfigDict, AwareDatetime, ValidationError,
    ValidationInfo,
)
from pydantic_settings import BaseSettings, SettingsConfigDict, CliApp, CliSettingsSource, SettingsError
import yaml
from yaml import YAMLError
from raps.job import Job


def deep_merge(a: dict, b: dict):
    a = {**a}
    for key in b.keys():
        if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
            a[key] = deep_merge(a[key], b[key])
        else:
            a[key] = b[key]
    return a


def deep_subtract_dicts(a: dict, b: dict):
    """
    Remove all fields from a that are already in b, such that
    deep_merge(deep_subtract_dicts(a, b), b) == a
    a should contain a superset of b's keys.
    """
    a = {**a}
    for key in b.keys():
        if key in a:
            if a[key] == b[key]:
                a.pop(key)
            elif isinstance(a[key], dict) and isinstance(b[key], dict):
                a[key] = deep_subtract_dicts(a[key], b[key])
            # otherwise keep key in a as is
    return a


def to_dict(arg):
    """
    Normalizes arg to a dictionary if necessary. Used to convert between legacy argparse.Namespace
    objects and dictionaries.
    """
    if isinstance(arg, dict):
        return arg
    elif isinstance(arg, argparse.Namespace):
        return vars(arg)
    else:
        raise ValueError(f"Cannot convert {arg} to dict")


DateType = TypeVar("DateType", date, datetime)


def date_range(start: DateType, end: DateType, step=timedelta(days=1)) -> Iterable[DateType]:
    window_start = start
    while window_start < end:
        yield window_start
        window_start += step


def sum_values(values):
    return sum(x[1] for x in values) if values else 0


def min_value(values):
    return min(x[1] for x in values) if values else 0


def max_value(values):
    return max(x[1] for x in values) if values else 0


def convert_seconds_to_hhmmss(seconds):
    """Convert seconds to time format: 3661s -> 01:01"""
    td = timedelta(seconds=seconds)
    h, m, s = str(td).split(':')
    return f"{h}:{m}:{s}"


def convert_seconds_to_hhmm(seconds):
    """Convert seconds to time format: 3661s -> 01:01"""
    # if it's a NumPy scalar, extract the Python value
    if hasattr(seconds, "item"):
        seconds = seconds.item()
    td = timedelta(seconds=seconds)
    h, m, _ = str(td).split(':')
    return f"{h}:{m}"


def truncated_normalvariate_int(mu, sigma, lower, upper):
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
    int
        Random number from the truncated normal distribution.
    """
    CUTOFF = 100000000
    i = 0
    while i < CUTOFF:
        number = random.normalvariate(mu, sigma)
        if lower < number < upper:
            return round(number)
        i += 1
    raise Exception(f"mu:{mu} sigma:{sigma}, not a single hit in {CUTOFF} tries.")


def truncated_normalvariate_float(mu, sigma, lower, upper):
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
    CUTOFF = 100000000
    i = 0
    while i < CUTOFF:
        number = random.normalvariate(mu, sigma)
        if lower < number < upper:
            return number
        i += 1
    raise Exception(f"mu:{mu} sigma:{sigma}, not a single hit in {CUTOFF} tries.")


def truncated_weibull(scale, shape, min, max):
    while True:
        number = random.weibullvariate(scale, shape)
        if min < number <= max:
            return int(number)


def truncated_weibull_float(scale, shape, min, max):
    while True:
        number = random.weibullvariate(scale, shape)
        if min < number <= max:
            return float(number)


def return_nearest_power_of(*, number, base):
    if base == 1:
        return number
    else:
        next_num = base ** math.ceil(math.log(number, base))
        prev_num = base ** math.floor(math.log(number, base))
        if next_num - number < number - prev_num:
            return next_num
        else:
            return prev_num


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
    df.set_index('time', inplace=True)
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


def create_file_indexed(prefix: str, path: str = None, ending: str = None, create=True) -> str:
    if path is not None:
        os.makedirs(path, exist_ok=True)
    else:
        path = "./"
    index = 1
    while True:
        if ending:
            filename = f"{prefix}_{index:03d}.{ending}"
        else:
            filename = f"{prefix}_{index:03d}"
        filepath = os.path.join(path, filename)
        if not os.path.exists(filepath):
            if create:
                open(filepath, "w").close()
            return filepath
        index += 1


def create_dir_indexed(dir: str, path: str = None) -> str:
    if dir is None:
        raise ValueError("'dir' cannot be none")
    if path is None:
        path = os.getcwd()
    index = 1
    while True:
        dirname = f"{dir}_{index:03d}"
        fullpath = os.path.join(path, dirname)
        if not os.path.exists(fullpath):
            os.makedirs(fullpath, exist_ok=False)
            return fullpath
        index += 1


def next_arrival_byconfargs(config, args, reset=False):
    args = to_dict(args)
    arrival_rate = 1
    arrival_time = config['JOB_ARRIVAL_TIME']
    downscale = args['downscale']

    if args['job_arrival_rate']:
        arrival_rate = args['job_arrival_rate']
    if args['job_arrival_time']:
        arrival_time = args['job_arrival_time']
    return next_arrival(arrival_rate / (arrival_time * downscale), reset)


def next_arrival_byconfkwargs(config, kwargs, reset=False):
    arrival_rate = 1
    arrival_time = config['JOB_ARRIVAL_TIME']
    if kwargs['job_arrival_rate']:
        arrival_rate = kwargs['job_arrival_rate']
    if kwargs['job_arrival_time']:
        arrival_time = kwargs['job_arrival_time']
    return next_arrival(arrival_rate / arrival_time, reset)


def next_arrival(lambda_rate, reset=False, start_time=0):
    if not hasattr(next_arrival, 'next_time') or reset is True:
        # Initialize the first time it's called
        next_arrival.next_time = start_time
    else:
        next_arrival.next_time += \
            -math.log(1.0 - random.random()) / lambda_rate
    return next_arrival.next_time


TIME_UNITS = {
    'd': timedelta(days=1),
    'h': timedelta(hours=1),
    'm': timedelta(minutes=1),
    's': timedelta(seconds=1),
    'ds': timedelta(milliseconds=100),
    'cs': timedelta(milliseconds=10),
    'ms': timedelta(milliseconds=1),
}


def parse_time_unit(unit) -> timedelta:
    parsed_unit = unit
    if TypeAdapter(timedelta).validator.isinstance_python(unit):
        parsed_unit = TypeAdapter(timedelta).validate_python(unit)
    elif isinstance(unit, str):
        parsed_unit = TIME_UNITS.get(unit)
    if not isinstance(parsed_unit, timedelta):
        raise ValueError(f"Invalid time unit {unit}")
    if parsed_unit not in TIME_UNITS.values() or parsed_unit > TIME_UNITS['s']:
        raise ValueError("Only time units of s, ds, cs, and ms are supported")
    return parsed_unit


def parse_td(td, unit: str | timedelta = 's') -> timedelta:
    """ Parse into a timedelta. Pass unit to interpret raw numbers as (default seconds) """
    unit = parse_time_unit(unit)
    if TypeAdapter(int).validator.isinstance_python(td):
        return unit * TypeAdapter(int).validate_python(td)
    if TypeAdapter(timedelta).validator.isinstance_python(td):
        return TypeAdapter(timedelta).validate_python(td)
    if isinstance(td, str):
        if not pd.isna(pd.to_timedelta(td, errors="coerce")):
            return pd.to_timedelta(td)
        # Special case parsing for ds and cs units which pandas doesn't support
        re_match = re.fullmatch(r"(\d+)\s*(\w+)", td.strip())
        if re_match and re_match[2] in TIME_UNITS:
            num_str, unit_str = re_match.groups()
            return int(num_str) * TIME_UNITS[unit_str]
    raise ValueError(f"Invalid timedelta: {td}")


def convert_to_time_unit(td, unit: str | timedelta = 's'):
    """
    Converts to integer number of time unit
    Throws if the given time is less than the unit
    """
    num = parse_td(td, unit) / parse_time_unit(unit)
    if (num != 0 and num < 1) or not num.is_integer():
        raise ValueError(f"{td} is not divisible by time unit {unit}")
    return int(num)


def infer_time_unit(td) -> timedelta:
    """ Infers the time unit the user meant for the input string """
    parsed_td = parse_td(td)
    time_unit = None
    if isinstance(td, str):  # infer unit from string, e.g. 1s or 200ms
        re_match = re.fullmatch(r"(\d+)\s*(\w+)", td.strip())
        if re_match and re_match[2] in TIME_UNITS:
            time_unit = TIME_UNITS[re_match[2]]
    if not time_unit:
        for unit in sorted(TIME_UNITS.values(), reverse=True):
            if (parsed_td % unit).total_seconds() == 0:
                time_unit = unit
                break
    return min(TIME_UNITS['s'], time_unit or TIME_UNITS['s'])


def encrypt(name):
    """Encrypts a given name using SHA-256 and returns the hexadecimal digest."""
    encoded_name = name.encode()
    hash_object = hashlib.sha256(encoded_name)
    return hash_object.hexdigest()


def write_dict_to_file(dictionary, file_path):
    """Function to write dictionary to a text file"""
    with open(file_path, 'w') as file:
        file.write("{")
        for j, (key, value) in enumerate(dictionary.items()):
            if isinstance(value, dict):
                file.write(f"\"{str(key)}\": {{\n")
                for i, (subkey, subvalue) in enumerate(value.items()):
                    base_subvalue = convert_numpy_to_builtin(subvalue)
                    json_string = toJSON(base_subvalue)
                    file.write(f"  \"{str(subkey)}\": {json_string}")
                    if i < len(value.items()) - 1:
                        file.write(", ")
                file.write("}")
            else:
                file.write(f"\"{str(key)}\": {value}")
            if j < len(dictionary.items()) - 1:
                file.write(", ")
            file.write("\n")
        file.write("}")


def toJSON(obj):
    """Function to dump a json string from object"""
    return json.dumps(
        obj,
        default=lambda o: o.__dict__,
        sort_keys=True,
        indent=4)


def convert_numpy_to_builtin(obj):
    if isinstance(obj, dict):
        tmp_obj = dict()
        for k, v in obj.items():
            tmp_obj[k] = convert_numpy_to_builtin(v)
        return tmp_obj
    elif isinstance(obj, list):
        return [convert_numpy_to_builtin(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        tmplist = obj.tolist()
        return convert_numpy_to_builtin(tmplist)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        return obj


def get_current_utilization(trace, job: Job):
    """Return utilization for a trace at the job's current running time.
       Note: this should move to a trace.py and a Trace class!
    """
    if (isinstance(trace, list) and trace) or \
       (isinstance(trace, np.ndarray) and trace.size != 0):

        if not job.trace_quanta:
            raise ValueError("job.trace_quanta is not set; cannot compute utilization.")

        time_quanta_index = int((job.current_run_time - job.trace_start_time) // job.trace_quanta)
        if time_quanta_index < 0:
            time_quanta_index = 0

        if time_quanta_index < len(trace):
            util = get_utilization(trace, time_quanta_index)
        else:
            util = get_utilization(trace, max(0, len(trace) - 1))
    elif isinstance(trace, (float, int)):
        util = trace
    else:
        raise ValueError(f"trace is of unexpected type: {type(trace)}.")
        util = 0.0

    return util


def get_utilization(trace, time_quanta_index):
    """Retrieve utilization value for a given trace at a specific time quanta index."""
    if isinstance(trace, (list, np.ndarray)):
        return trace[time_quanta_index]
    elif isinstance(trace, (int, float)):
        return float(trace)
    else:
        raise TypeError(f"Invalid type for utilization: {type(trace)}.")


class ValueComparableEnum(Enum):
    def __eq__(self, other):
        if isinstance(other, Enum):
            return self.value == other.value
        return self.value == other

    def __hash__(self):  # required if you override __eq__
        return hash(self.value)


def normalize_tz(d: datetime):
    """ Convert datetime to UTC. If naive, assume local time, then convert to UTC """
    if not d.tzinfo:
        return d.astimezone().astimezone(timezone.utc)
    else:
        return d.astimezone(timezone.utc)


def validate_resolved_path(path: str | Path, info: ValidationInfo):
    context = info.context or {}
    path = Path(path).expanduser()
    if context.get('base_path'):
        base_path = Path(context["base_path"]).expanduser().resolve()
    else:
        base_path = Path.cwd()
    path = (base_path / path).resolve()
    # This is used on the simulation server to block reading arbitrary files
    if context.get("force_under_base_path"):
        if not path.is_relative_to(base_path):
            raise ValueError(f"{path} is not under {base_path}")
    return path


ResolvedPath = A[Path, AfterValidator(validate_resolved_path)]
"""
Resolve a path, and expand ~ in the path string.
Paths can be resolved relative to specific path instead of cwd by passing
`context={"base_path": "my/path"}` in model_validate().
"""


AutoAwareDatetime = A[datetime, AfterValidator(normalize_tz)]
""" Datetime type wrapper, makes sure timezone is set """

SmartTimedelta = A[timedelta, BeforeValidator(parse_td)]
""" Can be passed as ISO 8601 format like PT5M, or a string like 9s, or a number of seconds """


class RAPSBaseModel(BaseModel):
    """ Base Pydantic model with shared config """
    model_config = ConfigDict(
        use_attribute_docstrings=True,
    )


T = TypeVar("T", bound=BaseModel, covariant=True)


class ModelArgsValidator(Protocol[T]):
    def __call__(self, args: argparse.Namespace, init_data: dict | None = None) -> T:
        ...


def pydantic_add_args(
    parser: argparse.ArgumentParser, model_cls: type[T],
    model_config: SettingsConfigDict | None = None,
) -> ModelArgsValidator[T]:
    """
    Add arguments to the parser from the model. Returns a function that can be used to parse the
    model from the argparse args.

    Normally you'd just configure Pydantic to just automatically create a BaseSettings object from
    sys.argv and/or env variables. But we want a bit more control over the cli parser, and to use
    the SimConfig model as a regular non-settings model in the simulation server. So here we do
    some hacks to apply the args manually.
    """
    model_config_dict = SettingsConfigDict({
        "cli_implicit_flags": True,
        "cli_kebab_case": True,
        "title": model_cls.__name__,
        **(model_config or {}),
        "cli_parse_args": False,  # Don't automatically parse args
    })

    class SettingsModel(model_cls, BaseSettings):
        @classmethod
        def settings_customise_sources(cls, settings_cls,
                                       init_settings, env_settings, dotenv_settings, file_secret_settings,
                                       ):
            return (init_settings,)  # Don't load from env vars or anything else

        model_config = model_config_dict

    cli_settings_source = CliSettingsSource(SettingsModel, root_parser=parser)

    def model_args_validator(args: argparse.Namespace, init_data: dict | None = None):
        try:
            model = CliApp.run(SettingsModel,
                               cli_args=args,
                               cli_settings_source=cli_settings_source,
                               **(init_data or {}),
                               )
            # Recreate model so we don't return the SettingsModel subclass
            # use exclude_unset so that model_field_set is preserved as well
            return model_cls.model_validate(model.model_dump(exclude_unset=True))
        except (ValidationError, SettingsError) as err:
            print(err)
            sys.exit(1)
    return model_args_validator


SubParsers: TypeAlias = "argparse._SubParsersAction[argparse.ArgumentParser]"
""" Alias for the result of argparse parser.add_subparsers """


def yaml_dump(data, header_comment=''):
    """ Dumps yaml with pretty formatting """
    if header_comment:
        header_comment = '\n'.join(f'# {ln}' for ln in header_comment.splitlines()) + "\n"

    class IndentDumper(yaml.Dumper):
        def represent_data(self, data):
            # Quote all strings with special characters to avoid confusion
            if (
                isinstance(data, str) and
                (not re.fullmatch(r"[\w-]+", data) or data.isdigit()) and
                "\n" not in data
            ):
                return self.represent_scalar('tag:yaml.org,2002:str', data, style='"')
            return super(IndentDumper, self).represent_data(data)

        def increase_indent(self, flow=False, indentless=False):
            # Indent lists
            return super(IndentDumper, self).increase_indent(flow, False)

    return header_comment + yaml.dump(
        data,
        Dumper=IndentDumper,
        sort_keys=False,
        indent=2,
        allow_unicode=True,
    )


def read_yaml(config_file: str | None) -> dict:
    """ Parses yaml file. Pass "-" to read from stdin """
    # Assume stdin if not terminal
    if config_file == "-" or (not config_file and not sys.stdin.isatty()):
        data = sys.stdin.read()
    elif config_file:
        data = Path(config_file).read_text()
    else:
        data = ""
    if data.strip():
        result = yaml.safe_load(data)
    else:
        result = {}
    if not isinstance(result, dict):
        raise ValueError("Expected yaml document to contain a top-level mapping")
    return result


def read_yaml_parsed(cls: type[T], config_file=None) -> dict:
    """
    Like read_yaml, but parses the input to resolve paths etc.
    Exits on error after printing message (for use in the CLI)
    """
    try:
        yaml_data = read_yaml(config_file)
        if yaml_data:
            # Resolve paths in yaml relative to the yaml file
            base_path = Path(config_file).parent if config_file and config_file != "-" else None
            model = cls.model_validate(yaml_data, context={"base_path": base_path})
            yaml_data = model.model_dump(mode='json', exclude_unset=True)
    except (ValidationError, ValueError, YAMLError) as err:
        print(f'Failed to parse yaml "{config_file}"')
        print(err)
        sys.exit(1)
    return yaml_data


def is_yaml_file(path: str | Path):
    """ Return true if the path is .yaml, .yml, or .json """
    return Path(path).suffix in ['.yaml', '.yml', '.json']


class WorkloadData(RAPSBaseModel):
    """
    Represents a workload, a list of jobs with some metadata. Returned by dataloaders load_data()
    function, and by Workload.generate_jobs().

    jobs:
        The list of parsed jobs.

    telemetry_start
        the first timestep in which the simulation be executed.

    telemetry_end
        the last timestep in which the simulation can be executed.

    start_date
        The actual date that telemetry_start represents.
    ----
    Explanation regarding times:

    The loaded dataframe contains
    a first timestamp with associated data
    and a last timestamp with associated data

    These form the maximum extent of the simuluation time.
    telemetry_start and telemetry_end.

            [                                    ]
            ^                                    ^
            telemetry_start          telemetry_end

    These values form the maximum extent of the simulation.
    telemetry_start is typically 0, but any int can be used as long as all the times in the
    jobs are relative to the telemetry_start.

    Next is the actual extent of the simulation:

            [                                   ]
                ^                   ^
                simulation_start    simulation_end

    The simulation will start at telemetry_start by default, but the user can specify an explicit
    simulation start time.

    Additionally, jobs can have started before telemetry_start,
    And can have a recorded ending after simulation_end,
            [                                   ]
    ^                                                ^
    first_start_timestamp           last_end_timestamp

    This means that the time between first_start_timestamp and telemetry_start
    has no associated values in the traces!
    The missing values after simulation_end can be ignored, as the simulatuion
    will have stoped before.

    However, the times before telemetry_start have to be padded to generate
    correct offsets within their data!
    Within the simulation a job's current time is specified as the difference
    between its start_time and the current timestep of the simulation.

    With this each job's
    - submit_time
    - time_limit
    - start_time  # Maybe Null
    - end_time  # Maybe Null
    - expected_run_time (end_time - start_time)  # Maybe Null
    - current_run_time (How long did the job run already, when loading)  # Maybe zero
    - trace_time (lenght of each trace in seconds)  # Maybe Null
    - trace_start_time (time offset in seconds after which the trace starts)  # Maybe Null
    - trace_end_time (time offset in seconds after which the trace ends)  # Maybe Null
    - trace_quanta (job's associated trace quanta, to correctly replay with different trace quanta) # Maybe Null
    has to be set for use within the simulation

    The values trace_start_time are similar to the telemetry_start and
    telemetry_stop but may different due to missing data, for each job.

    The returned values are these:
        - The list of parsed jobs. (as a Job object)
        - telemetry_start: int (in seconds)
        - telemetry_end: int (in seconds)
        - start_date: datetime
    """
    jobs: list[Job]
    telemetry_start: int
    telemetry_end: int
    # TODO: It might make more sense to make start_timestep/end_timestep always unix time, then we
    # wouldn't need this extra start_date field.
    # Don't use AutoAwareDatetime here as we want to enforce dataloaders returning timezone info
    start_date: A[AwareDatetime, AfterValidator(lambda d: d.astimezone(timezone.utc))]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
