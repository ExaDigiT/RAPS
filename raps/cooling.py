"""
This module provides functionality for simulating a thermo-fluids model using
an FMU (Functional Mock-up Unit).

The module defines a `ThermoFluidsModel` class that encapsulates the
initialization, simulation step execution,
data conversion, and cleanup processes for the FMU-based model.
"""
import shutil
import re
import numpy as np
from uncertainties import unumpy
from uncertainties.core import AffineScalarFunc

from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
from datetime import timedelta

from raps.policy import PolicyType


def get_matching_variables(variables, pattern):
    # Regex pattern to match strings containing .summary
    pattern = re.compile(pattern)

    # Filtering the list using the regex pattern
    filtered_vars = [var for var in variables if pattern.match(var)]

    return filtered_vars


class ThermoFluidsModel:
    """
    A class to represent a thermo-fluids model using an FMU (Functional Mock-up Unit).

    This class encapsulates the initialization, simulation step execution, data conversion,
    and cleanup processes for the FMU-based thermo-fluids model. It provides methods to
    initialize the model, execute simulation steps, generate runtime values, calculate Power
    Usage Effectiveness (PUE), and properly manage the FMU resources.

    Attributes
    ----------
    FMU_PATH : str
        The file path to the FMU file.
    fmu_history : list
        A list to store the history of FMU states, combining cooling input, datacenter output,
        and central energy plant (CEP) output for each simulation step.
    inputs : list
        A list of input variables for the FMU.
    outputs : list
        A list of output variables for the FMU.
    unzipdir : str
        The directory where the FMU file is extracted.
    fmu : FMU2Slave
        The instantiated FMU object.
    weather : Optional
        An object that provides weather-related data for simulations. Used when replay mode is on.

    Methods
    -------
    initialize():
        Initializes the FMU by extracting the file, reading the model description,
        setting up input and output variables, and preparing the model for simulation.
    generate_runtime_values(cdu_power, sc) -> dict:
        Generates runtime values dynamically for the FMU inputs based on CDU power and other configuration parameters.
    generate_fmu_inputs(runtime_values: dict, uncertainties: bool = False) -> list:
        Converts runtime values to a list suitable for FMU inputs, handling uncertainties if specified.
    calculate_pue(cooling_input: dict, datacenter_output: dict, cep_output: dict) -> float:
        Calculates the Power Usage Effectiveness (PUE) of the data center based on the cooling, datacenter,
        and CEP output power values.
    step(current_time: float, fmu_inputs: list, step_size: float) -> Tuple[dict, dict, dict, float]:
        Executes a simulation step with the given inputs and step size. Returns the cooling input, datacenter output,
        CEP output, and PUE for the current step.
    terminate():
        Terminates the FMU instance, ensuring that all resources are properly released.
    cleanup():
        Cleans up the extracted FMU directory, ensuring no temporary files are left behind.
    """

    def __init__(self, **config):
        """
        Constructs all the necessary attributes for the ThermoFluidsModel object.

        Parameters
        ----------
        FMU_PATH : str
            The file path to the FMU file.
        """
        self.config = config
        self.fmu_history = []
        self.inputs = None
        self.outputs = None
        self.unzipdir = None
        self.fmu = None
        self.weather = None

    def initialize(self):
        """
        Initializes the FMU by extracting the file and setting up the model.

        This method unzips the FMU file, reads the model description,
        collects value references for input and output variables,
        and initializes the FMU for simulation.
        """
        # Notify user that FMU is initializing
        print('Initializing FMU...')

        # Unzip the FMU file and get the unzip directory
        self.unzipdir = extract(self.config['FMU_PATH'])
        model_description = read_model_description(self.config['FMU_PATH'])

        # Add to list of variable names
        var_model = []
        for variable in model_description.modelVariables:
            var_model.append(variable.name)

        outputs = get_matching_variables(var_model, r'.*(\.summary\.|^summary).*')

        # Get the value references for the variables we want to get/set
        self.inputs = [v for v in model_description.modelVariables if v.causality == 'input']
        self.outputs = [v for v in model_description.modelVariables if v.name in outputs]

        # Instantiate and initialize the FMU
        self.fmu = FMU2Slave(guid=model_description.guid,
                             unzipDirectory=self.unzipdir,
                             modelIdentifier=model_description.coSimulation.modelIdentifier,
                             instanceName='instance1')
        self.fmu.instantiate()
        self.fmu.setupExperiment(startTime=0.0)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()

    def generate_runtime_values(self, cdu_power, engine) -> dict:
        """
        Generate the runtime values for the FMU inputs dynamically.

        Parameters:
        cdu_power (array): The array of CDU powers.
        sc (Scheduler Object): The current instance of a Scheduler.

        Returns:
        dict: A dictionary with the runtime values for the FMU inputs.
        """
        # Dynamically generate the power inputs
        runtime_values = {
            f"simulator_1_datacenter_1_computeBlock_{i + 1}"
            f"_cabinet_1_sources_Q_flow_total": cdu_power[i] *
            self.config['COOLING_EFFICIENCY'] / self.config['RACKS_PER_CDU']
            for i in range(self.config['NUM_CDUS'])
        }

        # Default temperature is from the config
        temperature = self.config['WET_BULB_TEMP']

        # If replay mode is on and weather data is available
        if engine.scheduler.policy == PolicyType.REPLAY and \
            self.weather and self.weather.start is not None and \
                self.weather.has_coords:
            # Convert total seconds to timedelta object
            delta = timedelta(seconds=engine.current_timestep)
            target_datetime = self.weather.start + delta

            # Get temperature from weather data
            temperature = self.weather.get_temperature(target_datetime) or self.config['WET_BULB_TEMP']

        # Set the temperature value(s)
        for temperature_key in self.config['TEMPERATURE_KEYS']:
            runtime_values[temperature_key] = temperature

        return runtime_values

    def generate_fmu_inputs(self, runtime_values, uncertainties=False):
        """
        Convert the runtime values based on the cooling model's inputs to a list suitable for FMU inputs.
        Raises an error if any input key is missing in runtime values.

        Parameters
        ----------
        runtime_values : dict
            A dictionary containing runtime values for FMU inputs.
        uncertainties : bool, optional
            If True, processes the values to strip uncertainties for certain inputs.

        Returns
        -------
        fmu_inputs : list
            A list of input values suitable for FMU.
        """
        # Initialize an empty list for FMU inputs
        fmu_inputs = []

        # Helper function to process uncertainty
        def process_uncertainty(value):
            """Strip uncertainty if present, otherwise return the value as-is."""
            # Convert to nominal value if it's an AffineScalarFunc and uncertainties flag is set
            return unumpy.nominal_values(value) if uncertainties and isinstance(value, AffineScalarFunc) else value

        # Iterate through the cooling model's inputs
        for input_var in self.inputs:
            input_name = input_var.name  # Get the name of the input variable

            # Fetch the runtime value for the input name
            try:
                value = runtime_values[input_name]
            except KeyError:
                raise KeyError(f"Missing value for key '{input_name}' in runtime values.")

            # Process the value based on uncertainty and append
            fmu_inputs.append(process_uncertainty(value))

        return fmu_inputs

    def calculate_pue(self, cooling_input, cooling_output):
        """
        Calculate the Power Usage Effectiveness (PUE) of the data center.

        Parameters
        ----------
        cooling_input : dict
            A dictionary containing input power values for cooling.
        datacenter_output : dict
            A dictionary containing output power values for the datacenter.
        cep_output : dict
            A dictionary containing output power values for the central energy plant.

        Returns
        -------
        pue : float
            The calculated Power Usage Effectiveness (PUE).
        """
        # Utility function to convert kW to Watts
        def convert_to_watts(value_in_kw):
            """Convert a value in kilowatts to Watts."""
            return np.array(value_in_kw) * 1e3 if value_in_kw is not None else 0.0

        # Convert values from kW to Watts using the utility function
        W_HTWPs = convert_to_watts(cooling_output.get(self.config['W_HTWPs_KEY']))
        W_CTWPs = convert_to_watts(cooling_output.get(self.config['W_CTWPs_KEY']))
        W_CTs = convert_to_watts(cooling_output.get(self.config['W_CTs_KEY']))

        # Get the sum of the work done by all CDU pumps
        W_CDUPs = sum(
            convert_to_watts(cooling_output.get(
                f'simulator[1].datacenter[1].computeBlock[{idx + 1}].cdu[1].summary.W_flow_CDUP_kW'))
            for idx in range(self.config['NUM_CDUS'])
        )

        # Sum all values in the cooling_input dictionary
        total_cooling_input_power = np.sum(list(cooling_input.values()))

        # Ensure a non-zero value for total input power to avoid division by zero
        total_input_power = np.maximum(total_cooling_input_power, 1e-3)

        # Calculate PUE
        pue = (total_input_power + np.sum(W_CDUPs) + np.sum(W_HTWPs) +
               np.sum(W_CTWPs) + np.sum(W_CTs)) / total_input_power

        return pue

    def step(self, current_time, fmu_inputs, step_size):
        """
        Executes a simulation step with the given inputs and step size.

        Parameters
        ----------
        current_time : float
            The current simulation time.
        fmu_inputs : list
            A list of input values to set in the FMU.
        step_size : float
            The size of the simulation step.

        Returns
        -------
        cooling_input : dict
            A dictionary containing the input values for cooling.
        datacenter_output : dict
            A dictionary containing the output values for the datacenter.
        cep_output : dict
            A dictionary containing the output values for the central energy plant.
        pue : float
            The Power Usage Effectiveness (PUE) calculated from the outputs.
        """
        # Set FMU inputs
        for index, v in enumerate(self.inputs):
            self.fmu.setReal([v.valueReference], [fmu_inputs[index]])

        # Perform one step in the FMU
        self.fmu.doStep(currentCommunicationPoint=current_time, communicationStepSize=step_size)

        # Initialize dictionaries for cooling input, datacenter output, and CEP output
        cooling_inputs = {v.name: self.fmu.getReal([v.valueReference])[0] for v in self.inputs}
        cooling_outputs = {v.name: self.fmu.getReal([v.valueReference])[0] for v in self.outputs}

        # Calculate PUE
        pue = self.calculate_pue(cooling_inputs, cooling_outputs)

        # Append time to each dictionary
        cooling_inputs['time'] = current_time
        cooling_outputs['pue'] = pue

        # Append the combined results to the history
        self.fmu_history.append({**cooling_inputs, **cooling_outputs})

        return cooling_inputs, cooling_outputs

    def terminate(self):
        """
        Terminates the FMU instance.

        This method properly terminates the FMU instance, ensuring that all
        resources are released.
        """
        # Close the FMU
        self.fmu.terminate()
        self.fmu.freeInstance()

    def cleanup(self):
        """
        Cleans up the extracted FMU directory.

        This method removes the directory where the FMU file was extracted,
        ensuring no temporary files are left behind.
        """
        # Cleanup - at the end of the simulation
        shutil.rmtree(self.unzipdir, ignore_errors=True)

    def simulate_cooling(self, *, rack_power, engine):
        cdu_power = rack_power.T[-1] * 1000
        runtime_values = self.generate_runtime_values(cdu_power, engine)

        # FMU inputs are N powers and the wetbulb temp
        fmu_inputs = self.generate_fmu_inputs(runtime_values,
                                              uncertainties=engine.power_manager.uncertainties)
        cooling_inputs, cooling_outputs = self.step(engine.current_timestep,
                                                    fmu_inputs,
                                                    engine.config['POWER_UPDATE_FREQ'])
        return cooling_inputs, cooling_outputs
