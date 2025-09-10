import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn
)

from contextlib import nullcontext

from raps.utils import summarize_ranges, convert_seconds_to_hhmmss, convert_seconds_to_hhmm
from raps.constants import ELLIPSES
from raps.engine import TickData, Engine

MAX_ROWS = 30

class LayoutManager:
    def __init__(self, layout_type, engine: Engine, total_timesteps=0, debug=None, args_dict=None, **config):
        self.debug = debug
        if args_dict is not None:
            self.noui = args_dict.get("noui")
            self.simulate_network = args_dict.get("simulate_network")
        else:
            self.noui = False
            self.simulate_network = False
        self.engine = engine
        self.config = config
        self.topology = self.engine.config.get("TOPOLOGY", "none")
        self.hascooling = layout_type == "layout2"
        self.power_df_header = self.config['POWER_DF_HEADER']
        self.racks_per_cdu = self.config['RACKS_PER_CDU']
        self.power_column = self.power_df_header[self.racks_per_cdu + 1]
        self.loss_column = self.power_df_header[-1]

        if self.debug or self.noui:
            return

        self.console = Console()
        self.layout = Layout()
        self.setup_layout(layout_type)
        self.progress = Progress(
            TextColumn("Progress: [progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(bar_width=None),
            TextColumn("•"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn()
        )
        self.progress_task = self.progress.add_task("Progress", total=total_timesteps, name="Progress")

    def setup_layout(self, layout_type):
        if not self.debug:
            self.layout.split_column(Layout(name="main"), Layout(name="progress", size=1))
            if layout_type == "layout2":
                self.layout["main"].split_row(Layout(name="left", ratio=3), Layout(name="right", ratio=2))
                self.layout["main"]["left"].split_column(
                    Layout(name="pressflow", ratio=6),
                    Layout(name="powertemp", ratio=11),
                    Layout(name="totpower", ratio=3),
                )
                self.layout["main"]["right"].split(Layout(name="scheduled", ratio=17), Layout(name="status", ratio=3))
            else:
                self.layout["main"].split_row(Layout(name="left", ratio=1), Layout(name="right", ratio=1))
                self.layout["main"]["left"].split_column(Layout(name="upper", ratio=8), Layout(name="lower", ratio=2))
                self.layout["main"]["right"].split_column(
                    Layout(name="scheduled", ratio=8), Layout(name="status", ratio=2))

    def create_table(self, title, columns, header_style="bold green"):
        """
        Creates a Rich Table with the given title and columns.

        Parameters
        ----------
        title : str
            Title of the table.
        columns : list of str
            List of column headers.
        header_style : str, optional
            Style for the headers (default is "bold green").

        Returns
        -------
        Table
            The created Rich Table.
        """
        table = Table(title=title, expand=True, header_style=header_style)
        for col in columns:
            table.add_column(col, justify="center")
        return table

    def add_table_rows(self, table, data, format_funcs=None):
        format_funcs = format_funcs or [str] * len(data[0])
        for row in data:
            formatted_row = [func(cell) for func, cell in zip(format_funcs, row)]
            table.add_row(*formatted_row)

    def calculate_totals(self, df):  # 'Sum' and 'Loss' columns
        total_power_kw = df[self.power_column].sum() + (self.config['NUM_CDUS'] * self.config['POWER_CDU'] / 1000.0)
        total_power_mw = total_power_kw / 1000.0
        total_loss_kw = df[self.loss_column].sum()
        total_loss_mw = total_loss_kw / 1000.0
        return \
            total_power_mw, \
            total_loss_mw, \
            f"{total_loss_mw / total_power_mw * 100:.2f}%", \
            total_power_kw, total_loss_kw

    def update_scheduled_jobs(self, jobs, show_nodes=False):
        """
        Updates the displayed scheduled jobs table with the provided job information.

        Parameters
        ----------
        jobs : list
            A list of job objects containing job information.
        show_nodes : bool, optional
            Flag indicating whether to display node information (default is False).
        """

        # Decide whether to show "SLOWDOWN" (if real topology) or "NODE SEGMENTS" (if capacity/none)
        # show_slowdown = (self.topology in ("fat-tree", "dragonfly", "capacity"))
        show_slowdown = self.simulate_network

        # Build the column headers
        # columns = ["JOBID", "WALL TIME", "NAME", "ACCOUNT", "ST"]
        columns = ["JOBID", "TIME LIMIT", "NAME", "ACCOUNT", "ST", "NODES"]
        if show_slowdown:
            columns.append("SLOW DOWN")
        else:
            if show_nodes:
                columns.append("NODELIST")
            else:
                columns.append("SEGMENT")  # NODE SEGMENTS

        columns.append("WALL TIME")

        # Create table with bold magenta headers
        table = Table(title="Job Queue", header_style="bold magenta", expand=True)
        for col in columns:
            table.add_column(col, justify="center")

        # Add data rows
        for job in jobs[:MAX_ROWS]:
            # Number of requested nodes as a string
            # n_nodes = str(job.nodes_required)  # Unused

            if show_slowdown:
                # Each Job should have job.net_congestion set in Engine.tick()
                slow = getattr(job, "slowdown_factor", 0.0)
                # Format as "1.23×" (if ≤1.00 you will see "1.00×")
                slowdown_str = f"{slow:.2f}×"
                col_slow = slowdown_str
            else:
                # Fallback to original NODE SEGMENTS logic
                node_segments = summarize_ranges(job.scheduled_nodes)
                if show_nodes:
                    if len(node_segments) > 4:
                        nodes_display = ", ".join(node_segments[:2] + [ELLIPSES] + node_segments[-2:])
                    else:
                        nodes_display = ", ".join(node_segments)
                    col_slow = nodes_display  # reused variable name for simplicity
                else:
                    # col_slow = str(len(node_segments))
                    col_slow = str(len(node_segments))

            # If show_nodes is True, we need to append NODELIST as well
            if show_nodes and not show_slowdown:
                # use the same node_segments variable to build the list of nodes
                if len(node_segments) > 4:
                    nodes_display = ", ".join(node_segments[:2] + [ELLIPSES] + node_segments[-2:])
                else:
                    nodes_display = ", ".join(node_segments)
                col_nodelist = nodes_display
            else:
                col_nodelist = col_slow  # This logic is a bit flawed...
                nodes_display = col_nodelist

            if self.engine.downscale != 1:
                running_time_str = convert_seconds_to_hhmmss(job.running_time // self.engine.downscale) + \
                    f" +{job.running_time % self.engine.downscale}/{self.engine.downscale}s"
            else:
                running_time_str = convert_seconds_to_hhmm(job.running_time)

            row = [
                str(job.id).zfill(5),
                convert_seconds_to_hhmm(job.time_limit // self.engine.downscale),
                # str(job.wall_time),
                str(job.name),
                str(job.account),
                job.current_state.value,
                str(job.nodes_required),
                nodes_display,
                running_time_str
            ]

            # If the job has been flagged as “dilated”, show its row in yellow
            if getattr(job, "dilated", False):
                row = [f"[yellow]{x}[/yellow]" for x in row]

            table.add_row(*row, style="white")

        # Update the layout
        self.layout["scheduled"].update(Panel(Align(table, align="center")))

    def update_status(self,
                      time,
                      nrun,
                      nqueue,
                      active_nodes,
                      free_nodes,
                      down_nodes,
                      avg_net_util,
                      slowdown,
                      time_delta,
                      timestep_start=0):
        """
        Updates the status information table with the provided system status data.

        Parameters
        ----------
        time : int or float
            The current time in seconds.
        nrun : int
            Number of jobs currently running.
        nqueue : int
            Number of jobs currently queued.
        active_nodes : int
            Number of active nodes.
        free_nodes : int
            Number of free nodes.
        down_nodes : list
            List of nodes that are down.
        """
        # Define columns with header styles
        columns = []
        time_header = "Time"
        if timestep_start != 0:  # append time simulated
            time_header += " (+Sim)"
        columns.append(time_header)
        columns.append("Jobs Running")
        columns.append("Jobs Queued")
        columns.append("Active Nodes")
        columns.append("Free Nodes")
        columns.append("Down Nodes")
        columns.append("Speed")

        if self.simulate_network:
            columns.extend(("Net Util (%)", "Slowdown per job"))
        table = Table(header_style="bold magenta", expand=True)
        for col in columns:
            table.add_column(col, justify="center")

        row = []
        # Add data row with white values
        time_in_s = time // self.engine.downscale
        if (time_in_s < 946684800):  # Introducing Y2K into our codebase! Kek
            time_str = convert_seconds_to_hhmm(time_in_s)
        else:
            # For the curious: If the simulation time in seconds is large than
            # unix timestamp for Jan 2000 this is a unix timestamp,
            time_str = f"{datetime.fromtimestamp(time_in_s).strftime('%Y-%m-%d %H:%M')}"
        if timestep_start != 0:  # append time simulated
            time_str += f"\nSim: {convert_seconds_to_hhmm(time_in_s - timestep_start)}"

        row.append(time_str)
        row.append(str(nrun))
        row.append(str(nqueue))
        row.append(str(active_nodes))
        row.append(str(free_nodes))
        row.append(str(len(down_nodes)))
        row.append(f"{time_delta}x")
        if self.simulate_network:
            row.append(f"{avg_net_util * 100:.0f}%")
            row.append(f"{slowdown:.1f}x")
        # Add the row with the 'white' style applied to the whole row
        table.add_row(*row, style="white")

        # Set the width of each column to match the "Power Stats" table
        num_columns = len(table.columns)
        column_width = int(100 / num_columns)
        for column in table.columns:
            column.width = column_width

        # Update the layout
        self.layout["status"].update(Panel(Align(table, align="center"), title="Scheduler Stats"))

    def update_pressflow_array(self, cooling_outputs):
        fmu_cols = self.config['FMU_COLUMN_MAPPING']
        columns = ["Output", "Average Value"]

        datacenter_df = self.get_datacenter_df(cooling_outputs)

        # List of keys to include in the table
        relevant_keys = [
            "W_flow_CDUP_kW", "p_prim_s_psig", "p_prim_r_psig",
            "V_flow_prim_GPM", "V_flow_sec_GPM", "p_sec_r_psig", "p_sec_s_psig"
        ]

        # Dynamically build the data list using FMU_COLUMN_MAPPING

        data = []
        for key in relevant_keys:
            if key in datacenter_df and key in fmu_cols:
                label = fmu_cols[key]
                average_value = round(datacenter_df[key].mean(), 1)
                data.append((label, average_value))

        # Create table with white headers
        table = self.create_table("Pressure and Flow Rates", columns, header_style="bold white")
        self.add_table_rows(table, data)
        self.layout["pressflow"].update(Panel(table))

    def get_datacenter_df(self, cooling_outputs):
        # Initialize data dictionary with keys from FMU_COLUMN_MAPPING
        fmu_cols = self.config['FMU_COLUMN_MAPPING']
        data = {key: [] for key in fmu_cols.keys()}

        # Loop over each compute block in the datacenter_outputs dictionary
        for i in range(1, self.config['NUM_CDUS'] + 1):
            compute_block_key = f"simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary."

            # Append data to the corresponding lists dynamically using FMU_COLUMN_MAPPING keys
            for key in fmu_cols.keys():
                data[key].append(cooling_outputs.get(compute_block_key + key))

        # Convert to DataFrame
        df = pd.DataFrame(data)

        return df

    def update_powertemp_array(self,
                               power_df,
                               cooling_outputs,
                               pflops,
                               gflop_per_watt,
                               system_util,
                               uncertainties=False):
        """
        Updates the displayed power and temperature table with the provided data.

        Parameters
        ----------
        power_df : pandas.DataFrame
            DataFrame containing power data.
        cooling_df : pandas.DataFrame
            DataFrame containing temperature and cooling data.
        """
        # Define the specific columns for power
        # power_columns = POWER_DF_HEADER[0:RACKS_PER_CDU + 2] + [POWER_DF_HEADER[-1]]
        # "CDU", "Rack 1", "Rack 2", "Rack 3", "Sum", "Loss"
        power_columns = self.power_df_header[0:self.racks_per_cdu + 2] + [self.power_df_header[-1]]
        fmu_cols = self.config['FMU_COLUMN_MAPPING']

        # Updated cooling keys to include temperature instead of pressure
        cooling_keys = ["T_prim_s_C", "T_prim_r_C", "T_sec_s_C", "T_sec_r_C"]

        datacenter_df = self.get_datacenter_df(cooling_outputs)

        # Create column headers with appropriate styles
        columns = [f"{col} (kW)" if col != "CDU" else col for col in power_columns]
        columns += [fmu_cols[key] for key in cooling_keys]

        # Define styles for data values
        data_styles = ["bold cyan"] + ["bold green"] * (len(power_columns) - 1)
        data_styles += [
            "bold blue" if "Supply" in fmu_cols[key] else "bold red" for key in cooling_keys
        ]

        # Initialize the table with header styles
        table = Table(title="Power and Temperature", expand=True)
        for col in columns:
            table.add_column(col, justify="center")

        # Convert power DataFrame values to integers beforehand
        if uncertainties:
            pass
        else:
            power_df = power_df.replace([np.nan], 0.0)
            power_df = power_df.replace([np.inf], sys.maxsize)
            power_df = power_df.replace([-np.inf], -sys.maxsize - 1)
            power_df = power_df[power_columns].astype(int)

        # Populate the table with data from the DataFrame, applying the data styles
        for power_row, cooling_row in zip(power_df.iterrows(), datacenter_df.iterrows()):
            power_values = [
                f"[{data_styles[i]}]{power_row[1][col]}[/]" for i, col in enumerate(power_columns)
            ]

            cooling_values = [
                f"[{data_styles[i + len(power_columns)]}]{cooling_row[1][key]:.1f}[/]" for
                i, key in enumerate(cooling_keys)
            ]
            table.add_row(*(power_values + cooling_values))

        # Calculate total power and loss from power_df
        total_power_mw, total_loss_mw, percent_loss_str, _, _ = self.calculate_totals(power_df)
        total_power_str = f"{total_power_mw:.3f} MW"
        total_loss_str = f"{total_loss_mw:.3f} MW"

        self.layout["powertemp"].update(Panel(table))

        # Create Total Power table with green headers and white data
        total_table = Table(show_header=True, header_style="bold green")
        total_table.add_column("System Utilization", justify="center", style="green")
        total_table.add_column("Total Power", justify="center", style="green")
        total_table.add_column("PFLOPS", justify="center", style="green")
        total_table.add_column("GFLOPS/W", justify="center", style="green")
        total_table.add_column("Total Loss", justify="center", style="green")
        total_table.add_column("PUE", justify="center", style="green")

        # Add row with white data values using the style parameter
        total_table.add_row(
            f"{system_util:.1f}%",
            total_power_str,
            str(f"{pflops:.2f}"),
            str(f"{gflop_per_watt:.1f}"),
            total_loss_str + " (" + percent_loss_str + ")",
            f"{cooling_outputs['pue']:.2f}",
            style="white"  # Apply white style to all elements in the row
        )

        # Set the width of each column
        num_columns = len(total_table.columns)
        column_width = int(100 / num_columns)

        for column in total_table.columns:
            column.width = column_width

        self.layout["totpower"].update(Panel(Align(total_table, align="center"), title="Power and Performance"))

    def update_power_array(self, power_df, pflops, gflop_per_watt, system_util, uncertainties=False):
        """
        Updates the displayed power array table with the provided data from df.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing power and loss data for racks.
        """
        # Define the specific columns to display
        display_columns = self.power_df_header[0:self.racks_per_cdu + 2] + [self.power_df_header[-1]]

        # Extract only the relevant columns and round the values
        if uncertainties:
            pass
        else:
            power_df = power_df.replace([np.nan], 0.0)
            power_df = power_df.replace([np.inf], sys.maxsize)
            power_df = power_df.replace([-np.inf], -sys.maxsize - 1)
            power_df = power_df[display_columns].round().astype(int)

        # Create table for displaying rack power and loss with styling
        header_styles = ["bold green"] * len(display_columns)
        data_styles = ["cyan"] + ["white"] * (len(display_columns) - 1)

        # Initialize the table with header styles
        table = Table(title="Power Array of Racks (kW)", expand=True, header_style="bold green")
        for col, header_style in zip(display_columns, header_styles):
            table.add_column(col, justify="center", style=header_style)

        # Populate the table with data from the DataFrame, applying the data styles
        for _, row in power_df.iterrows():
            row_values = [
                f"[{data_styles[i]}]{value}[/{data_styles[i]}]"
                for i, value in enumerate(row[display_columns])
            ]
            table.add_row(*row_values)

        total_power_mw, total_loss_mw, percent_loss_str, total_power_kw, total_loss_kw = self.calculate_totals(power_df)

        # Convert to string with MW units
        total_power_str = f"{total_power_mw:.3f} MW"
        total_loss_str = f"{total_loss_mw:.3f} MW"
        percent_loss_str = f"{total_loss_mw / total_power_mw * 100:.2f}%"

        if not self.hascooling:
            self.layout["upper"].update(Panel(Align(table, align="center"),
                                        title=self.engine.config["system_name"].capitalize()))

            # Create Total Power table with green headers and white data
            total_table = Table(show_header=True, header_style="bold green")
            total_table.add_column("System Utilization", justify="center", style="green")
            total_table.add_column("Total Power", justify="center", style="green")
            total_table.add_column("PFLOPS", justify="center", style="green")
            total_table.add_column("GFLOPS/W", justify="center", style="green")
            total_table.add_column("Total Loss", justify="center", style="green")

            # Add row with white data values
            total_table.add_row(
                f"{system_util:.1f}%",
                total_power_str,
                str(f"{pflops:.2f}" if pflops is not None else "None"),
                str(f"{gflop_per_watt:.1f}" if gflop_per_watt is not None else "None"),
                total_loss_str + " (" + percent_loss_str + ")",
                style="white"  # Apply 'white' style to the entire row
            )

            # Set the width of each column
            num_columns = len(total_table.columns)
            column_width = int(100 / num_columns)

            for column in total_table.columns:
                column.width = column_width

            self.layout["lower"].update(Panel(Align(total_table, align="center"), title="Power and Performance"))

    def update_progress_bar(self, timestamp):
        self.progress.update(self.progress_task, description=f"{timestamp}", advance=timestamp, transient=True)
        self.layout["progress"].update(self.progress.get_renderable())

    def update_full_layout(self, data: TickData, time_delta=1, timestep_start=0):
        if self.debug:
            return
        uncertainties = self.engine.power_manager.uncertainties

        # if data.current_time % self.config['UI_UPDATE_FREQ'] == 0:
        if self.engine.cooling_model:
            self.update_powertemp_array(
                data.power_df, data.fmu_outputs, data.p_flops, data.g_flops_w, data.system_util,
                uncertainties=uncertainties,
            )
            self.update_pressflow_array(data.fmu_outputs)

        self.update_scheduled_jobs(data.running + data.queue)

        self.update_status(
            data.current_timestep,
            len(data.running),
            len(data.queue),
            data.num_active_nodes,
            data.num_free_nodes,
            data.down_nodes,
            data.avg_net_util,
            data.slowdown_per_job,
            data.time_delta,
            timestep_start=timestep_start
        )

        self.update_power_array(
            data.power_df, data.p_flops, data.g_flops_w,
            data.system_util, uncertainties=uncertainties,
        )

    def run(self, jobs, timestep_start, timestep_end, time_delta):
        """ Runs the UI, blocking until the simulation is complete """
        if not self.debug and not self.noui:
            context = Live(self.layout, auto_refresh=True, refresh_per_second=3)
        else:
            context = nullcontext()
        try:
            with context:
                # last_i = 0
                for i, data in enumerate(self.engine.run_simulation(jobs,
                                                                    timestep_start,
                                                                    timestep_end,
                                                                    time_delta,
                                                                    autoshutdown=True)):
                    if data and (not self.debug and not self.noui):
                        self.update_full_layout(data, time_delta, timestep_start=timestep_start)
                        # self.update_progress_bar(i-last_i)
                        # last_i=i
                    if not self.debug and not self.noui:
                        self.update_progress_bar(1)
        finally:
            os.system("stty sane")
