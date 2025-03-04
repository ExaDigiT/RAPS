import pandas as pd
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.progress import Progress,TextColumn,BarColumn,TaskProgressColumn,TimeRemainingColumn, track, TimeElapsedColumn, MofNCompleteColumn

from .utils import summarize_ranges, convert_seconds
from .constants import ELLIPSES
from .engine import TickData, Engine


class LayoutManager:
    def __init__(self, layout_type, engine: Engine, total_timesteps=0, debug=None, **config):
        self.engine = engine
        self.config = config
        self.console = Console()
        self.layout = Layout()
        self.hascooling = layout_type == "layout2"
        self.debug = debug
        self.setup_layout(layout_type)
        self.power_df_header = self.config['POWER_DF_HEADER']
        self.racks_per_cdu = self.config['RACKS_PER_CDU']
        self.power_column = self.power_df_header[self.racks_per_cdu + 1]
        self.loss_column = self.power_df_header[-1]
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
        self.progress_task = self.progress.add_task("Progress",total=total_timesteps, name="Progress")

    def setup_layout(self, layout_type):
        self.layout.split_column(Layout(name="main"),Layout(name="progress",size=1))
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
            self.layout["main"]["right"].split_column(Layout(name="scheduled", ratio=8), Layout(name="status", ratio=2))

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
        return total_power_mw, total_loss_mw, f"{total_loss_mw / total_power_mw * 100:.2f}%", total_power_kw, total_loss_kw

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
        # Define columns with header styles
        columns = ["JOBID", "WALL TIME", "NAME", "ACCOUNT", "ST", "NODES", "NODE SEGMENTS"]
        if show_nodes:
            columns.append("NODELIST")
        columns.append("TIME")

        # Create table with bold magenta headers
        table = Table(title="Job Queue", header_style="bold magenta", expand=True)
        for col in columns:
            table.add_column(col, justify="center")

        # Add data rows with white values
        for job in jobs:
            node_segments = summarize_ranges(job.scheduled_nodes)
            if show_nodes:
                if len(node_segments) > 4:
                    nodes_display = ", ".join(node_segments[:2] + [ELLIPSES] + node_segments[-2:])
                else:
                    nodes_display = ", ".join(node_segments)
            else:
                nodes_display = str(len(node_segments))

            row = [
                str(job.id).zfill(5),
                convert_seconds(job.wall_time),
                str(job.name),
                str(job.account),
                job.state.value,
                str(job.nodes_required),
                nodes_display,
                convert_seconds(job.running_time)
            ]
            # Add the row with the 'white' style applied to the whole row
            table.add_row(*row, style="white")

        # Update the layout
        self.layout["scheduled"].update(Panel(Align(table, align="center")))

    def update_status(self, time, nrun, nqueue, active_nodes, free_nodes, down_nodes):
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
        columns = ["Time", "Jobs Running", "Jobs Queued", "Active Nodes", "Free Nodes", "Down Nodes"]
        table = Table(header_style="bold magenta", expand=True)
        for col in columns:
            table.add_column(col, justify="center")

        # Add data row with white values
        row = [
            convert_seconds(time),
            str(nrun),
            str(nqueue),
            str(active_nodes),
            str(free_nodes),
            str(len(down_nodes))
        ]
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


    def update_powertemp_array(self, power_df, cooling_outputs, pflops, gflop_per_watt, system_util, uncertainties=False):
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
        #power_columns = POWER_DF_HEADER[0:RACKS_PER_CDU + 2] + [POWER_DF_HEADER[-1]]  # "CDU", "Rack 1", "Rack 2", "Rack 3", "Sum", "Loss"
        power_columns = self.power_df_header[0:self.racks_per_cdu + 2] + [self.power_df_header[-1]]  # "CDU", "Rack 1", "Rack 2", "Rack 3", "Sum", "Loss"
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
            power_df = power_df[power_columns].astype(int)

        # Populate the table with data from the DataFrame, applying the data styles
        for power_row, cooling_row in zip(power_df.iterrows(), datacenter_df.iterrows()):
            power_values = [
                f"[{data_styles[i]}]{power_row[1][col]}[/]" for i, col in enumerate(power_columns)
            ]
            cooling_values = [
                f"[{data_styles[i + len(power_columns)]}]{cooling_row[1][key]:.1f}[/]" for i, key in enumerate(cooling_keys)
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
            total_loss_str + " (" + percent_loss_str+ ")",
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
            self.layout["upper"].update(Panel(Align(table, align="center"),title=self.engine.config["system_name"].capitalize()))

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
                str(f"{pflops:.2f}"),
                str(f"{gflop_per_watt:.1f}"),
                total_loss_str + " (" + percent_loss_str+ ")",
                style="white"  # Apply 'white' style to the entire row
            )

            # Set the width of each column
            num_columns = len(total_table.columns)
            column_width = int(100 / num_columns)

            for column in total_table.columns:
                column.width = column_width

            self.layout["lower"].update(Panel(Align(total_table, align="center"), title="Power and Performance"))

    def update_progress(self, timestamp):
        self.progress.update(self.progress_task, description=f"{timestamp}",advance=timestamp,transient=True)
        self.layout["progress"].update(self.progress.get_renderable())

    def update(self, data: TickData):
        uncertainties = self.engine.power_manager.uncertainties

        if data.current_time % self.config['UI_UPDATE_FREQ'] == 0:
            if self.engine.cooling_model:
                self.update_powertemp_array(
                    data.power_df, data.fmu_outputs, data.p_flops, data.g_flops_w, data.system_util,
                    uncertainties=uncertainties,
                )
                self.update_pressflow_array(data.fmu_outputs)

            self.update_scheduled_jobs(data.running + data.queue)
            self.update_status(
                data.current_time, len(data.running), len(data.queue), data.num_active_nodes,
                data.num_free_nodes, data.down_nodes,
            )
            self.update_power_array(
                data.power_df, data.p_flops, data.g_flops_w,
                data.system_util, uncertainties=uncertainties,
            )
            if False:
                self.render()
        self.update_progress(1)


    def render(self):
        if not self.debug:
            self.console.clear()
            self.console.print(self.layout)

    def run(self, jobs, timestep_start, timestep_end):
        """ Runs the UI, blocking until the simulation is complete """
        with Live(self.layout, refresh_per_second=5):
            for data in self.engine.run_simulation(jobs, timestep_start, timestep_end):
                self.update(data)

    def run_stepwise(self, jobs, timestep_start, timestep_end):
        """ Prepares the UI and returns a generator for the simulation """
        return self.engine.run_simulation(jobs, timestep_start, timestep_end)
