import numpy as np
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from .utils import summarize_ranges, convert_seconds
from .constants import ELLIPSES
from .config import load_config_variables

load_config_variables(['POWER_CDUS', 'POWER_DF_HEADER', 'FMU_COLUMN_MAPPING', 'RACKS_PER_CDU'], globals())

class LayoutManager:
    def __init__(self, layout_type, debug=False):
        self.console = Console()
        self.layout = Layout()
        self.hascooling = layout_type == "layout2"
        self.debug = debug
        self.setup_layout(layout_type)

    def setup_layout(self, layout_type):
        if layout_type == "layout2":
            self.layout.split_row(Layout(name="left", ratio=3), Layout(name="right", ratio=2))
            self.layout["left"].split_column(
                Layout(name="pressflow", ratio=6),
                Layout(name="powertemp", ratio=11),
                Layout(name="totpower", ratio=3),
            )
            self.layout["right"].split(Layout(name="scheduled", ratio=17), Layout(name="status", ratio=3))
        else:
            self.layout.split_row(Layout(name="left", ratio=1), Layout(name="right", ratio=1))
            self.layout["left"].split_column(Layout(name="upper", ratio=8), Layout(name="lower", ratio=2))
            self.layout["right"].split_column(Layout(name="scheduled", ratio=8), Layout(name="status", ratio=2))

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

    def calculate_totals(self, df, power_column=POWER_DF_HEADER[RACKS_PER_CDU + 1], loss_column=POWER_DF_HEADER[-1]): # 'Sum' and 'Loss' columns
        total_power_kw = df[power_column].sum() + (POWER_CDUS / 1000.0)
        total_power_mw = total_power_kw / 1000.0
        total_loss_kw = df[loss_column].sum()
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
        columns = ["JOBID", "WALL TIME", "NAME", "ST", "NODES", "NODE SEGMENTS"]
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
                job.name,
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

    def update_pressflow_array(self, cooling_df):
        columns = ["Output", "Average Value"]

        # List of keys to include in the table
        relevant_keys = [
            "W_CDUP_Out", "ps_pri_Out", "pr_pri_Out",
            "Q_pri_Out", "Q_sec_Out", "ps_sec_Out", "pr_sec_Out"
        ]

        # Dynamically build the data list using FMU_COLUMN_MAPPING
        data = []
        for key in relevant_keys:
            if key in cooling_df and key in FMU_COLUMN_MAPPING:
                label = FMU_COLUMN_MAPPING[key]
                average_value = round(cooling_df[key].mean(), 1)
                data.append((label, average_value))

        # Create table with white headers
        table = self.create_table("Pressure and Flow Rates", columns, header_style="bold white")
        self.add_table_rows(table, data)
        self.layout["pressflow"].update(Panel(table))

    def update_powertemp_array(self, power_df, cooling_df, uncertainties=False):
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
        power_columns = POWER_DF_HEADER[0:RACKS_PER_CDU + 2] + [POWER_DF_HEADER[-1]]  # "CDU", "Rack 1", "Rack 2", "Rack 3", "Sum", "Loss"
        cooling_keys = ["Ts_pri_Out", "Tr_pri_Out", "Ts_sec_Out", "Tr_sec_Out"]

        # Create column headers with appropriate styles
        columns = [f"{col} (kW)" if col != "CDU" else col for col in power_columns]
        columns += [FMU_COLUMN_MAPPING[key] for key in cooling_keys]

        # Define styles for data values
        data_styles = ["bold cyan"] + ["bold green"] * (len(power_columns) - 1)
        data_styles += [
            "bold blue" if "Facility Supply" in FMU_COLUMN_MAPPING[key] else
            "bold red" if "Facility Return" in FMU_COLUMN_MAPPING[key] else
            "bold blue" if "Rack Supply" in FMU_COLUMN_MAPPING[key] else
            "bold red" for key in cooling_keys
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
        for power_row, cooling_row in zip(power_df.iterrows(), cooling_df.iterrows()):
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
        total_table.add_column("Total Power", justify="center", style="green")
        total_table.add_column("Total Loss", justify="center", style="green")
        total_table.add_column("Percent Loss", justify="center", style="green")
        total_table.add_column("PUE", justify="center", style="green")

        # Add row with white data values using the style parameter
        total_table.add_row(
            total_power_str,
            total_loss_str,
            percent_loss_str,
            f"{cooling_df.iloc[0]['PUE_Out']:.2f}",  # Assuming PUE_Out is present in cooling_df
            style="white"  # Apply white style to all elements in the row
        )

        # Set the width of each column
        num_columns = len(total_table.columns)
        column_width = int(100 / num_columns)

        for column in total_table.columns:
            column.width = column_width

        self.layout["totpower"].update(Panel(Align(total_table, align="center"), title="Power Stats"))

    def update_power_array(self, power_df, uncertainties=False):
        """
        Updates the displayed power array table with the provided data from df.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing power and loss data for racks.
        """
        # Define the specific columns to display
        display_columns = POWER_DF_HEADER[0:RACKS_PER_CDU + 2] + [POWER_DF_HEADER[-1]]  # "CDU", "Rack 1", "Rack 2", "Rack 3", "Sum", "Loss"

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
            self.layout["upper"].update(Panel(Align(table, align="center")))

            # Create Total Power table with green headers and white data
            total_table = Table(show_header=True, header_style="bold green")
            total_table.add_column("Total Power", justify="center", style="green")
            total_table.add_column("Total Loss", justify="center", style="green")
            total_table.add_column("Percent Loss", justify="center", style="green")

            # Add row with white data values
            total_table.add_row(
                total_power_str,
                total_loss_str,
                percent_loss_str,
                style="white"  # Apply 'white' style to the entire row
            )

            # Set the width of each column
            num_columns = len(total_table.columns)
            column_width = int(100 / num_columns)

            for column in total_table.columns:
                column.width = column_width

            self.layout["lower"].update(Panel(Align(total_table, align="center"), title="Power Stats"))

    def render(self):
        if not self.debug:
            self.console.clear()
            self.console.print(self.layout)
