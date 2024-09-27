"""
This module provides functionality for creating and saving various types of plots using Matplotlib.

The module defines a `BasePlotter` class for setting up plots and saving them, and a `Plotter` class
that extends `BasePlotter` to include methods for plotting histories, histograms, and comparisons.

Classes
-------
BasePlotter
    A base class for setting up and saving plots.
Plotter
    A class for creating and saving specific types of plots, such as histories,
    histograms, and comparisons.
"""

import matplotlib.pyplot as plt
import numpy as np
from uncertainties import unumpy

class BasePlotter:
    """
    A base class for setting up and saving plots.

    Attributes
    ----------
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    title : str
        The title of the plot.
    """
    def __init__(self, xlabel, ylabel, title, uncertainties=False):
        """
        Constructs all the necessary attributes for the BasePlotter object.

        Parameters
        ----------
        xlabel : str
            The label for the x-axis.
        ylabel : str
            The label for the y-axis.
        title : str
            The title of the plot.
        """
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.uncertainties = uncertainties

    def setup_plot(self, figsize=(10, 5)):
        """
        Sets up the plot with the given figure size, labels, title, and grid.

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure (default is (10, 5)).
        """
        plt.figure(figsize=figsize)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.grid(True)

    def save_and_close_plot(self, save_path):
        """
        Saves the plot to the specified path and closes the plot.

        Parameters
        ----------
        save_path : str
            The path to save the plot.
        """
        plt.savefig(save_path)
        plt.close()

class Plotter(BasePlotter):
    """
    A class for creating and saving specific types of plots, such as histories,
    histograms, and comparisons.

    Attributes
    ----------
    save_path : str
        The path to save the plot.
    """
    def __init__(self, xlabel='', ylabel='', title='', save_path='out.svg', uncertainties=False):
        """
        Constructs all the necessary attributes for the Plotter object.

        Parameters
        ----------
        xlabel : str, optional
            The label for the x-axis (default is an empty string).
        ylabel : str, optional
            The label for the y-axis (default is an empty string).
        title : str, optional
            The title of the plot (default is an empty string).
        save_path : str, optional
            The path to save the plot (default is 'out.svg').
        uncertainties: boolean, optional
            Flag if uncertainties are enabled and ufloats are used.
        """
        super().__init__(xlabel, ylabel, title, uncertainties)
        self.save_path = save_path

    def plot_history(self, x, y):
        """
        Plots a history plot of the given x and y values and saves it.

        Parameters
        ----------
        x : list
            The x values for the plot.
        y : list
            The y values for the plot.
        """
        self.setup_plot()

        if self.uncertainties:
            nominal_curve = plt.plot(x, unumpy.nominal_values(y))
            plt.fill_between(x, unumpy.nominal_values(y)-unumpy.std_devs(y),
                             unumpy.nominal_values(y)+unumpy.std_devs(y),
                             facecolor=nominal_curve[0].get_color(),
                             edgecolor='face', alpha=0.1, linewidth=0)
        else:
            plt.plot(x, y)
        self.save_and_close_plot(self.save_path)

    def plot_histogram(self, data, bins=50):
        """
        Plots a histogram of the given data and saves it.

        Parameters
        ----------
        data : list
            The data to plot in the histogram.
        bins : int, optional
            The number of bins in the histogram (default is 50).
        """
        self.setup_plot()
        plt.hist(data, bins=bins)
        self.save_and_close_plot(self.save_path)

    def plot_compare(self, x, y):
        """
        Plots a comparison plot of the given x and y values and saves it.

        Parameters
        ----------
        x : list
            The x values for the plot.
        y : list
            The y values for the plot.
        """
        self.setup_plot()
        plt.plot(x, y)
        self.save_and_close_plot(self.save_path)


def plot_nodes_histogram(nr_list, num_bins=25):
    print("plotting nodes required histogram...")

    # Create logarithmically spaced bins
    bins = np.logspace(np.log2(min(nr_list)), np.log2(max(nr_list)), num=num_bins, base=2)

    # Set up the figure
    plt.clf()
    plt.figure(figsize=(10, 3))

    # Create the histogram
    plt.hist(nr_list, bins=bins, edgecolor='black')

    # Add a title and labels
    plt.xlabel('Number of Nodes')
    plt.ylabel('Frequency')

    # Set the axes to logarithmic scale
    plt.xscale('log', base=2)
    plt.yscale('log')

    # Customize the x-ticks: Choose positions like 1, 8, 64, etc.
    ticks = [2**i for i in range(0, 14)]
    plt.xticks(ticks, labels=[str(tick) for tick in ticks])

    # Set min-max axis bounds
    plt.xlim(1, max(nr_list))

    # Save the histogram to a file
    plt.savefig('histogram.png', dpi=300, bbox_inches='tight')


def plot_submit_times(submit_times, nr_list):
    """Plot number of nodes over time"""

    print("plotting submit times...")

    # Determine the time scale
    max_time = max(submit_times)

    if max_time >= 3600 * 24 * 7:  # If more than a week convert time to days
        submit_times = [time / (3600 * 24) for time in submit_times]
        time_label = 'Submit Time (days)'
    elif max_time >= 3600 * 24:  # If more than 24 hours convert time to hours
        submit_times = [time / 3600 for time in submit_times]
        time_label = 'Submit Time (hours)'
    else:
        time_label = 'Submit Time (s)'

    plt.clf()
    plt.figure(figsize=(10, 2))

    # Create a bar chart
    bar_width = (max(submit_times) - min(submit_times)) / len(submit_times) * 0.8
    plt.bar(submit_times, nr_list, width=bar_width, color='blue', edgecolor='black', alpha=0.7)

    # Add labels and title
    plt.xlabel(time_label)
    plt.ylabel('Number of Nodes')

    # Set min-max axis bounds
    plt.xlim(1, max(submit_times))

    # Set the y-axis to logarithmic scale with base 2
    plt.yscale('log', base=2)
    y_ticks = [2**i for i in range(0, 14)]
    plt.yticks(y_ticks, labels=[str(tick) for tick in y_ticks])

    # Save the plot to a file
    plt.savefig('submit_times.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    plotter = Plotter()
    #plotter.plot_history([1, 2, 3, 4])
