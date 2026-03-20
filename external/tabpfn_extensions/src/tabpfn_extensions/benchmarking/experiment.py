#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import matplotlib.pyplot as plt


class Experiment:
    """
    Base class for experiments.
    Experiments should be reproducible, i.e. the settings should give all the information
        needed to run the experiment.
    Experiments should be deterministic, i.e. the same settings should always give the same results.
    """

    name = "Experiment"

    def __init__(self, task_type, **kwargs):
        self.task_type = task_type
        self.settings = kwargs
        self.results = None

    def run(self, tabpfn, **kwargs):
        """
        Runs the experiment.

        Should set self.results
        """
        pass

    def plot(self, ax=None, **kwargs) -> dict:
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        self._plot(ax, **kwargs)

    def _plot(self, ax):
        """
        the idea behind _plot is that it allows to make nice multipanel plots by passing ax to the plotting code here.
        then you define a plot mosaic outside experiments and just tell the experiment to draw into one of its axis
        :param ax:
        :return:
        """
        pass
