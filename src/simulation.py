import logging
import typing
import pandas as pd
import numpy as np
import numpy.typing as npt
from datetime import datetime

from omegaconf import DictConfig
from hydra.utils import instantiate

from data import DataLoader
from model import IRModel
from solver import SDESolver
from visualizations import plot_sim
from util import working_days_between


log = logging.getLogger(__name__)

def _run_sim(config: DictConfig, t_start: float, t_stop: float, y: typing.Optional[np.float64], x: npt.NDArray[np.float64], model: IRModel, Y0: float):
    """
    Helper function
    """

    log.info("Initializing solver.")
    solver = instantiate(config.solver)(
        t_start=t_start,
        t_stop=t_stop,
        a=model.a, 
        b=model.b, 
        Y0=model.Y0()
    )

    log.info("Running simulation.")
    Ys = solver.run()
    plot_sim(Ys, y, x, config.sim.save_plots)

def simulation(config: DictConfig):
    """
    Simulation procedure.
    """

    log.info("Initializing data loader.")
    data_loader = instantiate(config.data)

    log.info("Initializing model.")
    model = instantiate(config.model)

    rates = data_loader.get_maturity(config.sim.bond)
    t_start = 0
    t_stop = len(rates)-1

    log.info("Model calibration.")
    model.calibrate(rates)

    log.info("Model fit.")
    _run_sim(config, t_start, t_stop, rates, data_loader.date_index, model, model.Y0())

    if ('forecast' in config.sim):
        start_date = pd.to_datetime(data_loader.date_index[-1]).to_pydatetime().date()
        stop_date = datetime.strptime(config.sim.forecast, "%m/%d/%Y").date()
        days = working_days_between(start_date, stop_date)+[stop_date]
        N = len(days)-1
        if N > 0:
            log.info('Forecasting.')
            _run_sim(config, t_stop, t_stop+N, None, days, model, rates[-1])