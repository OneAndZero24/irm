import logging
import pandas as pd
from datetime import datetime

from omegaconf import DictConfig
from hydra.utils import instantiate

from util import working_days_between, run_sim


log = logging.getLogger(__name__)


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
    run_sim(config, t_start, t_stop, rates, data_loader.date_index, model, model.Y0())

    if ('forecast' in config.sim):
        start_date = pd.to_datetime(data_loader.date_index[-1]).to_pydatetime().date()
        stop_date = datetime.strptime(config.sim.forecast, "%m/%d/%Y").date()
        days = working_days_between(start_date, stop_date)+[stop_date]
        N = len(days)-1
        if N > 0:
            log.info('Forecasting.')
            run_sim(config, t_stop, t_stop+N, None, days, model, rates[-1])