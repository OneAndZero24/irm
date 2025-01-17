import logging
import numpy as np
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

    date = np.datetime64(datetime.strptime(config.sim.date, "%m/%d/%Y").date())
    rates = data_loader.get_date(date)
    t_start = 0
    t_stop = len(rates)-1

    log.info("Model calibration.")
    model.calibrate(rates)

    log.info("Model fit.")
    run_sim(config, t_start, t_stop, rates, data_loader.maturity_index, model, model.Y0())