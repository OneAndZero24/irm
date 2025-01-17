import logging
from datetime import date, timedelta
import typing
import numpy as np
import numpy.typing as npt

from omegaconf import DictConfig
from hydra.utils import instantiate

from model import IRModel
from visualizations import plot_sim


log = logging.getLogger(__name__)

def working_days_between(a: date, b: date) -> list[date]:
    """
    Calculate the number of working days (Monday to Friday) between two dates.

    Args:
        a (date): The start date.
        b (date): The end date.

    Returns:
        list(date): List of working days between two dates
    """

    if b <= a:
        return []

    total_days = (b - a).days
    tmp = [ a+timedelta(days=day) for day in range(total_days) ]
    return list(filter(lambda x: x.weekday() < 5, tmp))

def run_sim(config: DictConfig, t_start: float, t_stop: float, y: typing.Optional[np.float64], x: npt.NDArray[np.float64], model: IRModel, Y0: float):
    """
    Helper function
    """

    if (config.solver == "Milstein") and (not model.differentiable()):
        raise RuntimeError("Milstein solver requires differentiable SDE!")

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