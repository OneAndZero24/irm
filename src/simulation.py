import logging

from omegaconf import DictConfig
from hydra.utils import instantiate

from data import DataLoader
from model import IRModel
from solver import SDESolver
from visualizations import plot_sim


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

    log.info(f"{model.theta}, {model.sigma}, {model.alpha}, {model.r0}")

    log.info("Initializing solver.")
    solver = instantiate(config.solver)(
        t_start=t_start,
        t_stop=t_stop,
        a=model.a, 
        b=model.b, 
        Y0=model.Y0()
    )

    log.info(solver.Y0)

    log.info("Running simulation.")
    Ys = solver.run()

    plot_sim(Ys, rates, data_loader.date_index, config.sim.save_plots)
