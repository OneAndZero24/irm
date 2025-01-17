import logging

import pyrootutils
from omegaconf import DictConfig
import hydra
from hydra.utils import call

from simulation import simulation

@hydra.main(version_base = None, config_path = "../config", config_name = "config")
def main(config: DictConfig):
    simulation(config)

if __name__ == "__main__":
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    pyrootutils.setup_root(
        search_from=__file__,
        indicator="requirements.txt",
        project_root_env_var=True,
        dotenv=True,
        pythonpath=True,
        cwd=True,
    )
    main()