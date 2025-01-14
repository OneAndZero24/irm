import pyrootutils
from omegaconf import DictConfig
import hydra
from hydra.utils import call

from src.simulation import simulation

@hydra.main(version_base = None, config_path = "../config", config_name = "config")
def main(config: DictConfig):
    call(simulation, config)

if __name__ == "__main__":
    pyrootutils.setup_root(
        search_from=__file__,
        indicator="requirements.txt",
        project_root_env_var=True,
        dotenv=True,
        pythonpath=True,
        cwd=True,
    )
    main()