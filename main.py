import os
import ray
import logging
import flwr as fl
import pickle
from pathlib import Path

# setting environment variables for Ray configuration
os.environ["RAY_memory_usage_threshold"] = "0.99"
# os.environ['RAY_memory_monitor_refresh_ms'] = '0'

# initialize Ray with a higher logging level
ray.init(logging_level=logging.WARNING)
logging.basicConfig(level=logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# ignore all warnings
import warnings
warnings.filterwarnings("ignore")

# config files
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# self-defined modules
from dataset import prepare_dataset, prepare_dataloaders
from client import generate_client_fn
from server import (
    get_on_fit_config,
    SaveModelStrategy,
    fit_metrics_aggregation,
    evaluate_metrics_aggregation,
)
from model import load_model
from utils import set_seed


# load the config in config/base.yaml
@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # reproducibility
    set_seed(cfg.seed)

    # parse config and get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir

    # prepare datasets and dataloaders
    # raw normalized datasets
    raw_datasets = prepare_dataset(
        feature_type="raw_normalized",
        class_type="binary",
        overlap_percent=cfg.overlap_percent,
    )
    raw_trainloaders, raw_valloaders, raw_testloaders = prepare_dataloaders(
        raw_datasets, cfg.batch_size, cfg.test_ratio, cfg.seed
    )

    # grid (differential entropy) datasets
    grid_datasets = prepare_dataset(
        feature_type="de_grid",
        class_type="binary",
        overlap_percent=cfg.overlap_percent,
    )
    grid_trainloaders, grid_valloaders, grid_testloaders = prepare_dataloaders(
        grid_datasets, cfg.batch_size, cfg.test_ratio, cfg.seed
    )

    # combine raw and grid datasets
    trainloaders = list(zip(raw_trainloaders, grid_trainloaders))
    valloaders = list(zip(raw_valloaders, grid_valloaders))
    testloaders = list(zip(raw_testloaders, grid_testloaders))

    # define the clients
    server_model = load_model(cfg.model_type, cfg.seed)
    client_fn = generate_client_fn(
        trainloaders, valloaders, testloaders, server_model, cfg.seed
    )

    # define aggregation strategy
    num_clients = 3
    strategy = SaveModelStrategy(
        # save model
        base_model=server_model,
        save_path=save_path,
        max_rounds=cfg.num_rounds,
        # federated learning strategy
        fraction_fit=1,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
        min_fit_clients=num_clients,  # number of clients to sample for fit()
        fraction_evaluate=1,  # similar to fraction_fit, we don't need to use this argument.
        min_evaluate_clients=num_clients,  # number of clients to sample for evaluate()
        min_available_clients=num_clients,  # total clients in the simulation
        on_fit_config_fn=get_on_fit_config(
            cfg.config_fit
        ),  # a function to execute to obtain the configuration to send to the clients during fit()
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        # evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),  # a function to run on the server side to evaluate the global model.
    )

    # start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # a function that spawns a particular client
        num_clients=num_clients,  # total number of clients
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds
        ),  # minimal config for the server loop telling the number of rounds in FL
        strategy=strategy,  # our strategy of choice
        client_resources={"num_cpus": cfg.num_cpus, "num_gpus": cfg.num_gpus},
    )

    # save training history into results
    results_path = Path(save_path) / "results.pkl"
    results = {"history": history}

    # save the results as a python pickle
    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
