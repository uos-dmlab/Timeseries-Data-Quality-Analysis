import os
import torch

from torch.utils.data import DataLoader
from TIMEBAND.loss import TIMEBANDLoss
from TIMEBAND.model import TIMEBANDModel
from TIMEBAND.metric import TIMEBANDMetric
from TIMEBAND.dataset import TIMEBANDDataset
from TIMEBAND.trainer import TIMEBANDTrainer
from TIMEBAND.runner import TIMEBANDRunner
from TIMEBAND.dashboard import TIMEBANDDashboard

logger = None


class TIMEBANDCore:
    """
    TIMEBANDBand : Timeseries Analysis using GAN Band

    The Model for Detecting anomalies / Imputating missing value in timeseries data

    """

    def __init__(self, config: dict) -> None:

        # Set Config
        self.set_config(config=config)

        # Dataset & Model Settings
        self.dataset = TIMEBANDDataset(self.dataset_cfg)
        self.models = TIMEBANDModel(self.models_cfg)

        # Losses and Metric Settings
        self.metric = TIMEBANDMetric(self.metric_cfg)
        self.losses = TIMEBANDLoss(self.losses_cfg)

        # Visualize Settings
        self.dashboard = TIMEBANDDashboard(self.dashboard_cfg, self.dataset)

    def set_config(self, config: dict) -> None:
        """
        Setting configuration

        """
        # Set Logger
        global logger
        logger = config["core"]["logger"]
        config["core"]["targets_dims"] = len(config["dataset"]["targets"])

        # Configuration Categories
        self.__dict__ = {**config["core"], **self.__dict__}
        self.dataset_cfg = {**config["core"], **config["dataset"]}
        self.models_cfg = {**config["core"], **config["models"]}
        self.metric_cfg = {**config["core"], **config["dataset"]}
        self.losses_cfg = {**config["core"], **config["losses"]}
        self.trainer_cfg = {**config["core"], **config["trainer"]}
        self.dashboard_cfg = {**config["core"], **config["dashboard"]}
        self.runner_cfg = {**config["core"], **config["trainer"]}

        self.output_path = os.path.join(self.outputs, self.data_name, self.TAG)

    def train(self) -> None:
        # Init the models
        self.models.initiate(dims=self.dataset.dims)

        self.trainer = TIMEBANDTrainer(
            self.trainer_cfg,
            self.dataset,
            self.models,
            self.metric,
            self.losses,
            self.dashboard,
        )

        for k in range(self.dataset.sliding_step + 1):
            logger.info(f"Train ({k + 1}/{self.dataset.sliding_step + 1})")

            if self.pretrain:
                self.models.load("BEST")

            # Dataset
            trainset, validset = self.dataset.prepare_dataset(k + 1)
            trainset, validset = self.loader(trainset), self.loader(validset)

            # Model
            self.trainer.train(trainset, validset)

            logger.info(f"Done ({k + 1}/{self.dataset.sliding_step + 1}) ")

    def run(self) -> None:
        # Init the models
        self.models.initiate(dims=self.dataset.dims)

        self.runner = TIMEBANDRunner(
            self.runner_cfg,
            self.dataset,
            self.models,
            self.losses,
            self.metric,
            self.dashboard,
        )

        if self.pretrain:
            self.models.load("BEST")

        dataset = self.dataset.prepare_testset()
        dataset = self.loader(dataset)

        # target_path = os.path.join(self.output_path, "target.csv")
        target_path = os.path.join(self.output_path, "visualize.csv")
        target_output = self.runner.run(dataset)
        target_output.to_csv(target_path)

        logger.info(f"{target_path} is saved")

    def loader(self, dataset: TIMEBANDDataset) -> DataLoader:
        dataloader = DataLoader(dataset, self.batch_size, num_workers=self.workers)
        return dataloader
