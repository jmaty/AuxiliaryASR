import logging
import os
import os.path as osp
import shutil
from logging import StreamHandler
import argparse

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from meldataset import build_dataloader
from models import build_model
from optimizers import build_optimizer
from trainer import Trainer
from utils import get_data_path_list, build_criterion, plot_image

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description="Auxiliary ASR training")
    parser.add_argument("--num_workers", type=int, default=1, help="number of workers")
    parser.add_argument("config_path", type=str, help="path to config")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_path, encoding="utf-8"))
    log_dir = config["log_dir"]
    if not osp.exists(log_dir):
        os.mkdir(log_dir)
    shutil.copy(args.config_path, osp.join(log_dir, osp.basename(args.config_path)))

    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, "train.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(levelname)s:%(asctime)s: %(message)s"))
    logger.addHandler(file_handler)

    batch_size = config.get("batch_size", 10)
    device = config.get("device", "cpu")
    epochs = config.get("epochs", 1000)
    save_freq = config.get("save_freq", 20)
    train_path = config.get("train_data", None)
    val_path = config.get("val_data", None)
    show_progress = config.get("show_progress", False)

    train_list, val_list = get_data_path_list(train_path, val_path)

    train_dataloader = build_dataloader(
        train_list,
        batch_size=batch_size,
        num_workers=args.num_workers,
        dataset_config=config.get("dataset_params", {}),
        device=device,
    )
    val_dataloader = build_dataloader(
        val_list,
        batch_size=batch_size,
        validation=True,
        num_workers=args.num_workers,
        device=device,
        dataset_config=config.get("dataset_params", {}),
    )

    model = build_model(model_params=config["model_params"] or {})

    scheduler_params = {
        "max_lr": float(config["optimizer_params"].get("lr", 5e-4)),
        "pct_start": float(config["optimizer_params"].get("pct_start", 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }

    model.to(device)
    optimizer, scheduler = build_optimizer(
        {
            "params": model.parameters(),
            "optimizer_params": {},
            "scheduler_params": scheduler_params,
        }
    )

    # blank_index = train_dataloader.dataset.text_cleaner.word_index_dictionary[" "] # get blank index
    _, blank_index = train_dataloader.dataset.text_cleaner.blank  # get blank index

    criterion = build_criterion(
        critic_params={
            "ctc": {"blank": blank_index},
        }
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        logger=logger,
    )

    if config.get("pretrained_model", "") != "":
        trainer.load_checkpoint(
            config["pretrained_model"],
            load_only_params=config.get("load_only_params", True),
        )

    for epoch in range(1, epochs + 1):
        train_results = trainer.train_epoch(show_progress=show_progress)
        eval_results = trainer.eval_epoch(show_progress=show_progress)
        results = train_results.copy()
        results.update(eval_results)
        logger.info("--- epoch %d ---", epoch)
        for key, value in results.items():
            if isinstance(value, float):
                logger.info("%-15s: %.4f", key, value)
                writer.add_scalar(key, value, epoch)
            else:
                for v in value:
                    writer.add_figure("eval_attn", plot_image(v), epoch)
        if (epoch % save_freq) == 0:
            trainer.save_checkpoint(osp.join(log_dir, f"epoch_{epoch:05d}.pth"))

    return 0


if __name__ == "__main__":
    main()
