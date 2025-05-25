import argparse
import json
import os
import random
import time

from sympy.matrices.expressions.kronecker import validate

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import wandb
import yaml



from src.nn.loss import (
    BinaryDiceLoss,
    CrossEntropyLoss,
    TrainEgoPathRegressionLoss,
)
from src.nn.model import ClassificationNet, RegressionNet, SegmentationNet, RegressionNetWithLSTM
from src.utils.common import set_seeds, set_worker_seeds, simple_logger, split_dataset
from src.utils.dataset import PathsDataset
from src.utils.evaluate import IoUEvaluator
from src.utils.trainer import train

torch.use_deterministic_algorithms(True)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Ego-Path Detection Training Script")
    parser.add_argument(
        "method",
        type=str,
        choices=["regression", "classification", "segmentation"],
        help="Method to use for the prediction head ('regression', 'classification' or 'segmentation').",
    )
    parser.add_argument(
        "backbone",
        type=str,
        choices=[f"resnet{x}" for x in [18, 34, 50]]
        + [f"efficientnet-b{x}" for x in [0, 1, 2, 3]],
        help="Backbone to use (e.g., 'resnet18', 'efficientnet-b3').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "mps"]
        + [f"cuda:{x}" for x in range(torch.cuda.device_count())],
        help="Device to use ('cpu', 'cuda', 'cuda:x' or 'mps').",
    )
    return parser.parse_args()


def main(args):
    method = args.method
    device = torch.device(args.device)
    logger = simple_logger(__name__, "info")
    base_path = os.path.dirname(__file__)

    with open(os.path.join(base_path, "configs", "global.yaml"), encoding= "utf-8") as f:
        global_config = yaml.safe_load(f)
    with open(os.path.join(base_path, "configs", f"{method}.yaml"), encoding= "utf-8") as f:
        method_config = yaml.safe_load(f)
    config = {
        **global_config,
        **method_config,
        "method": method,
        "backbone": args.backbone,
    }

    set_seeds(config["seed"])  # set random state
    with open(config["annotations_path"]) as json_file:
        indices = list(range(len(json.load(json_file).keys())))
    random.shuffle(indices)
    proportions = (config["train_prop"], config["val_prop"], config["test_prop"])
    train_indices, val_indices, test_indices = split_dataset(indices, proportions)
    set_seeds(config["seed"])  # reset random state

    train_dataset = PathsDataset(
        imgs_path=config["images_path"],
        annotations_path=config["annotations_path"],
        indices=train_indices,
        config=config,
        method=method,
        img_aug=True,
        to_tensor=True,
    )
    val_dataset = (
        PathsDataset(
            imgs_path=config["images_path"],
            annotations_path=config["annotations_path"],
            indices=val_indices,
            config=config,
            method=method,
            img_aug=True,
            to_tensor=True,
        )
        if len(val_indices) > 0
        else None
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["workers"],
        pin_memory=True,
        worker_init_fn=set_worker_seeds,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    val_loader = (
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            num_workers=config["workers"],
            pin_memory=True,
            worker_init_fn=set_worker_seeds,
            generator=torch.Generator().manual_seed(config["seed"]),
        )
        if val_dataset is not None
        else None
    )

    if method == "regression":
        if config.get("use_lstm", False):  # 检查是否使用 LSTM
            print(config["use_lstm"])
            model = RegressionNetWithLSTM(
                backbone=config["backbone"],
                input_shape=tuple(config["input_shape"]),
                anchors=config["anchors"],
                pool_channels=config["pool_channels"],
                fc_hidden_size=config["fc_hidden_size"],
                lstm_hidden_size=config["lstm_hidden_size"],
                lstm_num_layers=config["lstm_num_layers"],
                pretrained=config["pretrained"],
            ).to(device)
        else:
            model = RegressionNet(
                backbone=config["backbone"],
                input_shape=tuple(config["input_shape"]),
                anchors=config["anchors"],
                pool_channels=config["pool_channels"],
                fc_hidden_size=config["fc_hidden_size"],
                pretrained=config["pretrained"],
            ).to(device)
    elif method == "classification":
        model = ClassificationNet(
            backbone=config["backbone"],
            input_shape=tuple(config["input_shape"]),
            anchors=config["anchors"],
            classes=config["classes"],
            pool_channels=config["pool_channels"],
            fc_hidden_size=config["fc_hidden_size"],
            pretrained=config["pretrained"],
        ).to(device)
    elif method == "segmentation":
        model = SegmentationNet(
            backbone=config["backbone"],
            decoder_channels=tuple(config["decoder_channels"]),
            pretrained=config["pretrained"],
        ).to(device)
    else:
        raise ValueError


    # try:
    #     if torch.__version__ >= "2.0.0":  # 确保 PyTorch 版本支持 compile
    #         model = torch.compile(
    #             model,
    #             mode='reduce-overhead',  # 使用较为保守的优化模式
    #             fullgraph=False,  # 对于 LSTM 模型，设置为 False 可能更稳定
    #             dynamic=True,  # 支持动态输入
    #         )
    #         print("Model successfully compiled")
    # except Exception as e:
    #     print(f"torch.compile failed: {e}. Running model without compilation.")

    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = torch.nn.DataParallel(model)

    wandb.init(
        project="train-ego-path-detection",
        config=config,
        dir=os.path.join(base_path),
    )
    save_path = os.path.join(base_path, "weights", wandb.run.name)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    if method == "regression":
        criterion = TrainEgoPathRegressionLoss(
            ylimit_loss_weight=config["ylimit_loss_weight"],
            perspective_weight_limit=train_dataset.get_perspective_weight_limit(
                percentile=config["perspective_weight_limit_percentile"],
                logger=logger,
            )
            if config["perspective_weight_limit_percentile"] is not None
            else None,
        )
        if config["perspective_weight_limit_percentile"] is not None:
            set_seeds(config["seed"])  # reset random state
    elif method == "classification":
        criterion = CrossEntropyLoss()
    elif method == "segmentation":
        criterion = BinaryDiceLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = (
        torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=config["learning_rate"],
            total_steps=config["epochs"],
            pct_start=0.1,
            verbose=False,
        )
        if config["scheduler"] == "one_cycle"
        else None
    )

    logger.info(f"\nTraining {method} model for {config['epochs']} epochs...")
    train(
        epochs=config["epochs"],
        dataloaders=(train_loader, val_loader),
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        save_path=save_path,
        device=device,
        logger=logger,
        val_iterations=config["val_iterations"],
    )
    # if val_loader is not None:
    #     val_loss, val_accuracy = validate(val_loader, model, criterion, device, logger)
    #     logger.info(f"Validation loss: {val_loss}, validation accuracy: {val_accuracy:.2f}%")
    #     wandb.log({"final_val_accuracy": val_accuracy})

    if len(test_indices) > 0:
        logger.info("\nEvaluating on test set...")
        test_dataset = PathsDataset(
            imgs_path=config["images_path"],
            annotations_path=config["annotations_path"],
            indices=test_indices,
            config=config,
            method=method,  # 使用当前的method而不是固定为"segmentation"
            to_tensor=False  # 这里改为False，保证img为PIL.Image
        )
        iou_evaluator = IoUEvaluator(
            dataset=test_dataset,
            model_path=save_path,
            runtime="pytorch",
            device=device,
        )
        start_time = time.time()
        test_iou = iou_evaluator.evaluate()
        end_time = time.time()
        fps = len(test_dataset) / (end_time - start_time)
        logger.info(f"Test IoU: {test_iou:.5f}")
        wandb.log({"test_iou": test_iou})
        logger.info(f"Test FPS: {fps:.2f}")
        # wandb.log({"test_iou": test_iou, "test_fps": fps})



if __name__ == "__main__":
    wandb.login(key='fb8688472c81559eaa9821013722cf47aac8ceb1')
    args = parse_arguments()
    main(args)
