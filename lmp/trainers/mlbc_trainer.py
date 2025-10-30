from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv

import pytorch_lightning as pl
import torchmetrics
import torchmetrics.classification

import lmp
import lmp.nn as lnn
from lmp.util.typing import *

from .base import BaseTrainer


@lmp.register("multilabel-binaryc-trainer")
class MultiLabelBinaryClassificationTrainer(BaseTrainer):
    @dataclass
    class Config(BaseTrainer.Config):
        model_type: str = ""
        model: dict = field(default_factory=dict)

        n_labels: int = 10

        visualize: bool = True
        visualize_every_val: int = 2

        label_weights: Optional[list] = None

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.model = lmp.find(self.cfg.model_type)(self.cfg.model)
        lmp.info(self.model)

        label_weights = self.cfg.label_weights
        if label_weights is not None:
            label_weights = torch.tensor(label_weights, device=self.device, dtype=self.dtype)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=label_weights)

        num_labels = self.cfg.n_labels

        self.train_acc = torchmetrics.classification.Accuracy(task="multilabel", num_labels=num_labels)
        self.train_roc = torchmetrics.classification.AUROC(task="multilabel", num_labels=num_labels)

        self.val_acc = torchmetrics.classification.Accuracy(task="multilabel", num_labels=num_labels)
        self.val_roc = torchmetrics.classification.AUROC(task="multilabel", num_labels=num_labels)

        self.test_acc = torchmetrics.classification.Accuracy(task="multilabel", num_labels=num_labels)
        self.test_roc = torchmetrics.classification.AUROC(task="multilabel", num_labels=num_labels)

    def forward(self, x):
        pred, out = self.model(x)
        return pred, out

    def loss(self, input_dict, phase="train"):
        pred, other = self.forward(input_dict["input"])
        target = input_dict["target"].float()

        out = {
            "pred": pred,
            "target": target,
            **other
        }

        loss_terms = {}

        loss_prefix = f"loss_"

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value

        # Cross entropy
        set_loss("bce", self.cfg.n_labels * self.bce_loss(pred, target))

        if self.C(self.cfg.loss.lambda_minimal_filt) > 0:
            filt_minimal = optional_index_2d(out["filts"], 0, 0)

            set_loss("minimal_filt", F.l1_loss(filt_minimal, torch.zeros_like(filt_minimal)))

        if self.C(self.cfg.loss.lambda_maximal_filt) > 0:
            filt_maximal = optional_index_2d(out["filts"], -1, -1)

            set_loss("maximal_filt", F.l1_loss(filt_maximal, torch.ones_like(filt_maximal)))

        loss = 0.0
        on_step = (phase == "train")
        for name, value in loss_terms.items():
            self.log(f"{phase}/{name}", value, on_step=on_step, on_epoch=True)
            if name.startswith(loss_prefix):
                loss_weighted = value * self.C(
                    self.cfg.loss[name.replace(loss_prefix, "lambda_")]
                )
                self.log(f"{phase}/{name}_w", loss_weighted,
                         on_step=on_step, on_epoch=True)
                loss += loss_weighted

        if phase == "train":
            for name, value in self.cfg.loss.items():
                self.log(
                    f"train_params/{name}", self.C(value), on_step=True, on_epoch=False)

        self.log(f"{phase}/loss", loss, on_step=on_step,
                 on_epoch=True, prog_bar=True)

        return {"loss": loss, "output": out}

    def training_step(self, batch, batch_idx):
        out = self.loss(batch, "train")
        self.log_stats(out["output"], "train")
        return out["loss"]

    def on_train_epoch_end(self):
        if hasattr(self.model, "filtration") and hasattr(self.model.filtration, "ordered_thresholds"):
            with torch.no_grad():
                thresh: Tensor = self.model.filtration.ordered_thresholds
                lmp.info(f"Current thresholds: {thresh.tolist()}")

    def validation_step(self, batch, batch_idx):
        out = self.loss(batch, "val")
        self.log_stats(out["output"], "val")

        if self.cfg.visualize and (self.current_epoch % self.cfg.visualize_every_val == 0) and batch_idx == 0:
            self.visualize_output(batch, out["output"], [0, 1, 2], "val", batch_idx)

    def test_step(self, batch, batch_idx):
        out = self.loss(batch, "test")
        self.log_stats(out["output"], "test")

        if self.cfg.visualize and batch_idx == 0:
            self.visualize_output(batch, out["output"], [0, 1, 2], "test", batch_idx)

    def log_stats(self, output, phase="train"):
        acc_fn = getattr(self, f"{phase}_acc")
        roc_fn = getattr(self, f"{phase}_roc")

        acc_fn(output["pred"], output["target"].long())
        roc_fn(output["pred"], output["target"].long())

        self.log(f"{phase}/accuracy", acc_fn,
                 on_step=(phase == "train"), on_epoch=True)
        self.log(f"{phase}/roc_auc", roc_fn,
                 on_step=(phase == "train"), on_epoch=True)

    def visualize_output(self, batch, output, target_indices: list[int] = [0], phase: str = "val", batch_idx: int = 0):
        epoch = self.true_current_epoch
        for target_idx in target_indices:
            name_base = f"{phase}_{epoch}/{batch_idx}_{target_idx}_"

            input_fname = f"{name_base}inp.png"
            input_img = batch["input"][target_idx]
            data_range = (-1.0, 1.0)
            if input_img.shape[0] == 1:
                self.save_grayscale_image(
                    input_fname, input_img[0],
                    data_range=data_range,
                    cmap=None
                )
            else:
                self.save_rgb_image(
                    input_fname, input_img,
                    data_format="CHW",
                    data_range=data_range
                )

            if "comp_filt_norm" in output:
                cfn = output["comp_filt_norm"][target_idx]
                self.save_image_grid(
                    f"{name_base}cfn.png",
                    [[{
                        "img": cfn[i],
                        "type": "grayscale",
                        "kwargs": {
                            "cmap": None,
                            "data_range": (0., 1.)
                        }
                    }] for i in reversed(range(cfn.shape[0]))]
                )
            
            # if "comp_filt" in output:
            #     cf = output["comp_filt"][target_idx:target_idx+1]
            #     all_filt = comp_filt_convert(cf)[0]
            #     self.save_image_grid(
            #         f"{name_base}filt.png",
            #         [[{
            #             "img": all_filt[i, j],
            #             "type": "grayscale",
            #             "kwargs": {
            #                 "cmap": None,
            #                 "data_range": (0., 1.)
            #             }
            #         } for j in range(all_filt.shape[1])] for i in reversed(range(all_filt.shape[0]))]
            #     )

            if "filts" in output:
                filts = output["filts"]
                if isinstance(filts, Tensor):
                    filts = filts.movedim(0, 2).unsqueeze(3)
                self.save_image_grid(
                    f"{name_base}filt.png",
                    [[{
                        "img": filt[target_idx, 0],
                        "type": "grayscale",
                        "kwargs": {
                            "cmap": None,
                            "data_range": (0., 1.)
                        }
                    } for filt in row] for row in reversed(filts)]
                )


@torch.no_grad
def comp_filt_convert(compact_filt: Float[Tensor, "B C0 *D"]) -> Float[Tensor, "B C0 C1+1 *D"]:
    m, M = int(compact_filt.min()+0.5), int(compact_filt.max()+0.5)
    filts = []
    for k in range(m, M+1):
        compact_filt_k = (compact_filt <= (k+0.1)).to(torch.float32)
        filts.append(compact_filt_k)
    return torch.stack(filts, dim=2)


def optional_index_2d(x: Union[Float[Tensor, "B C0 C1 *D"], list[list[Float[Tensor, "B 1 *D"]]]], i: int, j: int) -> Float[Tensor, "B 1 *D"]:
    if isinstance(x, Tensor):
        return x[:, i:i+1, j]
    return x[i][j]