from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv

import pytorch_lightning as pl
import torchmetrics
import torchmetrics.classification

import lmp
import lmp.nn as pnn
from lmp.util.typing import *

from .base import BaseTrainer


@lmp.register("classification-trainer")
class BinaryClassificationTrainer(BaseTrainer):
    @dataclass
    class Config(BaseTrainer.Config):
        model_type: str = ""
        model: dict = field(default_factory=dict)

        task: str = "binary" # "binary" | "multiclass" | "multilabel"
        num_classes: Optional[int] = None # Shouldn't be None for "multiclass"
        num_labels: Optional[int] = None # Shouldn't be None for "multilabel"

        bifilt_loss_until: int = 0
        bifilt_loss_n_columns: int = -1

        has_betti: bool = False

        visualize: bool = True
        visualize_every_val: int = 2
        visualize_using_comp_filt: bool = False
        visualize_using_comp_filt_F: int = 30

        multiclass_weights: Optional[list] = None

        topo_pred: bool = False

        vis_swin: bool = False
        vis_swin_train: bool = False
        vis_swin_steps: int = 1
        vis_swin_targets: int = 1
        save_vis_tensors: bool = False

        cfn_entropy: bool = False
        cfn_patch_size: int = 8
        cfn_img_size: int = 112
        cfn_patch_size_input: int = 16
        cfn_img_size_input: int = 112
        cfn_entropy_feats: bool = False
        
        input_non_learned: bool = False
        input_direct: bool = False

        multifilt_reg_loss: bool = False
        multifilt_reg_feats: bool = False

        multifilt_reg_clamped: bool = True
        multifilt_reg_sg_next: bool = False
        multifilt_reg_sg_curr: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.model = lmp.find(self.cfg.model_type)(self.cfg.model)
        lmp.info(self.model)

        self.task = self.cfg.task
        self.num_classes = self.cfg.num_classes
        self.num_labels = self.cfg.num_labels
        if self.task == "binary":
            self.ce_loss = nn.BCEWithLogitsLoss()
            self.target_map = lambda t: t.float()
        elif self.task == "multiclass":
            assert self.num_classes is not None, "multiclass requires num_classes"
            weights = None
            if self.cfg.multiclass_weights is not None:
                weights = torch.FloatTensor(self.cfg.multiclass_weights, device=self.device)
            self.ce_loss = nn.CrossEntropyLoss(weight=weights)
            self.target_map = lambda t: t.squeeze(-1) if len(t.shape) > 1 and t.shape[-1] == 1 else t
        elif self.task == "multilabel":
            assert self.num_labels is not None, "multilabel requires num_labels"
            self.ce_loss = nn.BCEWithLogitsLoss()
            self.target_map = lambda t: t.float()
        else:
            raise ValueError(f"Unknown task {self.task}")        

        metric_kwargs = {
            "task": self.task,
            "num_classes": self.num_classes,
            "num_labels": self.num_labels
        }
        self.train_acc = torchmetrics.classification.Accuracy(**metric_kwargs)
        self.train_roc = torchmetrics.classification.AUROC(**metric_kwargs)

        self.val_acc = torchmetrics.classification.Accuracy(**metric_kwargs)
        self.val_roc = torchmetrics.classification.AUROC(**metric_kwargs)

        self.test_acc = torchmetrics.classification.Accuracy(**metric_kwargs)
        self.test_roc = torchmetrics.classification.AUROC(**metric_kwargs)

        if self.cfg.topo_pred:
            self.topo_train_acc = torchmetrics.classification.Accuracy(**metric_kwargs)
            self.topo_train_roc = torchmetrics.classification.AUROC(**metric_kwargs)

            self.topo_val_acc = torchmetrics.classification.Accuracy(**metric_kwargs)
            self.topo_val_roc = torchmetrics.classification.AUROC(**metric_kwargs)

            self.topo_test_acc = torchmetrics.classification.Accuracy(**metric_kwargs)
            self.topo_test_roc = torchmetrics.classification.AUROC(**metric_kwargs)

        if self.cfg.cfn_entropy:
            self.entropy_loss = pnn.Entropy(self.cfg.cfn_patch_size, self.cfg.cfn_img_size, self.cfg.cfn_img_size)
            self.entropy_loss_input = pnn.Entropy(self.cfg.cfn_patch_size_input, self.cfg.cfn_img_size_input, self.cfg.cfn_img_size_input) 

        if self.cfg.multifilt_reg_loss:
            self.multifilt_reg_loss = pnn.MultifiltRegularizationLoss(self.cfg.multifilt_reg_clamped, self.cfg.multifilt_reg_sg_curr, self.cfg.multifilt_reg_sg_next)

    def forward(self, x, input_dict):
        kwargs = {}
        if self.cfg.has_betti:
            kwargs["betti"] = input_dict["betti"]
        if "input_no_aug" in input_dict:
            kwargs["input_no_aug"] = input_dict["input_no_aug"]
        pred, out = self.model(x, **kwargs)
        return pred, out

    def loss(self, input_dict, phase="train"):
        pred, other = self.forward(input_dict["input"], input_dict)
        target = self.target_map(input_dict["target"])
        if self.task == "binary" and len(pred.shape) == 2:
            pred = pred.squeeze(-1)  # [B, 1] -> [B]

        out = {
            "pred": pred,
            "target": target,
            **other
        }

        loss_terms = {}

        loss_prefix = f"loss_"

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value

        if self.C(self.cfg.loss.lambda_minimal_filt) > 0 or self.C(self.cfg.loss.lambda_maximal_filt) > 0:
            assert "filts" in out, "given loss term requires filts but network didnt output"
        if "lambda_bifilt" in self.cfg.loss and self.C(self.cfg.loss.lambda_bifilt) > 0:
            assert "filts" in out, "given loss term requires filts but network didnt output"
            assert "bifilt" in input_dict, "given loss terms requires bifilt but not found in output dict"

        # (Binary) Cross entropy
        set_loss("ce", self.ce_loss(pred, target))

        if "filts" in out and self.C(self.cfg.loss.lambda_minimal_filt) > 0:
            filt_minimal = optional_index_2d(out["filts"], 0, 0)

            set_loss("minimal_filt", F.l1_loss(filt_minimal, torch.zeros_like(filt_minimal)))

        if "filts" in out and self.C(self.cfg.loss.lambda_maximal_filt) > 0:
            filt_maximal = optional_index_2d(out["filts"], -1, -1)

            set_loss("maximal_filt", F.l1_loss(filt_maximal, torch.ones_like(filt_maximal)))

        if "filts" in out and "bifilt" in input_dict and self.C(self.cfg.loss.lambda_bifilt) > 0:
            if self.true_current_epoch < self.cfg.bifilt_loss_until:
                filts = out["filts"]
                filts_c = []
                for filt_row in filts:
                    if self.cfg.bifilt_loss_n_columns > 0:
                        filt_row = filt_row[:self.cfg.bifilt_loss_n_columns]
                    filts_c.append(torch.cat(filt_row, dim=1))
                filts = torch.stack(filts_c, dim=1)

                gt_filts = input_dict["bifilt"]
                if self.cfg.bifilt_loss_n_columns > 0:
                    gt_filts = gt_filts[:, :, :self.cfg.bifilt_loss_n_columns]
                set_loss("bifilt", F.l1_loss(filts, gt_filts))

        if self.cfg.topo_pred:
            topo_pred = out["topo_pred"]
            if self.task == "binary" and len(topo_pred.shape) == 2:
                out["topo_pred"] = topo_pred.squeeze(-1)
            else:
                out["topo_pred"] = topo_pred

            # (Binary) Cross entropy
            set_loss("topo_ce", self.ce_loss(out["topo_pred"], target))

        if self.cfg.cfn_entropy:
            layers = out["pers_list"]
            if len(layers) == 5 and self.cfg.input_non_learned:
                layers = layers[:-1]

            cfn_loss_total = 0.0
            for i, layer in enumerate(layers):
                if layer is None:
                    continue
                if self.cfg.cfn_entropy_feats:
                    if "feat" not in layer: 
                        continue
                    cfn = F.sigmoid(layer["feat"])
                else:
                    if "comp_filt_norm" not in layer:
                        continue
                    cfn = layer["comp_filt_norm"]
                is_inp_layer = len(layers) == 5 and (i == len(layers)-1)
                if is_inp_layer:
                    cfn_loss_total += self.entropy_loss_input(cfn).mean()
                else:
                    cfn_loss_total += self.entropy_loss(cfn).mean()
            set_loss("cfn_entropy", cfn_loss_total)

        if self.cfg.multifilt_reg_loss:
            layers = out["pers_list"]
            if len(layers) == 5 and self.cfg.input_non_learned:
                layers = layers[:-1]

            all_cfn = []
            for layer in layers:
                if layer is None:
                    continue
                if self.cfg.multifilt_reg_feats:
                    if "feat" not in layer: 
                        continue
                    cfn = F.sigmoid(layer["feat"])
                else:
                    if "comp_filt_norm" not in layer:
                        continue
                    cfn = layer["comp_filt_norm"]
                all_cfn.append(cfn)
            if len(layers) == 5 and all_cfn[0].shape[-1] != all_cfn[-1].shape[-1]:
                input_cfn = all_cfn[-1]
                all_cfn = torch.stack(all_cfn[:-1], dim=2)
            else:
                input_cfn = None
                all_cfn = torch.stack(all_cfn, dim=2)
            N_F = all_cfn.shape[1]

            mult_reg_loss_total = 0.0
            for i in range(N_F-1):
                cfn_curr = all_cfn[:, i]
                cfn_next = all_cfn[:, i+1]
                mult_reg_loss_total += self.multifilt_reg_loss(cfn_curr, cfn_next)
                # mult_reg_loss_total += (cfn_next - cfn_curr).clamp(min=0.0).mean(dim=(0, 2, 3)).sum()
                if input_cfn is not None:
                    mult_reg_loss_total += self.multifilt_reg_loss(input_cfn[:, i:i+1], input_cfn[:, i+1:i+2])
                    # mult_reg_loss_total += (input_cfn[:, i+1] - input_cfn[:, i]).clamp(min=0.0).mean()

            set_loss("multifilt_reg", mult_reg_loss_total)

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
        if self.current_epoch == 0 and batch_idx == 0:
            lmp.info(f"Shapes: {batch['input'].shape}, {batch['target'].shape}")
            lmp.info(f"Target: {batch['target'][:2]}")
        out = self.loss(batch, "train")
        self.log_stats(out["output"], "train")

        if self.cfg.vis_swin_train and batch_idx <= self.cfg.vis_swin_steps:
            self.visualize_swin(batch, out["output"], list(range(self.cfg.vis_swin_targets)), "train", batch_idx)

        return out["loss"]
    
    def optimizer_step(
        self,
        *args, **kwargs
    ):
        # For CBIS-DDSM
        """
        Skipping updates in case of unstable gradients
        https://github.com/Lightning-AI/lightning/issues/4956
        """
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                # valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                valid_gradients = not (torch.isnan(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            print("detected inf or nan values in gradients. not updating model parameters")
            self.zero_grad()
        
        pl.LightningModule.optimizer_step(self, *args, **kwargs)
    
    def on_train_epoch_end(self):
        if hasattr(self.model, "filtration") and hasattr(self.model.filtration, "ordered_thresholds"):
            with torch.no_grad():
                thresh: Union[Tensor, list, tuple] = self.model.filtration.ordered_thresholds
                if isinstance(thresh, tuple) or isinstance(thresh, list):
                    thresh = [x.tolist() for x in thresh]
                elif isinstance(thresh, Tensor):
                    thresh = thresh.tolist()
                lmp.info(f"Current thresholds: {thresh}")

    def validation_step(self, batch, batch_idx):
        out = self.loss(batch, "val")
        self.log_stats(out["output"], "val")

        if self.cfg.visualize and (self.current_epoch % self.cfg.visualize_every_val == 0) and batch_idx == 0:
            self.visualize_output(batch, out["output"], [0, 1, 2], "val", batch_idx)
        
        if self.cfg.vis_swin and batch_idx <= self.cfg.vis_swin_steps:
            self.visualize_swin(batch, out["output"], list(range(self.cfg.vis_swin_targets)), "val", batch_idx)

    def test_step(self, batch, batch_idx):
        out = self.loss(batch, "test")
        self.log_stats(out["output"], "test")

        if self.cfg.visualize and batch_idx == 0:
            self.visualize_output(batch, out["output"], [0, 1, 2], "test", batch_idx)
        
        if self.cfg.vis_swin and batch_idx <= self.cfg.vis_swin_steps:
            self.visualize_swin(batch, out["output"], list(range(self.cfg.vis_swin_targets)), "test", batch_idx)

    def log_stats(self, output, phase="train"):
        self._log_stats(output, phase=phase)
        if self.cfg.topo_pred and "topo_pred" in output:
            self._log_stats({
                "pred": output["topo_pred"],
                "target": output["target"]
            }, phase=phase, prefix="topo_")

    def _log_stats(self, output, phase="train", prefix=""):
        acc_fn = getattr(self, f"{prefix}{phase}_acc")
        roc_fn = getattr(self, f"{prefix}{phase}_roc")

        target = output["target"]
        if self.task == "multilabel":
            target = target.long()
        acc_fn(output["pred"], target)
        roc_fn(output["pred"], target)

        self.log(f"{phase}/{prefix}accuracy", acc_fn,
                 on_step=(phase == "train"), on_epoch=True)
        self.log(f"{phase}/{prefix}roc_auc", roc_fn,
                 on_step=(phase == "train"), on_epoch=True)

        if self.task in "binary":
            bal_acc = compute_balanced_accuracy_binary(
                self.task, output["pred"], target)
            self.log(f"{phase}/{prefix}balanced_accuracy", bal_acc,
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
            
            if self.cfg.visualize_using_comp_filt:
                assert "comp_filt" in output, "visualize_using_comp_filt requires comp_filt"
                
                cf = output["comp_filt"][target_idx:target_idx+1]
                all_filt = comp_filt_convert(cf, self.cfg.visualize_using_comp_filt_F)[0]
                self.save_image_grid(
                    f"{name_base}filt.png",
                    [[{
                        "img": all_filt[i, j],
                        "type": "grayscale",
                        "kwargs": {
                            "cmap": None,
                            "data_range": (0., 1.)
                        }
                    } for j in range(all_filt.shape[1])] for i in reversed(range(all_filt.shape[0]))]
                )

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

    def visualize_swin(self, batch, output, target_indices: list[int] = [0], phase: str = "val", batch_idx: int = 0):
        epoch = self.true_current_epoch
        for target_idx in target_indices:
            name_base = f"{phase}_{epoch}_{batch_idx}/{target_idx}_"

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

            layers = output["pers_list"]
            name_map = ["L0", "L1", "L2", "L3", "Linp"]

            if self.cfg.save_vis_tensors:
                save_data = dict()

            for i, layer in enumerate(layers):
                layer_name = name_map[i]
                if layer is None or "comp_filt_norm" not in layer:
                    continue
                is_inp_layer = layer_name == name_map[-1]

                cfn = layer["comp_filt_norm"][target_idx]
                self.save_image_grid(
                    f"{name_base}{layer_name}_cfn.png",
                    [[{
                        "img": cfn[i],
                        "type": "grayscale",
                        "kwargs": {
                            "cmap": None,
                            "data_range": (0., 1.)
                        }
                    }] for i in reversed(range(cfn.shape[0]))]
                )
                                
                cf = layer["comp_filt"][target_idx:target_idx+1]
                all_filt = comp_filt_convert(cf, self.cfg.visualize_using_comp_filt_F)[0]
                self.save_image_grid(
                    f"{name_base}{layer_name}_filt.png",
                    [[{
                        "img": all_filt[i, j],
                        "type": "grayscale",
                        "kwargs": {
                            "cmap": None,
                            "data_range": (0., 1.)
                        }
                    } for j in range(all_filt.shape[1])] for i in reversed(range(all_filt.shape[0]))]
                )

                if self.cfg.save_vis_tensors:
                    cfn: Tensor
                    save_data[f"{layer_name}_cfn"] = cfn.cpu().detach().numpy() # type: ignore
                    save_data[f"{layer_name}_filt"] = all_filt.cpu().detach().numpy() # type: ignore

                if is_inp_layer and self.cfg.input_non_learned:
                    continue
                feat_raw = layer["feat"][target_idx]
                with torch.no_grad():
                    feat = torch.sigmoid(feat_raw)
                self.save_image_grid(
                    f"{name_base}{layer_name}_raw_cfn.png",
                    [[{
                        "img": feat[i],
                        "type": "grayscale",
                        "kwargs": {
                            "cmap": None,
                            "data_range": (0., 1.)
                        }
                    }] for i in reversed(range(cfn.shape[0]))]
                )
                if is_inp_layer and self.cfg.input_direct:
                    continue

                layer_inputs_raw = layer["inputs"][target_idx]
                with torch.no_grad():
                    layer_inputs = torch.sigmoid(layer_inputs_raw)
                inp_F = layer_inputs_raw.shape[0]
                R = math.isqrt(inp_F)
                C = inp_F // R
                self.save_image_grid(
                    f"{name_base}{layer_name}_raw_inp.png",
                    [[{
                        "img": layer_inputs[i * C + j],
                        "type": "grayscale",
                        "kwargs": {
                            "cmap": None,
                            "data_range": (0., 1.)
                        }
                    } for j in range(C)] for i in range(R)]
                )
            if self.cfg.save_vis_tensors:
                tensor_base = f"{phase}_{epoch}_{batch_idx}_{target_idx}_"
                self.save_data(f"{tensor_base}_data.npz", save_data) # type: ignore

@torch.no_grad
def comp_filt_convert(compact_filt: Float[Tensor, "B C0 *D"], F: int) -> Float[Tensor, "B C0 C1+1 *D"]:
    filts = []
    for k in range(0, F):
        compact_filt_k = (compact_filt <= (k+0.1)).to(torch.float32)
        filts.append(compact_filt_k)
    return torch.stack(filts, dim=2)


def optional_index_2d(x: Union[Float[Tensor, "B C0 C1 *D"], list[list[Float[Tensor, "B 1 *D"]]]], i: int, j: int) -> Float[Tensor, "B 1 *D"]:
    if isinstance(x, Tensor):
        return x[:, i:i+1, j]
    return x[i][j]

def compute_balanced_accuracy_binary(task: str, logits: torch.Tensor, target: torch.Tensor) -> float:
    if task == "binary":
        # if logits are shape [N, 1], squeeze
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)

        # if logits: threshold at 0.0, if probabilities use 0.5 instead
        pred_labels = (logits > 0.0).long()
        true_labels = target.long()

        # flatten just in case
        pred_labels = pred_labels.view(-1)
        true_labels = true_labels.view(-1)

        tp = ((pred_labels == 1) & (true_labels == 1)).sum()
        tn = ((pred_labels == 0) & (true_labels == 0)).sum()
        fp = ((pred_labels == 1) & (true_labels == 0)).sum()
        fn = ((pred_labels == 0) & (true_labels == 1)).sum()

        tpr = tp.float() / (tp + fn + 1e-8)  # sensitivity
        tnr = tn.float() / (tn + fp + 1e-8)  # specificity

        bal_acc = 0.5 * (tpr + tnr)
    else:
        raise ValueError(f"Unsupported task for balanced accuracy: {task}")
    return bal_acc.item()
