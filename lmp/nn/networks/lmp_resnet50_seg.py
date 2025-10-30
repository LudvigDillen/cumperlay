from dataclasses import dataclass, field
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *

from torchvision.ops.misc import Permute

from .base import BaseNetwork
from .resnet50_seg import Resnet50Seg

@lmp.register("network-lmp-resnet50-seg")
class LMPResnet50Seg(Resnet50Seg):
    @dataclass
    class Config(Resnet50Seg.Config):
        topo_ch: int = 256

        gate_bias: bool = True

        filtration_args: list[dict] = field(default_factory=list)
        persistence_args: list[dict] = field(default_factory=list)
        
        filt_rows: list[int] = field(default_factory=lambda: [16, 16, 16, 16])
        pers_chs: list[int] = field(default_factory=lambda: [128, 128, 128, 128])

        per_pixel_combine: bool = False

        input_guidance: bool = False
        input_filtration_type: str = ""
        input_filtration: dict = field(default_factory=dict)
        input_persistence_type: str = ""
        input_persistence: dict = field(default_factory=dict)
        input_filt_rows: int = 0
        input_pers_chs: int = 0
        input_direct: bool = False
        
        input_non_learned: bool = False
        input_no_grad_filt: bool = True

        no_deconv: bool = False

        grad_detach: bool = False

        topo_aux_ch: Optional[int] = None
        topo_aux_detach: bool = True

    cfg: Config

    def configure(self) -> None:
        self.topo_ch = self.cfg.topo_ch
        self.input_guidance = self.cfg.input_guidance
        self.input_direct = self.cfg.input_direct
        self.grad_detach = self.cfg.grad_detach
        self.topo_aux_ch = self.cfg.topo_aux_ch

        super().configure()

        input_pers_chs = self.cfg.input_pers_chs if self.input_guidance else 0

        self.update()

        if self.topo_aux_ch is not None:
            self.topo_net = nn.Sequential(
                nn.Linear(sum(self.cfg.pers_chs) + input_pers_chs, 2 * self.topo_ch),
                nn.ReLU(inplace=True),
                nn.Linear(2 * self.topo_ch, self.topo_aux_ch),
            )
            out_ch = 512 * self.expansion
            self.topo_out = nn.Sequential(
                nn.Conv2d(out_ch + self.topo_aux_ch, self.topo_aux_ch, kernel_size=1),
                nn.BatchNorm2d(self.topo_aux_ch),
                    nn.ReLU(inplace=True)
            )
        

    def _make_topo_layer(self, in_ch, filt_ch, pers_ch, f_args, p_args, n_blocks=1, filt_ch_last=256, upsample=True, **kwargs):
        if filt_ch == 0:
            # replace block
            assert pers_ch == 0 and f_args is None and p_args is None
            return SEReplacementBlock(in_ch, self.topo_ch)
        min_ch = filt_ch_last
        if "ch" in kwargs:
            min_ch = min(min_ch, kwargs["ch"])
        kwargs["permute"] = False
        return TopoBlock(
            in_ch, filt_ch, pers_ch, self.topo_ch, 
            lmp.find(f_args["type"])(f_args["config"]),
            lmp.find(p_args["type"])(p_args["config"]),
            n_blocks=n_blocks,
            filt_ch_last=filt_ch_last,
            no_deconv=self.cfg.no_deconv,
            upsample=upsample,
            num_groups=8 if min_ch >= 8 else 1,
            **kwargs
        )
    
    def _make_combine_layer(self, in_ch, out_ch, **kwargs):
        if self.cfg.per_pixel_combine:
            return PerPixelCombine(in_ch, out_ch, **kwargs)
        return GatedTopoLinear(in_ch, out_ch, **kwargs)

    def update(self):
        filtration_args = self.cfg.filtration_args
        persistence_args = self.cfg.persistence_args
        assert len(filtration_args) == 4 and len(persistence_args) == 4
        filt_rows = self.cfg.filt_rows
        pers_chs = self.cfg.pers_chs
        assert len(filt_rows) == 4 and len(pers_chs) == 4

        topo_ch_mult = 1
        if self.input_guidance:
            topo_ch_mult = 2
            inp_f_args = {
                "type": self.cfg.input_filtration_type,
                "config": self.cfg.input_filtration
            }
            inp_p_args = {
                "type": self.cfg.input_persistence_type,
                "config": self.cfg.input_persistence
            }
            if self.cfg.input_non_learned:
                self.input_topo = ClassicalTopoBlock(
                    3, self.cfg.input_filt_rows, self.cfg.input_pers_chs, self.topo_ch, 
                    lmp.find(inp_f_args["type"])(inp_f_args["config"]),
                    lmp.find(inp_p_args["type"])(inp_p_args["config"]),
                    permute=False,
                    no_grad_filt=self.cfg.input_no_grad_filt
                )
            elif self.input_direct:
                self.input_topo = self._make_topo_layer(3, self.cfg.input_filt_rows, self.cfg.input_pers_chs, inp_f_args, inp_p_args, 1, upsample=False, filt_ch_last=2*self.cfg.input_filt_rows, ch=2*self.cfg.input_filt_rows, permute=False)
            else:
                self.input_topo = self._make_topo_layer(64, self.cfg.input_filt_rows, self.cfg.input_pers_chs, inp_f_args, inp_p_args, 1)

        gate_kwargs = {
            "bias": self.cfg.gate_bias
        }
        self.topo_1 = self._make_topo_layer(64 * self.expansion, filt_rows[0], pers_chs[0], filtration_args[0], persistence_args[0], 1)
        self.linear_1 = self._make_combine_layer(topo_ch_mult * self.topo_ch, 64 * self.expansion, **gate_kwargs)

        self.topo_2 = self._make_topo_layer(128 * self.expansion, filt_rows[1], pers_chs[1], filtration_args[1], persistence_args[1], 2) 
        self.linear_2 = self._make_combine_layer(topo_ch_mult * self.topo_ch, 128 * self.expansion, **gate_kwargs)

        self.topo_3 = self._make_topo_layer(256 * self.expansion, filt_rows[2], pers_chs[2], filtration_args[2], persistence_args[2], 3) 
        self.linear_3 = self._make_combine_layer(topo_ch_mult * self.topo_ch, 256 * self.expansion, **gate_kwargs)

        self.topo_4 = self._make_topo_layer(512 * self.expansion, filt_rows[3], pers_chs[3], filtration_args[3], persistence_args[3], 4)
        self.linear_4 = self._make_combine_layer(topo_ch_mult * self.topo_ch, 512 * self.expansion, **gate_kwargs)

    def forward(self, x: Float[Tensor, "B Cin H W"], **kwargs) -> Float[Tensor, "B Cout"]:
        inp = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.input_guidance:
            if self.cfg.input_non_learned or self.input_direct:
                if "input_no_aug" in kwargs:
                    inp_for_topo = kwargs["input_no_aug"]
                else:
                    inp_for_topo = inp.detach()
                input_t, input_p = self.input_topo(inp_for_topo)
            else:
                input_t, input_p = self.input_topo(x.detach())

        x = self.layer1(x)
        t0, p0 = self.topo_1(x.detach() if self.grad_detach else x)
        if self.input_guidance:
            t0 = torch.cat((t0, input_t), dim=-1)
        x, _ = self.linear_1(x, t0)

        x = self.layer2(x)
        t1, p1 = self.topo_2(x.detach() if self.grad_detach else x)
        if self.input_guidance:
            t1 = torch.cat((t1, input_t), dim=-1)
        x, _ = self.linear_2(x, t1)

        x = self.layer3(x)
        t2, p2 = self.topo_3(x.detach() if self.grad_detach else x)
        if self.input_guidance:
            t2 = torch.cat((t2, input_t), dim=-1)
        out_3, _ = self.linear_3(x, t2)

        x = self.layer4(out_3)
        t3, p3 = self.topo_4(x.detach() if self.grad_detach else x)
        if self.input_guidance:
            t3 = torch.cat((t3, input_t), dim=-1)
        out_4, _ = self.linear_4(x, t3)

        out_dict = {
            "out": out_4,
        }
        if self.cfg.return_aux:
            out_dict["aux"] = out_3

        pers_list = [p0, p1, p2, p3]
        if self.input_guidance:
            pers_list.append(input_p)

        topo_l = torch.cat([torch.flatten(p_x["pers_rep"], start_dim=1, end_dim=-1) for p_x in pers_list if p_x is not None], dim=-1)
        topo_out = self.topo_net(topo_l)

        out_dict["pers_list"] = pers_list

        if self.topo_aux_ch is not None:
            topo_l = torch.cat([torch.flatten(p_x["pers_rep"], start_dim=1, end_dim=-1) for p_x in pers_list if p_x is not None], dim=-1)
            topo_out = self.topo_net(topo_l)
            topo_out = topo_out.unsqueeze(-1).unsqueeze(-2).expand((out_4.shape[0], self.topo_aux_ch, out_4.shape[2], out_4.shape[3]))
            out_for_topo = out_4.detach() if self.cfg.topo_aux_detach else out_4
            topo_out = self.topo_out(torch.cat((out_for_topo, topo_out), dim=1))
            out_dict["topo"] = topo_out

        return out_dict


class GatedTopoLinear(nn.Module):
    def __init__(self, in_ch, out_ch, bias=False):
        super().__init__()

        self.bias = bias
        mult = 2 if self.bias else 1
        self.lin = nn.Linear(in_ch, mult * out_ch)

    def forward(self, x, topo):
        out = self.lin(topo)
        if self.bias:
            scale, bias = F.sigmoid(out).chunk(2, dim=-1)
            scale = F.sigmoid(scale).unsqueeze(2).unsqueeze(3).expand_as(x)
            bias = bias.unsqueeze(2).unsqueeze(3).expand_as(x)
            out = x * scale + bias            
        else:
            scale = F.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
            out = x * scale
        x = x + out
        return x, out

class TopoBlock(nn.Module):
    def __init__(self, in_ch, filt_ch, pers_ch, out_ch, filtration, persistence, n_blocks=2, filt_ch_last = 128, num_groups = 8, no_deconv: bool = False, upsample: bool = True, ch = None, permute: bool = True) -> None:
        super().__init__()

        self.permute = Permute([0, 3, 1, 2]) if permute else nn.Identity()

        filt_dec_layers = []
        if ch is not None:
            filt_dec_layers.append(nn.Conv2d(in_ch, ch, kernel_size=3, padding=1, bias=True))
        curr_ch = in_ch if ch is None else ch
        for _ in range(n_blocks):
            next_ch = curr_ch // 2 if ch is None else ch
            filt_dec_layers.append(FiltrationDecoderBlock(curr_ch, next_ch, num_groups=num_groups, no_deconv=no_deconv, upsample=upsample))
            curr_ch = next_ch
        filt_dec_layers.extend([
            nn.GroupNorm(num_groups, curr_ch, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Conv2d(curr_ch, filt_ch_last, kernel_size=3, padding=1, bias=True),

            nn.GroupNorm(num_groups, filt_ch_last, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Conv2d(filt_ch_last, filt_ch, kernel_size=3, padding=1, bias=True),
        ])
        self.filtration_decoder = nn.Sequential(*filt_dec_layers)

        self.filtration = filtration
        self.persistence = persistence

        self.out_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pers_ch, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.permute(x)

        G: Float[Tensor, "B C H W"] = self.filtration_decoder(x)

        # Calculate compact filtration representations (grayscale, top level cells)
        G_hat_norm: Float[Tensor, "B C0 *D"]
        G_hat: Float[Tensor, "B C0 *D"]
        G_hat_norm, G_hat = self.filtration(G)

        # Calculate persistance representation
        G_p: Float[Tensor, "B C0 S dim"]
        G_p, pers_info = self.persistence(G_hat_norm, G_hat, None)

        out = self.out_net(G_p)

        return out, {
            "feat": G, # features
            "inputs": x,
            "comp_filt": G_hat, # compact filtration as top level elementary cells
            "comp_filt_norm": G_hat_norm, # compact filtration as top level elementary cells, normalized to [0, 1]
            # "filts": G_filts,
            "pers_rep": G_p, # persistence representation
            "pers_info": pers_info # persistence information
        }
    
class FiltrationDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups = 8, no_deconv: bool = False, upsample: bool = True) -> None:
        super().__init__()

        if no_deconv:
            up_layers = [
                nn.GroupNorm(num_groups, in_ch, eps=1e-5),
                nn.ReLU(inplace=True),
            ]
            if upsample:
                up_layers.append(nn.Upsample(scale_factor=2, mode="bilinear"))
            up_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True))
        else:
            up_layers = [
                nn.GroupNorm(num_groups, in_ch, eps=1e-5),
                nn.ReLU(inplace=True),
            ]
            if upsample:
                up_layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=True))
            else:
                up_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True))
            
            up_layers.extend([
                nn.GroupNorm(num_groups, out_ch, eps=1e-5),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
            ])
        self.up_block = nn.Sequential(*up_layers)
        self.net = nn.Sequential(
            nn.GroupNorm(num_groups, out_ch, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x):
        x = self.up_block(x)
        out = self.net(x)
        out += x
        return out
    

class SEReplacementBlock(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()

        self.net = nn.Sequential(
            Permute([0, 3, 1, 2]),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x), None

class ClassicalTopoBlock(nn.Module):
    def __init__(self, in_ch, filt_ch, pers_ch, out_ch, filtration, persistence, permute: bool = False, no_grad_filt: bool = True) -> None:
        super().__init__()

        self.no_grad_filt = no_grad_filt

        self.permute_block = Permute([0, 3, 1, 2])
        self.permute = permute
        
        self.filtration = filtration
        self.persistence = persistence

        self.out_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pers_ch, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        with torch.no_grad() if self.no_grad_filt else nullcontext():
            x_p = self.permute_block(x)
            if self.permute:
                x = x_p
            G: Float[Tensor, "B C H W"] = x
            
            # Calculate compact filtration representations (grayscale, top level cells)
            G_hat_norm: Float[Tensor, "B C0 *D"]
            G_hat: Float[Tensor, "B C0 *D"]
            G_hat_norm, G_hat = self.filtration(G)

        # Calculate persistance representation
        G_p: Float[Tensor, "B C0 S dim"]
        G_p, pers_info = self.persistence(G_hat_norm, G_hat, None)

        out = self.out_net(G_p)

        return out, {
            "feat": G, # features
            "inputs": x_p,
            "comp_filt": G_hat, # compact filtration as top level elementary cells
            "comp_filt_norm": G_hat_norm, # compact filtration as top level elementary cells, normalized to [0, 1]
            # "filts": G_filts,
            "pers_rep": G_p, # persistence representation
            "pers_info": pers_info # persistence information
        }
    
class PerPixelCombine(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()

        self.norm = nn.LayerNorm(out_ch)
        self.net = nn.Sequential(
            nn.Linear(in_ch + out_ch, 2 * out_ch),
            nn.GELU(),
            nn.Linear(2 * out_ch, out_ch)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: Tensor, topo: Tensor):
        _, H, W, _ = x.shape
        out = self.norm(x)
        out = torch.cat((
            out,
            topo.unsqueeze(1).unsqueeze(2).expand(-1, H, W, -1)
        ), dim=-1)
        x = x + self.net(out)
        return x, None