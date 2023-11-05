"""
Implements the knowledge distillation loss
"""
from abc import get_cache_token
import torch
from torch.nn import functional as F
from torch.nn.modules.loss import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from utils import batch_index_select
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import math

class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss
        

class IdleViTTotalLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and integrates the token cut loss.
    Knowledge distillation loss is selectable by the argument.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model, keep_ratios, idle_layers, 
                 cutloss_type='both', cut_weight=10.0, distill_loss=True, distill_weight=5.0):
        super().__init__()
        self.teacher_model = teacher_model
        self.base_criterion = base_criterion
        self.idle_layers = idle_layers
        self.keep_ratios = keep_ratios
        self.count = 0
        self.cls_loss = 0
        
        self.distill = distill_loss
        self.logits_distill_loss = 0
        self.feature_distill_loss = 0
        self.logits_distill_weight = distill_weight
        self.feature_distill_weight = distill_weight * 100
        
        self.cut_loss = 0
        self.cut_weight = cut_weight
        self.cutloss_type = cutloss_type

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        cls_s, token_s, out_attns, out_attn_masks, out_features = outputs
        
        # classification loss
        cls_loss = self.base_criterion(cls_s, labels)
        
        # token cut loss
        cut_loss = 0.0
        for i, mask in enumerate(out_attn_masks):
            B, N, _ = mask.shape
            W = out_attns[i].mean(1).softmax(dim = -1) # B,N,N
            
            if self.cutloss_type == "both":
                diffcut = torch.abs(mask.reshape(B,N,1) - mask.reshape(B,1,N)) # B,N,N
                diffcut[:,:,0] = 0.0
                samecut = mask.reshape(B,N,1) * mask.reshape(B,1,N) # B,N,N
                # cut
                inter_dist = (diffcut.reshape(B,N,N)*W).sum(-1) # B,N
                inter_loss = F.mse_loss(inter_dist, torch.zeros(B, N, dtype=inter_dist.dtype, device=inter_dist.device))
                intra_dist = (samecut*W).sum(-1) # B,N
                intra_loss = F.mse_loss(intra_dist, mask.reshape(B,N))
                cut_loss = cut_loss + inter_loss + intra_loss # B
                
            elif self.cutloss_type == "intra":
                samecut = mask.reshape(B,N,1) * mask.reshape(B,1,N) # B,N,N
                # cut
                intra_dist = (samecut*W).sum(-1) # B,N
                intra_loss = F.mse_loss(intra_dist, mask.reshape(B,N))
                cut_loss = cut_loss + intra_loss # B
                
            elif self.cutloss_type == "inter":
                diffcut = torch.abs(mask.reshape(B,N,1) - mask.reshape(B,1,N)) # B,N,N
                diffcut[:,:,0] = 0.0
                # cut
                inter_dist = (diffcut.reshape(B,N,N)*W).sum(-1) # B,N
                inter_loss = F.mse_loss(inter_dist, torch.zeros(B, N, dtype=inter_dist.dtype, device=inter_dist.device))
                cut_loss = cut_loss + inter_loss # B
                
            elif self.cutloss_type == "none":
                cut_loss = 0
                
            if i < len(out_attn_masks) - 1:
                del out_attns[i], out_attn_masks[i] # save memory during training
        
        if self.distill:
            # distillation loss
            with torch.no_grad():
                cls_t, token_t, teacher_features = self.teacher_model(inputs)
        
            # distilled logits loss
            logits_loss = F.kl_div(F.log_softmax(cls_s, dim=-1), F.log_softmax(cls_t, dim=-1), reduction='batchmean', log_target=True)
        
            # distilled feature loss
            if len(token_s.shape) == 2: # feature loss on [CLS] token
                feature_loss = F.mse_loss(F.normalize(token_s), F.normalize(token_t))
            else: # feature loss on selected image tokens
                feature_loss = F.mse_loss(F.normalize(token_s*out_attn_masks[-1]), F.normalize(token_t*out_attn_masks[-1]))
        else:
            logits_loss = 0
            feature_loss = 0
        
        loss = cls_loss + self.cut_weight * cut_loss / len(self.idle_layers) + self.logits_distill_weight * logits_loss + self.feature_distill_weight * feature_loss

        # print the loss per 100 epochs
        self.cls_loss += cls_loss.item()
        self.cut_loss += cut_loss.item() if self.cutloss_type!="none" else 0
        self.logits_distill_loss += logits_loss.item() if self.distill else 0
        self.feature_distill_loss += feature_loss.item() if self.distill else 0
        self.count += 1
        if self.count == 100:
            print('loss info: cls_loss=%.4f, cut_loss=%.4f, logits_loss=%.4f, feature_loss=%.4f' % (self.cls_loss / 100, self.cut_loss/ 100, self.logits_distill_loss/ 100, self.feature_distill_loss/100))
            self.count = 0
            self.cls_loss = 0
            self.cut_loss = 0
            self.logits_distill_loss = 0
            self.feature_distill_loss = 0
            
        return loss

