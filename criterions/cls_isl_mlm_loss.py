# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from sklearn.metrics import accuracy_score
from fairseq.data import data_utils


@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("params.optimization.sentence_avg")


@register_criterion("cls_isl_mlm_loss", dataclass=CrossEntropyCriterionConfig)
class AllTaskLoss(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        #self.count = 0
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        #print("cal loss",self.count)
        #self.count+=1
        net_output,tso_label_score,net_output_cls, net_output_mlm, isl_label_score,cls_label_score= model(**sample["net_input"])


        #isl_mlm
        isl_mlm_sample_size = (sample["isl_mlm_batch_size"] if self.sentence_avg else sample["ntokens"])
        if sample["net_input"]["isl_mlm"] == True:
            isl_mlm_cls_loss = self.compute_isl_mlm_cls_loss(net_output_cls, sample, reduce=reduce)
            mlm_loss = self.compute_mlm_loss(net_output_mlm, sample, reduce=reduce)
            isl_mlm_loss = isl_mlm_cls_loss+mlm_loss
        else:
            isl_mlm_cls_loss = 0
            mlm_loss = 0
            isl_mlm_loss = 0

        #isl
        isl_sample_size = (sample["isl_batch_size"] if self.sentence_avg else sample["ntokens"])
        if sample["net_input"]["isl"] == True:
            isl_loss = self.compute_isl_loss(isl_label_score, sample, reduce=reduce)
        else:
            isl_loss = 0
        #cls
        cls_sample_size = sample["cls_batch_size"]
        if sample["net_input"]["cls"] == True:
            cls_loss = self.compute_cls_loss(cls_label_score, sample, reduce=reduce)
            #cls_acc = accuracy_score(sample["cls_target"].view(-1).to("cpu"), cls_label_score.argmax(-1).to("cpu"))
            cls_acc=0
        else:
            cls_loss = 0
            cls_acc = 0

        if sample["net_input"]["isl_mlm"] == True:
            sample_size = isl_mlm_sample_size
            total_loss = isl_mlm_loss

        if sample["net_input"]["isl"] == True:
            if sample["net_input"]["isl_mlm"]!=True:
                sample_size = isl_sample_size
                total_loss = isl_loss
            else:
                total_loss += isl_loss
        if sample["net_input"]["cls"] == True:
            if sample["net_input"]["isl_mlm"]!=True and sample["net_input"]["isl"]!=True:

                total_loss = cls_loss
                sample_size = cls_sample_size
            else:
                total_loss += cls_loss



        logging_output = {
            "loss": total_loss,

            "isl_mlm_cls_loss": isl_mlm_cls_loss,
            "mlm_loss": mlm_loss,

            "isl_loss": isl_loss,
            #"ntokens": sample["isl_mlm_ntokens"],

            'cls_loss':cls_loss,
            'cls_acc':cls_acc,

            "ntokens": sample_size,
            "nsentences": sample_size,
            "isl_mlm_sample_size": isl_mlm_sample_size,
            "isl_sample_size": isl_sample_size,
            "cls_sample_size": cls_sample_size,
        }
        #torch.cuda.empty_cache()
        return total_loss,sample_size, logging_output

    def compute_mlm_loss(self, net_output, sample, reduce=True):
        target= sample["isl_mlm_mlm_targets"]
        lprobs = net_output
        #print("target",target.shape)
        #print("lprobs",lprobs.shape)
        loss = F.nll_loss(
            lprobs,
            target,
            #reduction="sum" if reduce else "none",
        )
        #print("lprobs.argmax(-1)",lprobs.argmax(-1))
        #acc = accuracy_score(target.cpu(), lprobs.cpu().argmax(-1))
        #print("mlm acc",acc)
        return loss
    def compute_isl_mlm_cls_loss(self, net_output, sample, reduce=True):
        ground_truth_list = data_utils.collate_tokens(sample["isl_mlm_cls_targets"],
                                      pad_idx=100,
                                      eos_idx=None,
                                      move_eos_to_beginning=False).view(-1).long()
        pad_lprobs = torch.zeros([ground_truth_list.size(-1), 7], dtype=torch.float).cuda()
        lprobs = net_output.transpose(0,1).reshape(-1,7)
        for idx, imgs in enumerate(lprobs):
            pad_lprobs[idx][: imgs.size(-1)] = imgs
        #print("lprobs",lprobs)
        #print("pad_lpribs",pad_lprobs)

        loss = F.nll_loss(
                    pad_lprobs,
                    ground_truth_list,
                    reduction="mean",
                    ignore_index=100
                )
        #print("compute_cls_loss",loss)
        return loss

    def compute_isl_loss(self, net_output, sample, reduce=True):
        ground_truth_list = data_utils.collate_tokens(sample["isl_target"],
                                      pad_idx=100,
                                      eos_idx=None,
                                      move_eos_to_beginning=False).view(-1).long()
        #print(ground_truth_list.shape)
        pad_lprobs = torch.zeros([ground_truth_list.size(-1), 7], dtype=torch.float).cuda()
        lprobs = net_output.transpose(0,1).reshape(-1,7)
        #print(lprobs.shape)
        for idx, imgs in enumerate(lprobs):
            pad_lprobs[idx][: imgs.size(-1)] = imgs
        #print(lprobs.shape)
        loss = F.nll_loss(
                pad_lprobs,
                ground_truth_list,
                reduction="mean",
                ignore_index=100
            )
        #print("isl loss",loss)
        return loss
    def compute_cls_loss(self, net_output, sample, reduce=True):
        lprobs = net_output
        target = sample["cls_target"].view(-1)
        #print(lprobs.argmax(-1),target) 
        loss = F.nll_loss(
            lprobs,
            target,
            reduction="mean",
            ignore_index=100
        )
        #print(lprobs.argmax(-1),target)
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        #if logging_outputs[0].get("tso", 0)==True:

        #if logging_outputs[0].get("isl_mlm", 0)==True:
        isl_mlm_sample_size = sum(log.get("isl_mlm_sample_size", 0) for log in logging_outputs)
        mlm_loss_sum = sum(log.get("mlm_loss", 0) for log in logging_outputs)
        isl_mlm_cls_loss = sum(log.get("isl_mlm_cls_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "mlm_loss", mlm_loss_sum / math.log(2), isl_mlm_sample_size, round=3
        )
        metrics.log_scalar(
            "isl_mlm_cls_loss", isl_mlm_cls_loss / math.log(2), isl_mlm_sample_size, round=3
        )
        #if logging_outputs[0].get("isl", 0)==True:
        isl_loss_sum = sum(log.get("isl_loss", 0) for log in logging_outputs)
        isl_sample_size = sum(log.get("isl_sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "isl_loss_sum", isl_loss_sum  / math.log(2), isl_sample_size, round=3
        )

        cls_sample_size = sum(log.get("cls_sample_size", 0) for log in logging_outputs)
        cls_loss_sum = sum(log.get("cls_loss", 0) for log in logging_outputs)
        #cls_acc_sum = sum(log.get("cls_acc", 0) for log in logging_outputs)
        metrics.log_scalar(
            "cls_loss", cls_loss_sum  / math.log(2), cls_sample_size, round=3
        )
        #metrics.log_scalar("cls_acc", cls_acc_sum  / math.log(2), cls_sample_size, round=3)
        total_loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        #ntokens =1

        metrics.log_scalar(
            "loss", total_loss_sum, round=3
        )


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
