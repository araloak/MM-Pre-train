# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from omegaconf import II

import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from sklearn.metrics import accuracy_score
#from sklearn.metrics import classification_report
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("params.optimization.sentence_avg")


@register_criterion("seq_cls_mlm_cross_entropy", dataclass=CrossEntropyCriterionConfig)
class SeqClsMlmCrossEntropyCriterion(FairseqCriterion):
    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        FairseqCriterion.add_args(parser)
        parser.add_argument('--sentence_avg', default = True,action='store_true',
                            help='image feature dimension')

    def __init__(self, task,sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
    def forward(self, model, sample, reduce=True):
        #print("sample[mlm_targets]",sample["mlm_targets"].shape)
        net_output_cls, net_output_mlm= model(**sample["net_input"])
        #print("net_output_mlm",net_output_mlm.shape)
        acc = self.compute_acc(model,net_output_cls,sample)
        cls_loss = self.compute_cls_loss(model, net_output_cls, sample, reduce=reduce)
        mlm_loss = self.compute_mlm_loss(net_output_mlm, sample, reduce=reduce)
        loss = cls_loss+mlm_loss
        sample_size = sample["nsentences"]  #batch_size
        logging_output = {
            "cls_loss": cls_loss.data,
            "mlm_loss": mlm_loss.data,
            "loss": loss.data,
            "sample_size": sample_size,
            "cls_acc":acc
        }
        #print(classification_report(model.get_targets(sample, net_output).view(-1).to("cpu"), net_output.argmax(-1).to("cpu")))
        return loss, sample_size, logging_output
    def compute_mlm_loss(self, net_output, sample, reduce=True):
        target= sample["mlm_targets"]
        lprobs = net_output
        #print("target",target.shape)
        #print("lprobs",lprobs.shape)
        loss = F.nll_loss(
            lprobs,
            target,
            #reduction="sum" if reduce else "none",
        )
        print("lprobs.argmax(-1)",lprobs.argmax(-1))
        acc = accuracy_score(target.cpu(), lprobs.cpu().argmax(-1))
        print("mlm acc",acc)
        return loss
    def compute_cls_loss(self, model, net_output, sample, reduce=True):
        ground_truth_list= model.get_targets(sample, net_output)
        lprobs = net_output.transpose(0,1)
        loss = 0
        #print(lprobs.size())
        for index, target in enumerate(ground_truth_list):
            #print(len(target))
            if len(target)>0:
                loss += F.nll_loss(
                    lprobs[index][:len(target)],
                    target.long(),
                    reduction="sum" if reduce else "none",
                )
        return loss
    def compute_acc(self,model,net_output,sample):
        ground_truth_list = model.get_targets(sample, net_output)
        lprobs = net_output.transpose(0, 1).argmax(-1)
        labels,prediction = [],[]
        for index,each in enumerate(ground_truth_list):
            labels+=each.tolist()
            prediction+=lprobs[index][:len(each)].tolist()
        if len(labels)!=0 and len(prediction)!=0:
            acc = accuracy_score(labels, prediction)
            return acc
        else:
            return 1


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        mlm_loss_sum = sum(log.get("mlm_loss", 0) for log in logging_outputs)
        cls_loss_sum = sum(log.get("cls_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        acc = sum(log.get("cls_acc", 0) for log in logging_outputs)
        #print("acc",acc)
        #print("sample_size",sample_size)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "mlm_loss", mlm_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "cls_loss", cls_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "acc", acc, sample_size, round=3
        )
        metrics.log_scalar(
            "batch_size", sample_size, sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
