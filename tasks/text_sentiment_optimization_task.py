import os
import numpy as np

from fairseq.data import Dictionary, data_utils
from ..data.utils import text_bin_file
from fairseq.tasks import register_task, FairseqTask
from ..data.feature_dataset import FeatureDataset
from ..data.text_sentiment_optimization_dataset import TextSentOptDataset

@register_task('tso')
class TextSentOptTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--data-dir', default='output',
                            help='data directory')
        parser.add_argument('--sentiment-predictor-dir', default='output',
                            help='data directory')
        parser.add_argument('--max-src-sent', type=int, default=5,
                            help='max source sentence num')
        parser.add_argument('--max-obj', type=int, default=20,
                            help='max objects per sentence')
        parser.add_argument('--img-type', type=str, default="objects", choices=["features", "objects","objects_feature","diy_features"],
                            help='image feature types')
        parser.add_argument('--tso',  default=True,action='store_true',
                            help='image feature dimension')

    @classmethod
    def setup_task(cls, args, **kwargs):
        vocab_dict_file = os.path.join(args.data_dir, f'dict.txt')
        vocab_dict = Dictionary.load(vocab_dict_file)
        return TextSentOptTask(args, vocab_dict)

    def __init__(self, args, vocab_dict):
        super().__init__(args)
        self.args = args
        self.vocab_dict = vocab_dict

    def load_dataset(self, split, **kwargs):
        features_dataset = FeatureDataset(self.args.data_dir, split)  
        span_idxs = self.item2span_idxs(sent_num=features_dataset.sent_num,
                                        max_src_sent=self.args.max_src_sent)

        text_file = text_bin_file(self.args.data_dir, split)  
        text_dataset = data_utils.load_indexed_dataset(text_file, self.vocab_dict)

        self.datasets[split] = TextSentOptDataset(data_dir=self.args.data_dir,
                                                  text_dataset=text_dataset,
                                                  image_dataset=features_dataset,
                                                  vocab_dict=self.vocab_dict,
                                                  span_idxs=span_idxs,
                                                  shuffle=True if split == "train" else False,
                                                  split=split
                                                  )
    @staticmethod
    def item2span_idxs(sent_num: np.array, max_src_sent: int) -> np.array:
        """
        compute each src/tgt span of dataset.
        For example, if we got [[0,1,2], [3,4]] as source texts,
        sent_num should be [3, 2], and we want to use only one sentence as src.
        the output should be [[0, 0, 1], [0, 1, 2], [1, 0, 1]]
        """
        span_idxs = []
        for group_idx in range(sent_num.shape[0]):
            num = int(sent_num[group_idx])
            for sent_idx in range(1, num):  
                start_idx = max(0, sent_idx - max_src_sent)
                span_idxs.append((group_idx, start_idx, sent_idx))
        return np.array(span_idxs)

    @property
    def source_dictionary(self):
        return self.vocab_dict

    @property
    def target_dictionary(self):
        return self.vocab_dict
