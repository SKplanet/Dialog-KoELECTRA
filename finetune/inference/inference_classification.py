from typing import Tuple, List, Union

import torch
import numpy as np
from transformers import ElectraTokenizer, ElectraForSequenceClassification


class ClassficationInfer:
    def __init__(
        self,
        model_dir,
        vocab_dir="skplanet/dialog-koelectra-small-discriminator",
        label_list=None,
        cuda=False,
    ):
        if cuda:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cpu"
        self.device = torch.device(device)
        self.model = ElectraForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = ElectraTokenizer.from_pretrained(
            vocab_dir, do_lower_case=False
        )
        self.label_list = None
        if label_list:
            self.label_list = label_list

    def convert_tokens_to_ids(self, x: List[str]) -> Tuple[list, list]:
        batch_encoding = self.tokenizer.batch_encode_plus(
            x, max_length=128, padding="max_length", truncation=True
        )
        input_ids = batch_encoding["input_ids"]
        att_mask = batch_encoding["attention_mask"]

        return input_ids, att_mask

    def inference(self, text: Union[str, List[str]]):
        with torch.no_grad():
            if type(text) == str:
                feat = self.convert_tokens_to_ids([text])
            else:
                feat = self.convert_tokens_to_ids(text)
            input_ids = torch.LongTensor(feat[0]).to(self.device)
            att_mask = torch.LongTensor(feat[1]).to(self.device)
            output = self.model(input_ids, att_mask)
            output = torch.softmax(output[0], dim=1)
        pred_label = None
        if type(text) == str:
            prob_dist = output[0].detach().cpu().numpy()
            max_idx = np.argmax(prob_dist)
            if self.label_list:
                pred_label = self.label_list[max_idx]
        else:
            prob_dist = output.detach().cpu().numpy()
            max_idx = np.argmax(prob_dist, axis=1)
            if self.label_list:
                pred_label = [self.label_list[idx] for idx in max_idx]

        return max_idx, pred_label

