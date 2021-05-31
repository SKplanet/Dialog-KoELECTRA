import random
import collections
from typing import Any, Dict

import torch
import numpy as np
import ruamel.yaml as yaml
from ruamel.yaml import CLoader
from scipy.stats import pearsonr, spearmanr
from seqeval import metrics as seqeval_metrics
from sklearn import metrics as sklearn_metrics

def yaml_load(path_or_file):
    if isinstance(path_or_file, str):
        with open(path_or_file, 'r', encoding='utf-8') as f:
            return yaml.load(f, Loader=CLoader)
    else:
        return yaml.load(path_or_file, Loader=CLoader)
    
def _nest_dict_rec(k: str, v: Any, out: Dict[str, Any], sep: str):
    k, *rest = k.split(sep, 1)
    if rest:
        _nest_dict_rec(rest[0], v, out.setdefault(k, {}), sep)
    else:
        out[k] = v
    
def make_nested(flat: Dict[str, Any], sep: str='.') -> Dict[str, Any]:
    result = {}
    for k, v in flat.items():
        _nest_dict_rec(k, v, result, sep)
    return result

def update(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in source.items():
        if isinstance(v, collections.abc.Mapping):
            target[k] = update(target.get(k, {}), v)
        else:
            target[k] = v
    return target

def update_nested(nested_to: Dict[str, Any], flat_from: Dict[str, Any], sep: str='.'):
    nested_from = make_nested(flat_from, sep=sep)
    return update(nested_to, nested_from)

def make_flat_dict(target: Dict[str, Any],
                   parent_key="",
                   ignore_key_fn=None,
                   sep=".") -> Dict[str, Any]:
    result = {}
    for k, v in target.items():
        if not ignore_key_fn is None and ignore_key_fn(k):
            continue
        cur_key = sep.join([p for p in [parent_key, k] if p])
        if isinstance(v, collections.abc.Mapping):
            result.update(make_flat_dict(v, cur_key, ignore_key_fn, sep))
        else:
            result.update({cur_key: v})

    return result

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def simple_accuracy(labels, preds):
    return (labels == preds).mean()


def acc_score(labels, preds):
    return {
        "acc": simple_accuracy(labels, preds),
    }


def pearson_and_spearman(labels, preds):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def f1_pre_rec(labels, preds, is_ner=True):
    if is_ner:
        return {
            "precision": seqeval_metrics.precision_score(labels, preds, suffix=True),
            "recall": seqeval_metrics.recall_score(labels, preds, suffix=True),
            "f1": seqeval_metrics.f1_score(labels, preds, suffix=True),
        }
    else:
        return {
            "precision": sklearn_metrics.precision_score(
                labels, preds, average="macro"
            ),
            "recall": sklearn_metrics.recall_score(labels, preds, average="macro"),
            "f1": sklearn_metrics.f1_score(labels, preds, average="macro"),
        }


def show_ner_report(labels, preds):
    return seqeval_metrics.classification_report(labels, preds, suffix=True)


def compute_metrics(task_name, labels, preds):
    assert len(preds) == len(labels)
    if task_name == "kornli":
        return acc_score(labels, preds), "acc"
    elif task_name == "ocb":
        return acc_score(labels, preds), "acc"
    elif task_name == "isac":
        return acc_score(labels, preds), "acc"
    elif task_name == "nsmc":
        return acc_score(labels, preds), "acc"
    elif task_name == "paws":
        return acc_score(labels, preds), "acc"
    elif task_name == "korsts":
        return pearson_and_spearman(labels, preds), "spearmanr"
    elif task_name == "question-pair":
        return acc_score(labels, preds), "acc"
    elif task_name == "naver-ner":
        return f1_pre_rec(labels, preds, is_ner=True), "f1"
    elif task_name == "hate-speech":
        return f1_pre_rec(labels, preds, is_ner=False), "f1"
    else:
        raise KeyError(task_name)
