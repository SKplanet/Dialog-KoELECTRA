import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger()


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(self, guid, text_a, text_b, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def seq_cls_convert_examples_to_features(args, examples, tokenizer, max_length):

    output_mode = args.output_mode

    if "nsmc" in args.train_file:
        processor = NSMCProcessor(args)
    elif "kornli" in args.train_file:
        processor = KorNLIProcessor(args)
    elif "paws" in args.train_file:
        processor = PawsProcessor(args)
    elif "korsts" in args.train_file:
        processor = KorSTSProcessor(args)
    elif "question-pair" in args.train_file:
        processor = QuestionPairProcessor(args)
    elif "hate-speech" in args.train_file:
        processor = HateSpeechProcessor(args)
    else:
        processor = IntentProcessor(args)

    label_list = processor.get_labels()
    logger.info("Using label list {}".format(label_list))
    logger.info("Using output mode {}".format(output_mode))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example):
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        add_special_tokens=True,
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = [0] * len(inputs["input_ids"])  # For xlm-roberta

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.error("*** Example ***")
        logger.error("guid: {}".format(example.guid))
        logger.error(
            "input_ids: {}".format(" ".join([str(x) for x in features[i].input_ids]))
        )
        logger.error(
            "attention_mask: {}".format(
                " ".join([str(x) for x in features[i].attention_mask])
            )
        )
        logger.error(
            "token_type_ids: {}".format(
                " ".join([str(x) for x in features[i].token_type_ids])
            )
        )
        logger.error("label: {}".format(features[i].label))

    return features


class IntentProcessor(object):
    def __init__(self, args):
        self.args = args

    def get_labels(self):
        labels = set()
        for line in open(self.args.train_file, "r", encoding="utf-8"):
            label = line.split("\t")[1]
            labels.add(label.strip())
        return list(labels)

    @classmethod
    def _read_file(cls, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            line = line.split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            if i % 10000 == 0:
                logger.info(line)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
            )
        return examples

    def get_examples(self, mode):
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(file_to_read))
        return self._create_examples(self._read_file(file_to_read), mode)


class NSMCProcessor(object):
    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ["0", "1"]

    @classmethod
    def _read_file(cls, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines[1:]):
            line = line.split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[2]
            if i % 10000 == 0:
                logger.info(line)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
            )
        return examples

    def get_examples(self, mode):
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(file_to_read))
        return self._create_examples(self._read_file(file_to_read), mode)


class KorNLIProcessor(object):
    """Processor for the KorNLI data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    @classmethod
    def _read_file(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[1:]):
            line = line.split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            if i % 100000 == 0:
                logger.info(line)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file  # Only mnli for training
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        logger.info(
            "LOOKING AT {}".format(
                os.path.join(self.args.data_dir, self.args.task, file_to_read)
            )
        )
        return self._create_examples(
            self._read_file(
                os.path.join(self.args.data_dir, self.args.task, file_to_read)
            ),
            mode,
        )


class PawsProcessor(object):
    """Processor for the PAWS data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ["0", "1"]

    @classmethod
    def _read_file(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[1:]):
            line = line.split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            if text_a == "" or text_b == "":
                continue
            if i % 10000 == 0:
                logger.info(line)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        logger.info(
            "LOOKING AT {}".format(
                os.path.join(self.args.data_dir, self.args.task, file_to_read)
            )
        )
        return self._create_examples(
            self._read_file(
                os.path.join(self.args.data_dir, self.args.task, file_to_read)
            ),
            mode,
        )


class KorSTSProcessor(object):
    """Processor for the KorSTS data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return [None]

    @classmethod
    def _read_file(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[1:]):
            line = line.split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = line[5]
            text_b = line[6]
            label = line[4]
            if i % 1000 == 0:
                logger.info(line)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file  # Only mnli for training
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        logger.info(
            "LOOKING AT {}".format(
                os.path.join(self.args.data_dir, self.args.task, file_to_read)
            )
        )
        return self._create_examples(
            self._read_file(
                os.path.join(self.args.data_dir, self.args.task, file_to_read)
            ),
            mode,
        )


class QuestionPairProcessor(object):
    """Processor for the Question-Pair data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ["0", "1"]

    @classmethod
    def _read_file(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[1:]):
            line = line.split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            if text_a == "" or text_b == "":
                continue
            if i % 10000 == 0:
                logger.info(line)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        logger.info(
            "LOOKING AT {}".format(
                os.path.join(self.args.data_dir, self.args.task, file_to_read)
            )
        )
        return self._create_examples(
            self._read_file(
                os.path.join(self.args.data_dir, self.args.task, file_to_read)
            ),
            mode,
        )


class HateSpeechProcessor(object):
    """Processor for the Korean Hate Speech data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ["none", "offensive", "hate"]

    @classmethod
    def _read_file(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[1:]):
            line = line.split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[3]
            if i % 1000 == 0:
                logger.info(line)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
            )
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        logger.info(
            "LOOKING AT {}".format(
                os.path.join(self.args.data_dir, self.args.task, file_to_read)
            )
        )
        return self._create_examples(
            self._read_file(
                os.path.join(self.args.data_dir, self.args.task, file_to_read)
            ),
            mode,
        )


def seq_cls_load_and_cache_examples(args, tokenizer, mode):
    output_mode = args.output_mode
    if "nsmc" in args.train_file:
        processor = NSMCProcessor(args)
    elif "kornli" in args.train_file:
        processor = KorNLIProcessor(args)
    elif "paws" in args.train_file:
        processor = PawsProcessor(args)
    elif "korsts" in args.train_file:
        processor = KorSTSProcessor(args)
    elif "question-pair" in args.train_file:
        processor = QuestionPairProcessor(args)
    elif "hate-speech" in args.train_file:
        processor = HateSpeechProcessor(args)
    else:
        processor = IntentProcessor(args)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            str(args.task),
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_len),
            mode,
        ),
    )
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise ValueError("For mode, only train, dev, test is avaiable")
        features = seq_cls_convert_examples_to_features(
            args, examples, tokenizer, max_length=args.max_seq_len
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels
    )
    return dataset
