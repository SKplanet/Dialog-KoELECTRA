import sys
import logging
import argparse

from konlpy.tag import Mecab
from tokenizers import BertWordPieceTokenizer
from huggingface_konlpy.tokenizers_konlpy import KoNLPyBertWordPieceTrainer

logger = logging.getLogger()


def build_vocab(args):
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    special_tokens += ["[unused{}]".format(idx) for idx in range(args.unused_size)]

    if args.tokenizer_model == "mecab_wordpiece":
        mecab_wordpiece_notag_trainer = KoNLPyBertWordPieceTrainer(
            Mecab(), use_tag=False
        )
        mecab_wordpiece_notag_trainer.train(
            files=[args.corpus],
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            limit_alphabet=args.limit_alphabet,
            special_tokens=special_tokens,
        )
        mecab_wordpiece_notag_trainer.save_model(
            "./data/vocab/mecab_normalize_{}".format(args.vocab_size), "notag"
        )

    elif args.tokenizer_model == "wordpiece":
        tokenizer = BertWordPieceTokenizer(
            vocab=None,
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=False,
            lowercase=False,
            wordpieces_prefix="##",
        )

        tokenizer.train(
            files=[args.corpus],
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            limit_alphabet=args.limit_alphabet,
            special_tokens=special_tokens,
        )

        tokenizer.save_model("./data/vocab/wordpiece")

    else:
        logger.info("tokenizer model : wordpiece / mecab_wordpiece")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        required=True,
        type=str,
        help="tokenizer model. (wordpiece / mecab_wordpiece)",
    )
    parser.add_argument(
        "--corpus", required=True, type=str, help="Train corpus file path."
    )
    parser.add_argument(
        "--vocab_size", required=False, type=int, default=40000, help="Vocabulary size."
    )
    parser.add_argument(
        "--min_frequency",
        required=False,
        type=int,
        default=3,
        help="word min frequency.",
    )
    parser.add_argument(
        "--limit_alphabet",
        required=False,
        type=int,
        default=6000,
        help="limit alphabet size.",
    )
    parser.add_argument(
        "--unused_size",
        required=False,
        type=int,
        default=500,
        help="vacab unused token size.",
    )
    args = parser.parse_args()

    build_vocab(args)


if __name__ == "__main__":
    main()
