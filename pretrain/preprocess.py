import re
import os
import json
import glob
import shutil
import logging
import argparse

import kss
import numpy as np

logger = logging.getLogger()


def normalize_repeat_symbols(text: str, org_removed: "np.ndarray[np.int]"):
    repeat_sym_pattern = r"([^0-9a-zA-Z가-힣])\1{2,}"

    is_repeat = np.zeros(len(text), np.int)
    for m in re.finditer(repeat_sym_pattern, text):
        is_repeat[m.start(0) + 2 : m.end(0)] = 1

    norm_text = "".join(np.asarray([t for t in text])[is_repeat == 0])

    return norm_text


def make_train_data(corpus_dir, output_file):
    output_set = set()

    def parse_json_recursively(json_object, target_key):
        if type(json_object) is dict and json_object:
            for key in json_object:
                if key in target_key:
                    for sent in kss.split_sentences(json_object[key]):
                        output_set.add(str(sent))
                parse_json_recursively(json_object[key], target_key)
        elif type(json_object) is list and json_object:
            for item in json_object:
                parse_json_recursively(item, target_key)

    # Aihub 한국어 대화 말뭉치
    with open(os.path.join(corpus_dir, "ai_hub_dialog/dialog.json"), "r") as st_json:
        st_python = json.load(st_json)

    target_key = ["question", "answer"]
    parse_json_recursively(st_python, target_key)
    logger.info("sentence : {}".format(len(output_set)))
    logger.info("Finished Aihub dialog corpus...")

    # 국립국어원 신문 말뭉치
    for file_path in glob.glob(os.path.join(corpus_dir, "NIKL_newspaper/*.json")):
        try:
            with open(file_path, "r") as st_json:
                st_python = json.load(st_json)
            target_key = ["form"]
            parse_json_recursively(st_python, target_key)
        except:
            pass
    logger.info("sentence : {}".format(len(output_set)))
    logger.info("Finished NIKL newspaper corpus...")

    # 국립국어원 구어 말뭉치
    for file_path in glob.glob(os.path.join(corpus_dir, "NIKL_spoken/*.json")):
        try:
            with open(file_path, "r") as st_json:
                st_python = json.load(st_json)
            target_key = ["form"]
            parse_json_recursively(st_python, target_key)
        except:
            pass
    logger.info("sentence : {}".format(len(output_set)))
    logger.info("Finished NIKL spoken corpus...")

    # 챗봇 트레이닝용 문답 페어
    for line in open(os.path.join(corpus_dir, "korean_chatbot_data/ChatbotData.csv")):
        split = line.split(',')
        output_set.add(split[0])
        output_set.add(split[1])
    logger.info("Finished korean chatbot data...")

    # namuwikitext Corpus
    for file_type in ['train','test','dev']:
        for line in open(os.path.join(corpus_dir, f"namuwikitext/namuwikitext_20200302.{file_type}")):
            line = line.strip()
            if line:
                for sent in kss.split_sentences(line):
                    output_set.add(item.text)
    logger.info("Finished namuwikitext corpus...")

    writer = open(os.path.join(corpus_dir, "train_data/raw", output_file), "w")

    # KcBERT 댓글 데이터
    for line in open(os.path.join(corpus_dir, "kcbert/20190101_20200611_v2.txt"), "r"):
        for sent in kss.split_sentences(line):
            sent = normalize_repeat_symbols(sent, None)
            writer.write(str(sent) + "\n")
    logger.info("Finished KcBERT reply corpus...")

    # 문장 전처리
    for line in output_set:
        for sent in kss.split_sentences(line):
            sent = normalize_repeat_symbols(sent, None)
            writer.write(str(sent) + "\n")

    writer.close()
    logger.info("Finished...")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus_dir", type=str, required=True, help="corpus dir path")
    parser.add_argument(
        "--output_file", type=str, required=True, help="output file path"
    )

    args = parser.parse_args()

    make_train_data(args.data_dir, args.output_file)


if __name__ == "__main__":
    main()
