# coding: utf-8
import logging
import os
from typing import Dict

import hydra
import pandas as pd
from tqdm import tqdm

from config.hydra import cleanup_hydra
from detie_predict import DetIETripletExtractor
from modules.model import models


def load_sentences(filename: str) -> Dict[str, str]:
    """
        Loads the input sentences and their IDs from the golden annotations written in filename

        Args
        ----
        filename: str
            The name of the file where the golden annotations are written

        Returns
        -------
        sentences: dict[str, str]
            a mapping from sentence id to sentence
    """
    sentences = {}

    with open(filename, 'r', encoding='UTF-8') as f:
        file_lines = [line.strip() for line in f]

    for i in range(0, len(file_lines)):
        line = file_lines[i]

        if "sent_id:" in line:
            sent_id = line.split("\t")[0].split("sent_id:")[1]
            sentence = line.split("\t")[1]
            if sent_id in sentences:
                raise Exception
            sentences[sent_id] = sentence

    return sentences


def prepare_detie_benchie_format(sentences_raw_file_path, save_file_path, cfg, save_file=True, most_common=False):
    logging.info("Loading triplet extractor from checkpoint...")

    try:
        mte = DetIETripletExtractor(cfg, most_common=most_common)
    except Exception as e:
        logging.warning(str(e) + "; moving on...")
        mte = DetIETripletExtractor(
            model_name=cfg.model.name,
            best_ckpt_path=cfg.model.best_ckpt_path,
            best_hparams_path=cfg.model.best_hparams_path,
            most_common=most_common,
        )

    future_dataframe = {
        "sent_id": [],
        "arg1": [],
        "rel": [],
        "arg2": [],
    }
    raw_sentences = load_sentences(sentences_raw_file_path)
    for sent_id, raw_sentence in tqdm(raw_sentences.items()):
        oie_spans = mte(raw_sentence)

        for s, r, o in oie_spans:
            if len(s.strip()) == 0 or len(r.strip()) == 0 or len(o.strip()) == 0:
                continue
            future_dataframe["sent_id"].append(sent_id)
            future_dataframe["arg1"].append(s)
            future_dataframe["arg2"].append(o)
            future_dataframe["rel"].append(r)

    result_dataframe = pd.DataFrame(future_dataframe)

    if save_file:
        result_dataframe.to_csv(save_file_path, header=False, index=False, sep="\t")

    return result_dataframe


@cleanup_hydra
@hydra.main("../../../../config", "config.yaml")
def main(cfg):

    VERSIONS = [243, 263]

    for VERSION in VERSIONS:
        for language in ["de", "en", "zh"]:
            cfg.model.best_version = VERSION
            current_dir = os.path.dirname(__file__)

            test_set = f"{current_dir}/data/benchie/benchie_gold_annotations_{language}.txt"
            save_path = f"{current_dir}/systems_output/detie{cfg.model.best_version}benchie_output_{language}.txt"

            try:
                prepare_detie_benchie_format(test_set, save_path, cfg)
            except RuntimeError as rte:
                logging.error(str(rte) + " " + str(dir(models)))

                for model_name in dir(models):
                    if "Triplet" not in model_name:
                        continue
                    try:
                        cfg.model.name = model_name
                        prepare_detie_benchie_format(test_set, save_path, cfg)
                    except Exception as e:
                        logging.error(
                            str(e) + " " + f"This '{model_name}' is the wrong model name, moving on with {VERSION}"
                        )
                        # raise e


if __name__ == "__main__":
    main()
