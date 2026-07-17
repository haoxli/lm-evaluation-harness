import re

import datasets


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)


def process_docs_cot(dataset: datasets.Dataset) -> datasets.Dataset:
    """Generative/CoT variant of :func:`process_docs`.

    Reuses the standard preprocessing but also emits ``gold_letter`` (A-D) so a
    generate_until task can score the model's chosen letter with exact_match.
    """
    letters = ["A", "B", "C", "D"]

    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        gold = int(doc["label"])
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": gold,
            "gold_letter": letters[gold],
        }
        return out_doc

    return dataset.map(_process_doc)
