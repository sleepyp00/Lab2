import os
import argparse
import evaluate
from tqdm import tqdm
from pathlib import Path
from transformers import pipeline
from datasets import load_dataset, Audio
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import numpy as np

wer_metric = evaluate.load("wer")

def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    elif "transcription" in sample:
        return sample["transcription"]
    else:
        raise ValueError(
            f"Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of "
            ".join{sample.keys()}. Ensure a text column name is present in the dataset.")

def get_text_column_names(column_names):
    if "text" in column_names:
        return "text"
    elif "sentence" in column_names:
        return "sentence"
    elif "normalized_text" in column_names:
        return "normalized_text"
    elif "transcript" in column_names:
        return "transcript"
    elif "transcription" in column_names:
        return "transcription"


whisper_norm = BasicTextNormalizer()
def normalise(batch):
    batch["norm_text"] = whisper_norm(get_text(batch))
    return batch


""" def data(dataset):
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference": get_text(item), "norm_reference": item["norm_text"]} """

def data(dataset):
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference":item["sentence"]}


""" def batch(dataset):
    for i, features in enumerate(dataset):
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

    return batch """


def main():
    whisper_asr = pipeline("automatic-speech-recognition", model = "Sleepyp00/whisper-small-sv-test2")
    #whisper_asr = pipeline("automatic-speech-recognition", model = "Sleepyp00/whisper-small-swedish-cont")

    """ whisper_asr = pipeline(
        "automatic-speech-recognition", model="openai/whisper-small"
    ) """

    #common_voice = load_dataset(path="common_voice_features")

    

    #dataset = common_voice['test']

    whisper_asr.model.config.forced_decoder_ids = (
        whisper_asr.tokenizer.get_decoder_prompt_ids(
            language="Swedish", task="transcribe"
        )
    )

    dataset = load_dataset("mozilla-foundation/common_voice_13_0", "sv-SE", split="test")

    predictions = []
    references = []
    norm_predictions = []

    """ for d in dataset:
        features = d['input_features']
        labels = d['labels']
        v = whisper_asr(np.array(features))
        predictions.append(out["text"]) """

    for out in tqdm(whisper_asr(data(dataset), batch_size=16), desc='Decode Progress'):
        predictions.append(out["text"])
        references.append(out["reference"][0])
        wer = wer_metric.compute(references=references, predictions=predictions)
        norm_predictions.append(whisper_norm(out["text"]))
        print(wer)

    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2)

    print("\nWER : ", wer)


if __name__ == "__main__":
    main()