import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, DatasetDict
from transformers import WhisperProcessor
import evaluate
from transformers import Seq2SeqTrainer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from functools import partial
import numpy as np

from nlpaug.util.audio.loader import AudioLoader
from nlpaug.util.audio.visualizer import AudioVisualizer
import nlpaug.augmenter.spectrogram as nas
import nlpaug.flow as naf
import librosa
import matplotlib.pyplot as plt
from augment import time_warp, time_mask, freq_mask


#TODO load processor, model_checkpoint, and dataset
combined_data = load_dataset(path="common_voice_features")
processor = WhisperProcessor.from_pretrained("Models/whisper-small-Swedish", language="Swedish", task="transcribe")

def plot_spec(ax, spec, title):
    ax.set_title(title)
    ax.imshow(librosa.amplitude_to_db(spec), origin="lower", aspect="auto")


def plot_spec(ax, spec, title):
    ax.set_title(title)
    ax.imshow(librosa.amplitude_to_db(spec), origin="lower", aspect="auto")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        data_orig = batch['input_features'].clone()

        #No augment during evaluation
        if data_orig.shape[0] == 8:
            data_aug = data_orig
        else:
            data_aug = self.data_augment(data_orig)
        batch['input_features'] = data_aug


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

        return batch
    
    def data_augment(self, data:torch.Tensor)->torch.Tensor:
        #data = time_warp(data, W=10)
        data = freq_mask(data, F=27, num_masks=2)
        data = time_mask(data, T=20, num_masks=2)
        return data

    

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}




model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
#model = WhisperForConditionalGeneration.from_pretrained("whisper-small-Swedish/checkpoint-4000")

# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(
    model.generate, language="Swedish", task="transcribe", use_cache=True
)



training_args = Seq2SeqTrainingArguments(
    output_dir="Models/whisper-small-sv-test2",  # name on the HF Hub
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_steps=500,
    max_steps=6000,  # increase to 4000 if you have your own GPU or a Colab paid plan
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    hub_strategy = "all_checkpoints"
)



trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=combined_data["train"],
    eval_dataset=combined_data["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

trainer.train()

kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_13_0",
    #"dataset": "Common Voice 13.0",  # a 'pretty' name for the training dataset
    "dataset_args": "config: sv, split: test",
    "language": "sv",
    "model_name": "whisper-small-test2",  # a 'pretty' name for our model
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
    "tags": "hf-asr-leaderboard",
}

trainer.push_to_hub(**kwargs) 




 

