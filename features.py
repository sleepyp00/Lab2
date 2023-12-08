from huggingface_hub import login
import os
from datasets import load_dataset, DatasetDict
from transformers import WhisperProcessor
import hopsworks
from datasets import Audio
project = hopsworks.login()
dataset_api = project.get_dataset_api()


def upload_folder_to_hopsworks(dataset_api, local_folder, HW_folder):
    #Will upload a folder to hopsworks

    # Iterate over each file in the folder
    if not dataset_api.exists(HW_folder):
        dataset_api.mkdir(HW_folder)
    for filename in os.listdir(local_folder):
        path = os.path.join(local_folder, filename)

        # Check if it's a file (not a subdirectory)
        if os.path.isfile(path):
            #upload_path = os.path.join(HW_folder, local_folder)
            dataset_api.upload(path, HW_folder)
        elif os.path.isdir(path):
            upload_folder_to_hopsworks(dataset_api, path, HW_folder + "/" + filename)
        else:
            raise Exception
        
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the token
my_token = os.getenv("HF_TOKEN")
login(token = my_token)


common_voice = DatasetDict()

common_voice["train"] = load_dataset(
    "mozilla-foundation/common_voice_13_0", "sv-SE", split="train+validation"
)
common_voice["test"] = load_dataset(
    "mozilla-foundation/common_voice_13_0", "sv-SE", split="test"
)

common_voice = common_voice.select_columns(["audio", "sentence"])

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="Swedish", task="transcribe"
)

processor.save_pretrained("Models/whisper-small-Swedish")

print(common_voice["train"].features)

sampling_rate = processor.feature_extractor.sampling_rate
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=sampling_rate))

def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["sentence"],
    )

    # compute input length of audio sample in seconds
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example

#print(common_voice['train'][0])
common_voice = common_voice.map(
    prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1
)

max_input_length = 30.0


def is_audio_in_length_range(length):
    return length < max_input_length

common_voice["train"] = common_voice["train"].filter(
    is_audio_in_length_range,
    input_columns=["input_length"],
)

print(common_voice["train"])

common_voice.save_to_disk("common_voice_features")


#upload_folder_to_hopsworks(dataset_api, "common_voice_features", "Lab2")