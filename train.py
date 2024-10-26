# %%

from nnsight import LanguageModel
from dictionary_learning import GradientBuffer, AutoEncoderTopK
from dictionary_learning.trainers import TrainerTopK
from dictionary_learning.training import trainSAE

device = "cuda:0"
model_name = "EleutherAI/pythia-70m-deduped" # can be any Huggingface model

model = LanguageModel(
    model_name,
    device_map=device,
    dispatch=True
)
submodule = model.gpt_neox.layers[3] # layer 1 MLP
activation_dim = 512 # output dimension of the MLP
dictionary_size = 64 * activation_dim


# %%

import json
import zstandard as zstd
import io
import os

def load_zst_files(base_path):
    for i in range(30):  # 00 to 29
        filepath = os.path.join(base_path, f"{i:02d}.jsonl.zst")
        with open(filepath, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for line in text_stream:
                    if line.strip():  # Skip empty lines
                        doc = json.loads(line)
                        yield doc['text']

base_path = "/share/data/datasets/pile/the-eye.eu/public/AI/pile/train"
data_iterator = load_zst_files(base_path)

# %%

from datasets import load_dataset

data = load_dataset("kh4dien/fineweb-100m-sample", split="train")
data = iter(data_iterator)

buffer = GradientBuffer(
    data=data,
    model=model,
    submodule=submodule,
    d_submodule=activation_dim, # output dimension of the model component
    n_ctxs=3e4,  # you can set this higher or lower dependong on your available memory
    device=device,
    refresh_batch_size=64,
)  # buffer will yield batches of tensors of dimension = submodule's output dimension


# %%

trainer_cfg = {
    "trainer": TrainerTopK,
    "dict_class": AutoEncoderTopK,
    "activation_dim": activation_dim,
    "dict_size": dictionary_size,
    "layer" : 1,
    "lm_name" : model_name,
    "device": device,
    "wandb_name" : "test",
}

# train the sparse autoencoder (SAE)
ae = trainSAE(
    data=buffer,  # you could also use another (i.e. pytorch dataloader) here instead of buffer
    trainer_configs=[trainer_cfg],
    save_dir="dictionaries",
    use_wandb=True,
    wandb_entity="gradient-features",
    wandb_project="topk",
    log_steps=128,
)
