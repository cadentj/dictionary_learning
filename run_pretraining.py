#%%
from buffer import AllActivationBuffer
from training import train_scae_suite
from trainers.scae import TrainerSCAESuite, TrainerConfig, SubmoduleConfig
from utils import load_model_with_folded_ln2, load_iterable_dataset

from datasets import load_dataset
import torch as t
from nnsight import LanguageModel
from collections import defaultdict
from huggingface_hub import login
login("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")
device = "cuda:0" if t.cuda.is_available() else "cpu"


#%%
DTYPE = t.float32
MODEL_NAME = "gpt2"
expansion = 16
k = 128
remove_bos = True

out_batch_size = int(4*2048)
num_tokens = int(4e8)


#%%
model = load_model_with_folded_ln2(MODEL_NAME, device=device, torch_dtype=DTYPE)
data = load_iterable_dataset('Skylion007/openwebtext')

num_features = model.config.n_embd * expansion
n_layer = model.config.n_layer

#%%
initial_submodule = model.transformer.h[0]
submodules = {}
layernorm_submodules = {}
for layer in range(n_layer):
    submodules[f"mlp_{layer}"] = (model.transformer.h[layer].mlp, "in_and_out")
    submodules[f"attn_{layer}"] = (model.transformer.h[layer].attn, "out")

    layernorm_submodules[f"mlp_{layer}"] = model.transformer.h[layer].ln_2

submodule_names = list(submodules.keys())

buffer = AllActivationBuffer(
    data=data,
    model=model,
    submodules=submodules,
    initial_submodule=initial_submodule,
    layernorm_submodules=layernorm_submodules,
    d_submodule=model.config.n_embd, # output dimension of the model component
    n_ctxs=256,  # you can set this higher or lower depending on your available memory
    device="cuda",
    out_batch_size = out_batch_size,
    refresh_batch_size = 128,
    remove_bos=remove_bos
) 

#%%
submodule_cfg = SubmoduleConfig(
            dict_size=num_features,
            activation_dim=model.config.n_embd,
            k=k)
submodule_configs = {f'{module}_{down_layer}' : submodule_cfg for down_layer in range(n_layer) for module in ['attn', 'mlp']}

print("steps = ", num_tokens // out_batch_size)
trainer_cfg = TrainerConfig(
    steps=num_tokens // out_batch_size,
    use_vanilla_training=True,
    base_lr=5e-4
)

#%%
trainer = train_scae_suite(
    buffer,
    submodule_configs=submodule_configs,
    trainer_config=trainer_cfg,
    steps=num_tokens // out_batch_size,
    save_steps = 1000,
    dtype=DTYPE,
    device=device,
    log_steps = 20,
    use_wandb = True,
    repo_id_out = "jacobcd52/gpt2_suite_folded_ln_nobos",
    seed = 42,
    wandb_project_name="gpt2_pretrain_nobos"
)
# %%
