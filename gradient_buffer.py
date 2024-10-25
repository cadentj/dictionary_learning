import torch as t
from nnsight import LanguageModel
import gc
from tqdm import tqdm
from typing import Literal

class GradientBuffer:
    """
    Implements a buffer of activations. The buffer stores activations from a model, yields them in batches, and refreshes them when the buffer is less than half full.

    Attributes:
        data: generator which yields text data
        model: LanguageModel from which to extract activations
        submodule: submodule of the model from which to extract activations
        d_submodule: submodule dimension; if None, try to detect automatically
        io: can be 'in' or 'out'; whether to extract input or output activations
        n_ctxs: approximate number of contexts to store in the buffer
        ctx_len: length of each context
        refresh_batch_size: size of batches in which to process the data when adding to buffer
        out_batch_size: size of batches in which to yield activations
        device: device on which to store the activations
    """
    def __init__(self, 
        data, 
        model: LanguageModel, 
        submodule, 
        d_submodule = None, 
        io: Literal['in', 'out'] = 'out', 
        n_ctxs: float = 3e4, 
        ctx_len: int = 128, 
        refresh_batch_size: int = 512, 
        out_batch_size: int = 8192, 
        device: str = 'cpu'
    ):
        if io not in ['in', 'out']:
            raise ValueError("io must be either 'in' or 'out'")

        if d_submodule is None:
            try:
                if io == 'in':
                    d_submodule = submodule.in_features
                else:
                    d_submodule = submodule.out_features
            except:
                raise ValueError("d_submodule cannot be inferred and must be specified directly")
        self.activations = t.empty(0, d_submodule, device=device)
        self.read = t.zeros(0).bool()

        self.data = data
        self.model = model
        self.submodule = submodule
        self.d_submodule = d_submodule
        self.io = io
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.gradient_buffer_size: int = int(n_ctxs * ctx_len)
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device

        self.criterion = t.nn.CrossEntropyLoss()
        self.is_tuple = False
    
    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations
        """

        with t.no_grad():
            # if buffer is less than half full, refresh
            if (~self.read).sum() < self.gradient_buffer_size // 2:
                self.refresh()

            # return a batch
            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[t.randperm(len(unreads), device=unreads.device)[:self.out_batch_size]]
            self.read[idxs] = True

            return self.activations[idxs]
        
    def _is_tuple(self):
        with self.model.scan(" "):
            is_tuple = isinstance(self.submodule.output.shape, tuple)
        return is_tuple

    def text_batch(self, batch_size=None):
        """
        Return a list of text
        """
        if batch_size is None:
            batch_size = self.refresh_batch_size
        try:
            return [
                next(self.data) for _ in range(batch_size)
            ]
        except StopIteration:
            raise StopIteration("End of data stream reached")
    
    def tokenized_batch(self, batch_size=None):
        """
        Return a batch of tokenized inputs.
        """
        texts = self.text_batch(batch_size=batch_size)
        return self.model.tokenizer(
            texts,
            return_tensors='pt',
            max_length=self.ctx_len + 1, # +1 for the offset
            padding=True,
            truncation=True
        )

    def loss(self, logits, labels, mask):        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous()

        # Set labels to -100 where the mask is 0
        shift_labels[shift_mask == 0] = -100

        # Calculate and return the loss
        return self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )

    def refresh(self):
        gc.collect()
        t.cuda.empty_cache()
        self.activations = self.activations[~self.read]

        current_idx = len(self.activations)
        new_activations = t.empty(self.gradient_buffer_size, self.d_submodule, device=self.device)

        new_activations[: len(self.activations)] = self.activations
        self.activations = new_activations

        pbar = tqdm(total=self.gradient_buffer_size, initial=current_idx, desc="Refreshing activations")

        while current_idx < self.gradient_buffer_size:

            tokenized_batch = self.tokenized_batch()

            with t.enable_grad():

                with self.model.trace(tokenized_batch):
                    if self.io == "in":
                        hidden_states = self.submodule.inputs[0]
                    else:
                        hidden_states = self.submodule.output

                    if self.is_tuple:
                        gradients = hidden_states[0].grad
                    else:
                        gradients = hidden_states.grad

                    gradients.save()

                    loss = self.loss(
                        self.model.output.logits,
                        tokenized_batch["input_ids"],
                        tokenized_batch["attention_mask"]
                    )
                    loss.backward()
            
            attn_mask = tokenized_batch["attention_mask"][:, -1:]
            gradients = gradients.value
            gradients = gradients[attn_mask != 0][:, :-1]

            remaining_space = self.gradient_buffer_size - current_idx
            assert remaining_space > 0
            gradients = gradients[:remaining_space]

            self.activations[current_idx : current_idx + len(gradients)] = gradients.to(
                self.device
            )
            current_idx += len(gradients)

            pbar.update(len(gradients))

        pbar.close()
        self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

    @property
    def config(self):
        return {
            'd_submodule' : self.d_submodule,
            'io' : self.io,
            'n_ctxs' : self.n_ctxs,
            'ctx_len' : self.ctx_len,
            'refresh_batch_size' : self.refresh_batch_size,
            'out_batch_size' : self.out_batch_size,
            'device' : self.device
        }

    def close(self):
        """
        Close the text stream and the underlying compressed file.
        """
        self.text_stream.close()

