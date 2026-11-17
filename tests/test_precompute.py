from pathlib import Path

import numpy as np
import pytest
import torch

from defdti.precompute import encode_texts


def test_t5_checkpoint_pools_encoder_states_only(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "CCO": 2, "CCN": 3}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer.save_pretrained(tmp_path)
    model = transformers.T5ForConditionalGeneration(
        transformers.T5Config(
            vocab_size=4, d_model=8, d_kv=4, d_ff=16, num_layers=1, num_heads=2, pad_token_id=0
        )
    ).eval()
    model.save_pretrained(tmp_path)

    texts = ["CCO", "CCN CCO"]
    actual = encode_texts(texts, model_name=str(tmp_path), max_length=8, device="cpu")
    with torch.inference_mode():
        expected = [
            model.encoder(**tokenizer(text, return_tensors="pt")).last_hidden_state.mean(dim=1)
            for text in texts
        ]

    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, torch.cat(expected).numpy(), rtol=1e-5, atol=1e-6)
