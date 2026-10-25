# SignBart WLASL-100 checkpoint

The model config in this folder supplies SignBart's architecture and its exact
100-label output order. Download `model.safetensors` from the author's
[SignBart WLASL-100 files](https://huggingface.co/tinh2312/SignBart-WLASL-100/tree/main)
and save it in this folder:

```text
models/signbart/model.safetensors
```

The weight file is ignored by Git. The project requires `safetensors`, included
in `requirements.txt`.

**Vocabulary note:** this released checkpoint's 100 labels differ from the
previous SPOTER WLASL-100 list in `config/settings.py` (only four labels overlap).
The recognizer uses the checkpoint's own `id2label` order from `config.json`.
If you need the previous SPOTER vocabulary, SignBart must be trained or
fine-tuned on that exact 100-class mapping before its weights can replace SPOTER.
