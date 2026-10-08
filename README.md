# bone-suppression-inference

Batch bone suppression for frontal chest X-rays of the RSHS dataset with two models:

| Model | Input size | Weights |
|---|---|---|
| Qure.ai bone suppression (TorchScript) | 1024x1024 | [bone_suppression.ts](https://huggingface.co/qureaiorg/bone-suppression/resolve/main/weights/bone_suppression.ts), CC BY-NC-SA 4.0 |
| ResNet-BS, Rajaraman et al. 2021 (ported from Keras to PyTorch) | 256x256 | [resnet_bs.h5](https://raw.githubusercontent.com/sivaramakrishnan-rajaraman/CXR-bone-suppression/main/resnet_bs.h5), license not stated |

Each image yields `<name>_soft_tissue.png` and `<name>_bone.png`. Finished images are skipped on re-run.

## Download the models

```bash
mkdir -p models/bone-suppression models/rajaraman-resnetbs
curl -L -o models/bone-suppression/bone_suppression.ts https://huggingface.co/qureaiorg/bone-suppression/resolve/main/weights/bone_suppression.ts
curl -L -o models/rajaraman-resnetbs/resnet_bs.h5 https://raw.githubusercontent.com/sivaramakrishnan-rajaraman/CXR-bone-suppression/main/resnet_bs.h5
```

## Run inference

```bash
uv sync
uv run suppress-bones --config configs/rshs.yaml
uv run suppress-bones --config configs/rshs-rajaraman-resnetbs.yaml
```

`--limit N` processes the first N images only. Edit `dataset_root`, `weights` and `output_dir` in the config to match your paths. The dataset folder needs a `manifest.csv` with the columns `path` (image path relative to the dataset folder) and `view` (only rows with `frontal` are processed). Requires a CUDA GPU.

## References

- Qure.ai bone suppression: https://huggingface.co/qureaiorg/bone-suppression
- Rajaraman et al. 2021: https://github.com/sivaramakrishnan-rajaraman/CXR-bone-suppression
