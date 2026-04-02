import torch
from pathlib import Path
from safetensors.torch import save_file


def main():
    variants = ["n", "s", "m", "l"]
    for v in variants:
        print(f"yolov5{v}")
        model = torch.hub.load(
            "ultralytics/yolov5",
            f"yolov5{v}",
            pretrained=True,
            autoshape=False,
            skip_validation=True,
            trust_repo=True,
        )
        sd = model.state_dict()
        # the original model have duplicated "model." prefix: `model.model.<name>`
        # we remove the first "model." prefix here
        sd = {k.removeprefix("model."): v for k, v in sd.items()}
        out = Path(f"yolov5{v}.safetensors")
        save_file(sd, out)
        print(out.resolve())


if __name__ == "__main__":
    main()
