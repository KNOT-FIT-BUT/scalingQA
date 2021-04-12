# Author: Martin Fajčík, 2021
import sys

import torch

if __name__ == "__main__":
    m = torch.load(sys.argv[1], map_location=torch.device("cpu"))
    model_dict = {
        "state_dict": m.half().state_dict() if len(sys.argv) > 3 and sys.argv[-1] == "fp16" else m.state_dict(),
        "config": m.config
    }
    torch.save(model_dict, sys.argv[2])
