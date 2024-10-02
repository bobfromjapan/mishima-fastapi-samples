from fastapi import FastAPI, Depends
import uvicorn
from typing import Optional, List
from pydantic import BaseModel

from reinvent import version, runmodes, config_parse, setup_logger
from reinvent.runmodes.utils import set_torch_device
import torch
import platform
import random
import numpy as np
import os
import json
import pandas as pd
import uuid


# リクエストbodyを定義
class SamplingParameters(BaseModel):
    model_file: str

    smiles_file: Optional[str]
    sample_strategy: Optional[str]
    temperature: Optional[float]
    tb_logdir: Optional[str]

    output_file: str
    num_smiles: int
    unique_molecules: bool
    randomize_smiles: bool


class SamplingConf(BaseModel):
    run_type: str
    use_cuda: bool
    json_out_config: str
    seed: Optional[int]
    parameters: SamplingParameters


class Smiles(BaseModel):
    smiles: str
    note: Optional[str]


class MultipleSmiles(BaseModel):
    row: List[Smiles]

def set_seed(seed: int):
    """Set global seed for reproducibility

    :param seed: the seed to initialize the random generators
    """

    if seed is None:
        return

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


app = FastAPI()


@app.post("/sampling")
async def post(conf: SamplingConf, smiles: MultipleSmiles | None = None):
    run_type = conf.run_type
    runner = getattr(runmodes, f"run_{run_type}")
    logger = setup_logger(name=__package__, level="WARN", filename="logfile")

    use_cuda = conf.use_cuda
    actual_device = "cuda"

    if use_cuda == "true":
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        logger.info(f"Using CUDA device:{current_device} {device_name}")

        free_memory, total_memory = torch.cuda.mem_get_info()
        logger.info(
            f"GPU memory: {free_memory // 1024**2} MiB free, "
            f"{total_memory // 1024**2} MiB total"
        )
    else:
        logger.info(f"Using CPU {platform.processor()}")
        actual_device = "cpu"

    seed = conf.seed
    file_name = str(uuid.uuid4())
    conf.parameters.output_file = "/tmp/" + file_name + ".csv"

    if smiles is not None:
        conf.parameters.smiles_file = "/tmp/" + file_name + ".smi"
        with open(conf.parameters.smiles_file, "w") as smi_file:
            for r in smiles.row:
                if r.note:
                    smi_file.write(r.smiles + "\t" + r.note + "\n")
                else:
                    smi_file.write(r.smiles + "\n")

    if seed is not None:
        set_seed(seed)
        logger.info(f"Set seed for all random generators to {seed}")

    tb_logdir = conf.parameters.tb_logdir

    if tb_logdir:
        logger.info(f"Writing TensorBoard summary to {tb_logdir}")

    # conf().model_dump(mode='json') cannot use because reinvent4 requiries old 1.x pydantic
    runner(json.loads(conf.json()), actual_device, tb_logdir, None)

    df = pd.read_csv(conf.parameters.output_file)

    return {
        "message": "success",
        "run_type": conf.run_type,
        "results": df.to_dict(orient="records"),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
