from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from msinf.dtypes import set_precision
from msinf.fit import run_inference
from msinf.hydra_config import numpyro_config


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Both of these have to run before the first array is built: precision fixes
    # the float type, and run_inference sets the XLA host device count, and JAX
    # honours neither once its backend is initialized.
    set_precision(str(cfg.run.precision))

    run_outdir = str(cfg.run.outdir).strip() if cfg.run.outdir is not None else ""
    outdir = Path(to_absolute_path(run_outdir)) if run_outdir else Path(HydraConfig.get().runtime.output_dir)

    result = run_inference(outdir=outdir, cfg=numpyro_config(cfg))
    print(result)


if __name__ == "__main__":
    main()
