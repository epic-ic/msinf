"""Held-out evaluation on the Hydra path.

Resolves the config, calls :func:`msinf.evaluate.evaluate`, and writes what
comes back into the run's output directory.

This is a *Griffin* fit, so it keeps CORE_PARAMS, the reference draws and
init-by-value; only a caller passing its own parameters/curves needs the
name-free init (``mcmc.init_strategy=median``).

    python scripts/run_holdout.py +experiment=holdout_study
    python scripts/run_holdout.py -m +experiment=holdout_age holdout.seed=1,2,3,4,5
"""

import json
from pathlib import Path

import hydra
import numpyro
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from msinf.dtypes import set_precision

    # Precision fixes the float type and set_host_device_count fixes the device
    # count; JAX honours neither once its backend is initialized, so both have
    # to run before the first array is built.
    set_precision(str(cfg.run.precision))
    numpyro.set_host_device_count(int(cfg.mcmc.num_chains))

    from msinf.evaluate import evaluate
    from msinf.hydra_config import numpyro_config, split_from_cfg

    run_outdir = str(cfg.run.outdir).strip() if cfg.run.outdir is not None else ""
    outdir = Path(to_absolute_path(run_outdir)) if run_outdir else Path(HydraConfig.get().runtime.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    resolved = numpyro_config(cfg)
    split = split_from_cfg(cfg)
    result = evaluate(
        resolved,
        split=split,
        n_score_draws=(None if cfg.eval.n_score_draws is None else int(cfg.eval.n_score_draws)),
        n_replicates=int(cfg.eval.n_replicates),
        seed=int(cfg.run.seed),
        figures=bool(cfg.eval.figures),
    )

    (outdir / "metrics.json").write_text(
        json.dumps({"metrics": result.metrics, "info": result.info}, indent=2, sort_keys=True)
    )
    for name, payload in result.figures.items():
        (outdir / name).write_bytes(payload)

    print(json.dumps({"split": result.info.get("split"), **result.metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
