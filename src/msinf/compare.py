"""Reading posterior draws saved by :func:`msinf.fit.run_inference`."""

import numpy as np

#: Key ``run_inference`` writes to say whether the arrays are grouped by chain.
#: Needed because the shape cannot say: a scalar site over 4 chains x 800 draws
#: and a 41-site vector over 800 variational draws are both 2-D.
BY_CHAIN_KEY = "__grouped_by_chain__"


def load_draws_npz(path, *, drop_warmup: bool = True, by_chain: bool | None = None) -> dict[str, np.ndarray]:
    """Posterior draws from a ``samples.npz`` written by ``run_inference``.

    Chains are flattened, so the result has the ``(S, ...)`` shape
    :class:`msinf.fit.FitResult` uses and can be compared against any backend's
    draws directly. ``warmup__*`` entries are dropped: they are not posterior
    draws and mixing them into one would be wrong.

    ``by_chain`` says whether the leading axis is chains. It cannot be inferred
    from the shapes -- ``(4, 800)`` and ``(800, 41)`` are both 2-D and mean
    different things -- so it is read from :data:`BY_CHAIN_KEY` when the file
    carries it, and otherwise assumed true, which is what every file written
    before that key existed is.
    """
    loaded = np.load(path)
    if by_chain is None:
        by_chain = bool(loaded[BY_CHAIN_KEY]) if BY_CHAIN_KEY in loaded.files else True

    out = {}
    for key in loaded.files:
        if key == BY_CHAIN_KEY or (drop_warmup and key.startswith("warmup__")):
            continue
        arr = np.asarray(loaded[key])
        out[key] = arr.reshape((-1,) + arr.shape[2:]) if by_chain and arr.ndim >= 2 else arr
    return out
