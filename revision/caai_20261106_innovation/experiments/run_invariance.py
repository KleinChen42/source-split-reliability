"""Real-data representation interventions and capacity-matched selection.

Fit only F, select only V, save all choices, then join held-out outcomes.
Run one development flow per GPU. All paths are within the shared project.
"""
from pathlib import Path
import argparse
import csv
import json
import sys
import time

import numpy as np
import torch
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
RECOVERY = ROOT / "revision/neucom_21462_20261104/recovery"
sys.path.insert(0, str(ROOT))
from neurocomputing_transfer.statistics.queue_bootstrap import ranked_order

ALPHAS = (.01, .1, 1., 10., 100., 1000.)
CAPACITIES = (.1, .3, .5, .7, .9)
NORMALIZATIONS = ("coordinate", "coordinate_block", "energy")
ENCODINGS = ("identity", "repeat4", "rotate")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    part.replace(path)


def bank(path, outcomes=False):
    with np.load(path, allow_pickle=False) as z:
        keys = ["record_ids", "source_ids", "candidates", "text", "native",
                "visual", "question_hidden"]
        if outcomes:
            keys.append("utility")
        return {key: z[key] for key in keys}


def rotate_visual(values):
    """X -> X diag(signs) H / sqrt(d), a fixed orthogonal re-encoding."""
    dim = values.shape[1]
    if dim != 2048:
        raise ValueError("The declared intervention uses 2048 visual coordinates")
    signs = np.random.default_rng(20261106).choice((-1., 1.), size=dim)
    x = torch.as_tensor(values, dtype=torch.float64, device="cuda")
    x = x * torch.as_tensor(signs, device="cuda")
    stride = 1
    while stride < dim:
        chunks = x.reshape(len(x), -1, 2, stride)
        left, right = chunks[:, :, 0, :], chunks[:, :, 1, :]
        x = torch.cat((left + right, left - right), dim=-1).reshape(len(x), dim)
        stride *= 2
    return (x / np.sqrt(dim)).cpu().numpy()


def raw_features(b, ci, block, encoding):
    pieces = [b["text"][ci]]
    if block != "none":
        key = "question_hidden" if block == "question" else (
            "visual_rotated" if encoding == "rotate" else "visual")
        pieces.append(b[key])
    pieces.append(b["native"][ci, :, None])
    return np.concatenate(pieces, axis=1).astype(np.float64)


def fit_geometry(raw, indices, normalization, repeat, block):
    x = torch.as_tensor(raw, dtype=torch.float64, device="cuda")
    fitting = x[indices]
    mean = fitting.mean(0)
    centered = fitting - mean
    variances = centered.square().mean(0)
    gain = torch.ones_like(mean)
    spans = [(0, 13), (13, raw.shape[1] - 1), (raw.shape[1] - 1, raw.shape[1])]
    for j, (lo, hi) in enumerate(spans):
        if lo == hi:
            continue
        multiplicity = repeat if j == 1 and block == "visual" else 1
        variance = variances[lo:hi]
        if normalization == "energy":
            energy = variance.sum()
            scale = torch.sqrt(energy) if energy > 0 else torch.ones_like(energy)
            # The actual repeated block has multiplicity times the energy.
            gain[lo:hi] = np.sqrt(multiplicity) / (scale * np.sqrt(multiplicity))
        else:
            std = torch.sqrt(variance)
            std[std == 0] = 1
            gain[lo:hi] = np.sqrt(multiplicity) / std
            if normalization == "coordinate_block":
                dim = max(1, int(torch.count_nonzero(variance > 0)))
                gain[lo:hi] /= np.sqrt(multiplicity * dim)
    x = (x - mean) * gain
    state = dict(mean=mean.cpu().numpy(), gain=gain.cpu().numpy(), block=block,
                 normalization=normalization, encoding="repeat4" if repeat == 4 else "identity")
    return state, x


def ridge_path(x, y, fit):
    xf = x[fit]
    x_mean, y_mean = xf.mean(0), y[fit].mean()
    xf = xf - x_mean
    gram = xf.T @ xf
    cross = xf.T @ (y[fit] - y_mean)
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    matrices = gram[None] + torch.tensor(ALPHAS, dtype=gram.dtype,
                                       device=gram.device)[:, None, None] * eye[None]
    chol, status = torch.linalg.cholesky_ex(matrices)
    if torch.any(status != 0):
        raise RuntimeError(f"Non-positive ridge system: {status.tolist()}")
    beta = torch.cholesky_solve(cross[None, :, None].expand(len(ALPHAS), -1, -1),
                                chol).squeeze(-1).T
    intercept = y_mean - x_mean @ beta
    return beta, intercept


def menus():
    return {
        "native": ("native",),
        "trees": ("native", "constant", "trees"),
        "visual_coordinate": ("native", "coordinate_block__identity"),
        "visual_energy": ("native", "energy__identity"),
        "question_coordinate": ("native", "question_coordinate_block"),
        "question_energy": ("native", "question_energy"),
        "combined_visual": ("native", "constant", "trees", "energy__identity"),
        "combined_question": ("native", "constant", "trees", "energy__identity", "question_energy"),
    }


def chosen_policies(configs, candidates):
    selected = []

    def choose(panel, eligible, q, selection_q, scope="ALL"):
        c = min(eligible, key=lambda item: (item["validation"][selection_q], item["id"]))
        selected.append(dict(panel=panel, scope=scope, capacity=q,
                             selection_capacity=selection_q, config=c["id"],
                             ci=c["ci"], candidate=c["candidate"], kind=c["kind"],
                             group=c["group"], alpha=c.get("alpha"),
                             validation_J=c["validation"][selection_q]))

    for norm in NORMALIZATIONS:
        for encoding in ENCODINGS:
            group = norm + "__" + encoding
            for ci, candidate in enumerate(candidates):
                subset = [c for c in configs if c["group"] == group and
                          c["ci"] == ci and c.get("alpha") == 1.]
                choose("fixed/" + group, subset, .7, .7, candidate)
            subset = [c for c in configs if c["group"] in (group, "native")]
            choose("expanded/" + group, subset, .7, .7)
    for menu, groups in menus().items():
        subset = [c for c in configs if c["group"] in groups]
        for q in CAPACITIES:
            choose("matched/" + menu, subset, q, q)
            choose("frozen70/" + menu, subset, q, .7)
    return selected


def real_input_checks(d, raw, x, fit, beta, intercept):
    """Two checks directly required by the new solver and compressed replication."""
    a = x.cpu().numpy()
    y = d["utility"][0].astype(float)
    reference = Ridge(alpha=1., solver="cholesky").fit(a[fit], y[fit]).predict(a[:512])
    actual = (x[:512] @ beta[:, 2] + intercept[2]).cpu().numpy()
    error = float(np.max(np.abs(reference - actual)))
    if error > 1e-7:
        raise ValueError(f"GPU/CPU ridge discrepancy: {error}")
    # Explicit repeat vs the sqrt(k) compressed ridge on real fitting records.
    v = raw[fit[:256], 13:45]
    other = raw[fit[:256]][:, [0, 1, -1]]
    explicit = np.column_stack((other, np.tile(v, (1, 4))))
    compact = np.column_stack((other, 2 * v))
    targets = y[fit[:256]]
    r1 = Ridge(alpha=1., solver="cholesky").fit(explicit, targets).predict(explicit)
    r2 = Ridge(alpha=1., solver="cholesky").fit(compact, targets).predict(compact)
    duplication_error = float(np.max(np.abs(r1 - r2)))
    if duplication_error > 1e-8:
        raise ValueError(f"Explicit replication discrepancy: {duplication_error}")
    return dict(real_record_cpu_gpu_max_error=error,
                explicit_compact_replication_max_error=duplication_error)


def run_unit(a, d, splits, flow, seed):
    dest = a.output / f"{flow}_{seed:02d}"
    if (dest / "complete.json").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    split_key = {"R": "record_random_fit", "S": "source_disjoint_fit"}[flow]
    fit = np.flatnonzero(splits[split_key][seed])
    validation = np.flatnonzero(~splits[split_key][seed])
    routing_seed = int(splits["routing_seeds"][seed])
    fit_gpu = torch.as_tensor(fit, device="cuda")
    candidates = d["candidates"].astype(str).tolist()
    configs, geometries, checks = [], {}, {}
    ids = d["record_ids"][validation]

    def add(group, ci, kind, score, tie, **parameters):
        order = ranked_order(score, ids, routing_seed, tie)
        losses = 1. - d["utility"][ci, validation]
        val = {q: float(losses[order[:int(np.floor(q * len(order)))]] .sum() / len(order))
               for q in CAPACITIES}
        configs.append(dict(id=group + "/" + tie, group=group, ci=ci,
                            candidate=candidates[ci], kind=kind, tie_key=tie,
                            validation=val, **parameters))

    for ci, candidate in enumerate(candidates):
        y = torch.as_tensor(d["utility"][ci], dtype=torch.float64, device="cuda")
        add("native", ci, "native", d["native"][ci, validation], candidate + "::native")
        constant = float(d["utility"][ci, fit].mean())
        add("constant", ci, "constant", np.full(len(validation), constant),
            candidate + "::constant:None", constant=constant)
        raw_tn = raw_features(d, ci, "none", "identity")
        tree = HistGradientBoostingRegressor(loss="squared_error", max_iter=200,
                    learning_rate=.1, max_leaf_nodes=15, min_samples_leaf=20,
                    l2_regularization=1., max_bins=255, early_stopping=False,
                    random_state=20261106 + 1000 * seed + ci).fit(raw_tn[fit], d["utility"][ci, fit])
        add("trees", ci, "tree", np.clip(tree.predict(raw_tn[validation]), 0, 1),
            candidate + "::hgb", model=tree)
        specifications = [(norm + "__" + enc, "visual", norm, enc)
                          for norm in NORMALIZATIONS for enc in ENCODINGS]
        specifications += [("question_" + norm, "question", norm, "identity")
                           for norm in ("coordinate_block", "energy")]
        for group, block, norm, encoding in specifications:
            raw = raw_features(d, ci, block, encoding)
            state, x = fit_geometry(raw, fit_gpu, norm, 4 if encoding == "repeat4" else 1, block)
            state["encoding"] = encoding
            geometries[ci, group] = state
            beta, intercept = ridge_path(x, y, fit_gpu)
            scores = torch.clamp(x[validation] @ beta + intercept, 0, 1).cpu().numpy()
            if flow == "R" and seed == 0 and ci == 0 and group == "coordinate__identity":
                checks.update(real_input_checks(d, raw, x, fit, beta, intercept))
            b, b0 = beta.cpu().numpy(), intercept.cpu().numpy()
            for j, alpha in enumerate(ALPHAS):
                add(group, ci, "ridge", scores[:, j], candidate + f"::ridge:{alpha:g}",
                    alpha=alpha, beta=b[:, j], intercept=float(b0[j]))
        print(json.dumps(dict(flow=flow, seed=seed, candidate=candidate,
                              fitted=True, seconds=round(time.perf_counter()-started, 1))), flush=True)

    selected = chosen_policies(configs, candidates)
    save(dest / "D_selected_policies.json", dict(flow=flow, seed=seed,
         fit_records=len(fit), validation_records=len(validation), policies=selected))
    config_map = {c["id"]: c for c in configs}
    used_ids = sorted({p["config"] for p in selected})
    id_index = {key: i for i, key in enumerate(used_ids)}
    joblib.dump(dict(configs={key: config_map[key] for key in used_ids}, geometries=geometries),
                dest / "fitted_policies.joblib", compress=3)
    arrays = {"config_ids": np.array(used_ids),
              "candidate_indices": np.array([config_map[k]["ci"] for k in used_ids], dtype=np.int16)}
    result_rows, invariance = [], []
    for cohort in ("H", "H2"):
        b = bank(a.bank_dir / f"activitynetqa_{cohort}.npz", outcomes=False)
        b["visual_rotated"] = rotate_visual(b["visual"])
        # This is the first access to the held-out outcomes in this unit.
        with np.load(a.bank_dir / f"activitynetqa_{cohort}.npz", allow_pickle=False) as z:
            losses = 1. - z["utility"]
        scores_by_id, transformed = {}, {}
        orders = []
        for key in used_ids:
            c = config_map[key]
            ci, kind, group = c["ci"], c["kind"], c["group"]
            if kind == "native":
                score = b["native"][ci]
            elif kind == "constant":
                score = np.full(len(b["record_ids"]), c["constant"])
            elif kind == "tree":
                score = np.clip(c["model"].predict(raw_features(b, ci, "none", "identity")), 0, 1)
            else:
                if (ci, group) not in transformed:
                    state = geometries[ci, group]
                    raw = raw_features(b, ci, state["block"], state["encoding"])
                    transformed[ci, group] = torch.as_tensor((raw-state["mean"]) * state["gain"],
                                                           dtype=torch.float64, device="cuda")
                z = transformed[ci, group]
                beta = torch.as_tensor(c["beta"], device="cuda")
                score = torch.clamp(z @ beta + c["intercept"], 0, 1).cpu().numpy()
            scores_by_id[key] = score
            orders.append(ranked_order(score, b["record_ids"], routing_seed, c["tie_key"]))
        arrays[cohort + "_orders"] = np.stack(orders).astype(np.int32)
        arrays[cohort + "_losses"] = losses.astype(np.float32)
        arrays[cohort + "_sources"] = b["source_ids"]
        arrays[cohort + "_records"] = b["record_ids"]
        for p in selected:
            ix = id_index[p["config"]]
            order = orders[ix]
            k = int(np.floor(p["capacity"] * len(order)))
            j = float(losses[p["ci"], order[:k]].sum() / len(order))
            result_rows.append(dict(flow=flow, seed=seed, cohort=cohort, **p,
                                   order_index=ix, incoming=len(order), accepted=k, J=j))
        for ci, candidate in enumerate(candidates):
            for norm in NORMALIZATIONS:
                base = f"{norm}__identity/{candidate}::ridge:1"
                for encoding in ("repeat4", "rotate"):
                    other = f"{norm}__{encoding}/{candidate}::ridge:1"
                    a_order, b_order = orders[id_index[base]], orders[id_index[other]]
                    k = int(np.floor(.7 * len(a_order)))
                    common = len(np.intersect1d(a_order[:k], b_order[:k]))
                    invariance.append(dict(flow=flow, seed=seed, cohort=cohort,
                        candidate=candidate, normalization=norm, encoding=encoding,
                        max_score_change=float(np.max(np.abs(scores_by_id[base]-scores_by_id[other]))),
                        changed_accepted_records=k-common,
                        loss_change=float((losses[ci, b_order[:k]].sum()-losses[ci, a_order[:k]].sum())/len(a_order))))
        del transformed, b, scores_by_id
        torch.cuda.empty_cache()
    with (dest / "evaluation_arrays.npz").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    save(dest / "results.json", result_rows)
    save(dest / "invariance.json", invariance)
    if checks:
        save(dest / "solver_checks.json", checks)
    save(dest / "complete.json", dict(status="REAL_DATA_UNIT_COMPLETE", flow=flow, seed=seed,
           gpu=torch.cuda.get_device_name(), seconds=time.perf_counter()-started,
           policies=len(selected), evaluated_configs=len(used_ids),
           solver="float64 primal ridge Cholesky", torch=torch.__version__))
    print(json.dumps(dict(flow=flow, seed=seed, status="COMPLETE",
                          seconds=round(time.perf_counter()-started, 1))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path, default=RECOVERY / "rebuilt/banks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flow", choices=("R", "S"), required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=30)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    d = bank(args.bank_dir / "activitynetqa_D.npz", outcomes=True)
    d["visual_rotated"] = rotate_visual(d["visual"])
    energy_difference = np.max(np.abs(np.square(d["visual"].astype(float)).sum(1)
                                      - np.square(d["visual_rotated"]).sum(1)))
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), orthogonal_norm_error=float(energy_difference))), flush=True)
    with np.load(RECOVERY / "splits/activitynetqa_30_seeds.npz", allow_pickle=False) as s:
        splits = {k: s[k] for k in s.files}
    if not np.array_equal(d["record_ids"], splits["record_ids"]):
        raise ValueError("Development split rows do not match the input bank")
    with threadpool_limits(limits=4):
        for seed in range(args.start, args.stop):
            run_unit(args, d, splits, args.flow, seed)


if __name__ == "__main__":
    main()
