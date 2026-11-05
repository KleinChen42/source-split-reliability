"""Reviewer-requested analyses of saved policy outputs; no refitting or new holdout.

Run from any directory with Python and NumPy. Inputs are the existing R3
portable numerical release and R2 H2 release. Historical checks are reused.
"""
from pathlib import Path
import csv
import itertools
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "analysis"
BASE = ROOT / "revision/r3/analysis/control_inference/portable"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    family_rows, family_draws, count_rows, seed_rows, capacity_rows = [], [], [], [], []
    source_rows, identity = [], []
    all_cell_points = {}
    for ds in ["vqav2", "activitynetqa"]:
        cells = []
        for stage in ["controls", "strong"]:
            folder = BASE / stage / ds
            meta = json.loads((folder / "policies.json").read_text(encoding="utf-8"))
            with np.load(folder / "queue_draws.npz", allow_pickle=False) as z:
                point, draws = z["point"].copy(), z["draws"].copy()
            with np.load(folder / "inputs.npz", allow_pickle=False) as z:
                orders, losses = z["orders"], z["losses"]
                n = orders.shape[1]
                sorted_losses = np.take_along_axis(losses, orders, axis=1)
                cumulative = np.cumsum(sorted_losses, axis=1)
                capacity_points = {q: cumulative[:, int(np.floor(q*n))-1] / n for q in [.1, .3, .5, .7, .9]}
                # This is the necessary check for the new capacity calculation.
                np.testing.assert_allclose(capacity_points[.7], point, atol=1e-12, rtol=0)
                if stage == "controls":
                    source_rows.extend(dict(dataset=ds, cohort="H", record_id=str(r), source_id=str(s))
                                       for r, s in zip(z["record_ids"], z["source_ids"]))
            groups = list(dict.fromkeys((m["panel"], m["representation"]) for m in meta))
            for panel, rep in groups:
                idx = {}
                for flow in ["record_random", "source_disjoint"]:
                    idx[flow] = sorted([i for i, m in enumerate(meta) if (m["panel"], m["representation"], m["flow"]) == (panel, rep, flow)], key=lambda i: meta[i]["seed"])
                    assert [meta[i]["seed"] for i in idx[flow]] == list(range(5))
                ir, iss = idx["record_random"], idx["source_disjoint"]
                p = np.array([point[ir].mean(), point[iss].mean(), point[ir].mean()-point[iss].mean()])
                d = np.column_stack([draws[:, ir].mean(1), draws[:, iss].mean(1), draws[:, ir].mean(1)-draws[:, iss].mean(1)])
                cells.append((f"{stage}/{panel}/{rep}", p, d))
                all_cell_points[ds, stage, panel, rep] = p
                per_seed = point[ir] - point[iss]
                loo = (per_seed.sum()-per_seed)/4
                seed_rows.append(dict(dataset=ds, stage=stage, panel=panel, representation=rep,
                                      mean_delta=float(per_seed.mean()), seed_min=float(per_seed.min()), seed_max=float(per_seed.max()),
                                      leave_one_out_min=float(loo.min()), leave_one_out_max=float(loo.max()),
                                      positive_seeds=int((per_seed>1e-12).sum()), negative_seeds=int((per_seed < -1e-12).sum()), n_seeds=5))
                for s in range(5):
                    identity.append(dict(dataset=ds, stage=stage, panel=panel, representation=rep, seed=s,
                                         delta=float(per_seed[s]), leave_this_seed_out=float(loo[s])))
                for q, qp in capacity_points.items():
                    jr, js = float(qp[ir].mean()), float(qp[iss].mean())
                    capacity_rows.append(dict(dataset=ds, stage=stage, panel=panel, representation=rep,
                                              capacity=q, accepted_records=int(q*n), incoming_records=n,
                                              J_R=jr, J_S=js, delta=jr-js, risk_R=jr/(int(q*n)/n), risk_S=js/(int(q*n)/n),
                                              delta_per_1000=1000*(jr-js), selection_capacity=.7,
                                              role="EXPLORATORY_FIXED_POLICY_CAPACITY_SENSITIVITY"))
            if stage == "controls":
                for m in meta:
                    if m["panel"] == "original_mix" and m["representation"] == "Tn":
                        count_rows.append({k: m[k] for k in ["dataset", "seed", "flow", "fit_records", "validation_records"]})
        # The family contains every cell and every unordered cell pair, for
        # J_R, J_S and delta, on BOTH datasets. No favorable subset is selected.
        for i, (left, p, d) in enumerate(cells):
            for e, endpoint in enumerate(["J_R", "J_S", "delta"]):
                family_rows.append(dict(dataset=ds, left=left, right="zero", endpoint=endpoint, estimate=float(p[e])))
                family_draws.append(d[:, e])
            for right, rp, rd in cells[i+1:]:
                for e, endpoint in enumerate(["J_R", "J_S", "delta"]):
                    family_rows.append(dict(dataset=ds, left=left, right=right, endpoint=endpoint, estimate=float(p[e]-rp[e])))
                    family_draws.append(d[:, e]-rd[:, e])

    theta = np.array([r["estimate"] for r in family_rows])
    matrix = np.column_stack(family_draws)
    assert matrix.shape == (1999, 2436)
    se = matrix.std(axis=0, ddof=1)
    nondegenerate = se > 1e-12
    # Same source draws within each dataset/stage. Pair the independently
    # generated dataset streams by draw number to approximate their product law.
    maxima = np.max(np.abs((matrix[:, nondegenerate]-theta[nondegenerate])/se[nondegenerate]), axis=1)
    critical = float(np.quantile(maxima, .95, method="higher"))
    lows, highs = theta-critical*se, theta+critical*se
    lows[~nondegenerate] = highs[~nondegenerate] = theta[~nondegenerate]
    nominal = np.quantile(matrix, [.025, .975], axis=0)
    for i, row in enumerate(family_rows):
        row.update(bootstrap_se=float(se[i]), nominal95_low=float(nominal[0,i]), nominal95_high=float(nominal[1,i]),
                   simultaneous95_low=float(lows[i]), simultaneous95_high=float(highs[i]),
                   estimate_per_1000=1000*row["estimate"], simultaneous95_low_per_1000=1000*float(lows[i]),
                   simultaneous95_high_per_1000=1000*float(highs[i]),
                   bootstrap_degenerate=bool(not nondegenerate[i]), role="EXPLORATORY_APPROXIMATE_SIMULTANEOUS_CONDITIONAL_D")
    write_csv(OUT/"implementation_multiplicity.csv", family_rows)
    write_csv(OUT/"implementation_seed_sensitivity.csv", seed_rows)
    write_csv(OUT/"implementation_seed_details.csv", identity)
    write_csv(OUT/"development_record_counts.csv", count_rows)
    write_csv(OUT/"implementation_capacity.csv", capacity_rows)
    np.savez_compressed(OUT/"multiplicity_maxima.npz", maxima=maxima)
    save_json(OUT/"multiplicity_method.json", dict(
        status="COMPLETED", family_size=len(family_rows), bootstrap_draws=len(maxima), critical_value=critical,
        nondegenerate_contrasts=int(nondegenerate.sum()), seeds=list(range(5)),
        family="Both datasets; 28 cells per dataset (20 implementation controls + 8 strong-learner cells); each cell and all 378 within-dataset cell pairs; each of J_R,J_S,delta. Total 2*(28+378)*3=2436.",
        interval="theta_hat +/- c*sd(theta_star); c is the conservative empirical .95 quantile (method=higher) of max(abs((theta_star-theta_hat)/sd(theta_star))) over the entire family.",
        scope="Approximate joint source-bootstrap intervals conditional on D and the five-seed fitted policy collection. The explored matrix was designed after H. Multiplicity adjustment does not create independent confirmation, repair adaptive search outside this finite family, prove bootstrap coverage, or add development seeds.",
        independent_datasets="Existing independently generated source-bootstrap streams are paired by draw index; identical within-dataset source weights are reused across stages by the saved release design.",
        degenerate="sd <= 1e-12 rows retained and marked bootstrap_degenerate. Zero empirical variation is not a population equivalence certificate.",
        no_new_fitting=True, no_new_holdout=True))

    h2rows = []
    for ds in ["vqav2", "activitynetqa"]:
        folder = ROOT / "revision/numeric_release/E2_confirmation" / ds
        meta = json.loads((folder/"policies.json").read_text(encoding="utf-8"))
        if isinstance(meta, dict):
            print("H2 metadata keys", ds, list(meta))
            continue
        with np.load(folder/"joint_queue_arrays.npz", allow_pickle=False) as z:
            source_rows.extend(dict(dataset=ds, cohort="H2", record_id=str(r), source_id=str(s))
                               for r,s in zip(z["record_ids"], z["source_ids"]))
        # Existing CSV already contains the frozen five-seed point losses.
        with (folder/"all_policy_losses.csv").open(encoding="utf-8",newline="") as f:
            h2rows.extend({**r, "dataset": ds} for r in csv.DictReader(f))
    write_csv(OUT/"evaluation_source_indices.csv", source_rows)
    if h2rows: write_csv(OUT/"H2_policy_losses.csv", h2rows)

    selected = [r for r in family_rows if r["endpoint"] == "delta" and (
        (r["right"]=="zero" and r["left"].startswith("controls/original_mix/")) or
        (r["left"]=="controls/original_mix/TVn" and r["right"]=="controls/original_mix/block_TVn"))]
    write_csv(OUT/"main_text_multiplicity.csv", selected)
    print(json.dumps({"family":len(family_rows),"critical":critical,"main_results":selected,
                      "counts":count_rows,"source_index_rows":len(source_rows)},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
