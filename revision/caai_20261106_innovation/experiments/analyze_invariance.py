"""Aggregate new real-data interventions and reuse the established queue bootstrap."""
from pathlib import Path
from collections import defaultdict
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RECOVERY = ROOT / "revision/neucom_21462_20261104/recovery"
sys.path[:0] = [str(ROOT), str(RECOVERY)]
from analyze_extensions import bootstrap_problem, joint_inference, write_csv, save_json, save_npz

FLOWS = ("R", "S")
COHORTS = ("H", "H2")
NORMALIZATIONS = ("coordinate", "coordinate_block", "energy")
ENCODINGS = ("identity", "repeat4", "rotate")
PANELS = ["expanded/"+n+"__"+e for n in NORMALIZATIONS for e in ENCODINGS] + [
    "matched/native", "matched/trees", "matched/question_coordinate",
    "matched/question_energy", "matched/combined_visual", "matched/combined_question"]
QUEUE_MENUS = ("native", "trees", "visual_coordinate", "visual_energy",
               "question_coordinate", "question_energy", "combined_visual", "combined_question")


def unit_dirs(results):
    for seed in range(30):
        for flow in FLOWS:
            folder = results / f"{flow}_{seed:02d}"
            if not (folder/"complete.json").exists():
                raise RuntimeError("Incomplete real-data unit: "+str(folder))
            yield seed, flow, folder


def summaries(results, output):
    grouped = defaultdict(lambda: {f: [] for f in FLOWS})
    interventions, choices = [], []
    for seed, flow, folder in unit_dirs(results):
        rows = json.loads((folder/"results.json").read_text())
        interventions.extend(json.loads((folder/"invariance.json").read_text()))
        for row in rows:
            key = tuple(row[k] for k in ("cohort", "panel", "scope", "capacity", "selection_capacity"))
            grouped[key][flow].append(row["J"])
            if row["cohort"] == "H" and row["panel"].startswith("matched/"):
                choices.append({k: row[k] for k in ("seed","flow","panel","capacity","group","candidate","kind","alpha","validation_J")})
    points = []
    for key, flows in sorted(grouped.items()):
        if any(len(flows[f]) != 30 for f in FLOWS):
            raise ValueError("A point estimate is missing paired development seeds")
        row = dict(zip(("cohort", "panel", "scope", "capacity", "selection_capacity"), key))
        r, s = np.array(flows["R"]), np.array(flows["S"])
        points.append(dict(**row, J_R=float(r.mean()), J_S=float(s.mean()),
                           delta=float((r-s).mean()), seeds=30))
    write_csv(output/"points.csv", points)
    write_csv(output/"encoding_sensitivity.csv", interventions)
    write_csv(output/"development_choices.csv", choices)
    simple = []
    for norm in NORMALIZATIONS:
        for encoding in ("repeat4", "rotate"):
            sub = [r for r in interventions if r["normalization"] == norm and r["encoding"] == encoding]
            simple.append(dict(normalization=norm, encoding=encoding, comparisons=len(sub),
                max_score_change=max(r["max_score_change"] for r in sub),
                changed_sets=sum(r["changed_accepted_records"] > 0 for r in sub),
                mean_changed_records=float(np.mean([r["changed_accepted_records"] for r in sub])),
                max_changed_records=max(r["changed_accepted_records"] for r in sub)))
    write_csv(output/"invariance_summary.csv", simple)
    return points


def make_problem(results, cohort):
    assignment = np.full((30,2,len(PANELS)), -1, dtype=np.int32)
    orders, losses, semantic = [], [], defaultdict(list)
    expected = np.zeros((30,2,len(PANELS)))
    record_ids = source_ids = None
    for seed, flow, folder in unit_dirs(results):
        rows = [r for r in json.loads((folder/"results.json").read_text())
                if r["cohort"] == cohort and r["capacity"] == .7 and r["panel"] in PANELS]
        with np.load(folder/"evaluation_arrays.npz", allow_pickle=False) as data:
            if record_ids is None:
                record_ids = data[cohort+"_records"]
                source_ids = data[cohort+"_sources"]
            arr_orders, arr_loss = data[cohort+"_orders"], data[cohort+"_losses"]
            for row in rows:
                order, loss = arr_orders[row["order_index"]], arr_loss[row["ci"]]
                key = seed, row["ci"]
                uid = next((j for j in semantic[key] if np.array_equal(orders[j], order)), None)
                if uid is None:
                    uid = len(orders); orders.append(order); losses.append(loss)
                    semantic[key].append(uid)
                index = PANELS.index(row["panel"])
                assignment[seed, FLOWS.index(flow), index] = uid
                expected[seed, FLOWS.index(flow), index] = row["J"]
    if (assignment < 0).any():
        raise ValueError("Incomplete interval family")
    orders, losses = np.stack(orders), np.stack(losses).astype(float)
    n = orders.shape[1]
    computed = np.take_along_axis(losses, orders, axis=1)[:, :int(.7*n)].sum(1)/n
    if np.max(np.abs(computed[assignment]-expected)) > 1e-12:
        raise ValueError("Saved order and point loss differ")
    sources = np.unique(source_ids)
    if cohort == "H2":
        source_stratum = {}
        for line in (RECOVERY/"prepared/activitynetqa/H2.records.jsonl").read_text().splitlines():
            row=json.loads(line); source_stratum[row["source_id"]] = row["official_split"]
        strata = np.array([source_stratum[str(source)] for source in sources])
    else:
        strata = np.array(["all"]*len(sources))
    p = dict(dataset="activitynetqa", cohort=cohort, cells=[dict(cell_id=p) for p in PANELS],
             orders=orders, losses=losses, assignment=assignment, source_ids=source_ids,
             unique_source_ids=sources, source_strata=strata, record_ids=record_ids)
    return p, expected.mean(axis=0)


def finite_queues(results, output):
    rows = []
    for cohort in COHORTS:
        reference = results/"S_00/evaluation_arrays.npz"
        with np.load(reference, allow_pickle=False) as z:
            sources, inverse = np.unique(z[cohort+"_sources"], return_inverse=True)
        grouped = [np.flatnonzero(inverse == i) for i in range(len(sources))]
        rng = np.random.default_rng(2026110602 + COHORTS.index(cohort))
        source_orders = [rng.permutation(len(sources)) for _ in range(20)]
        batches = {size: [[np.concatenate([grouped[i] for i in order[start:start+size]])
                          for start in range(0,len(sources),size)] for order in source_orders]
                   for size in (10,50,100,1000)}
        aggregate = {size: np.zeros((30,len(QUEUE_MENUS),20)) for size in batches}
        for seed in range(30):
            folder = results/f"S_{seed:02d}"
            selected = {r["panel"].split("/")[1]: r for r in json.loads((folder/"results.json").read_text())
                        if r["cohort"]==cohort and r["capacity"]==.7 and
                        r["panel"] in ["matched/"+m for m in QUEUE_MENUS]}
            with np.load(folder/"evaluation_arrays.npz",allow_pickle=False) as z:
                orders, losses = z[cohort+"_orders"], z[cohort+"_losses"]
            cached = {}
            for mi, menu in enumerate(QUEUE_MENUS):
                p = selected[menu]; index = p["order_index"]
                if index not in cached:
                    rank = np.empty(len(inverse), dtype=np.int32)
                    rank[orders[index]] = np.arange(len(inverse))
                    loss = losses[p["ci"]]
                    values = {}
                    for size, repetitions in batches.items():
                        observed = []
                        for batch_list in repetitions:
                            total = 0.
                            for members in batch_list:
                                k = int(np.floor(.7*len(members)))
                                accepted = members[np.argpartition(rank[members], k-1)[:k]]
                                total += float(loss[accepted].sum())
                            observed.append(total/len(inverse))
                        values[size] = observed
                    cached[index] = values
                for size in batches:
                    aggregate[size][seed, mi] = cached[index][size]
        for size, values in aggregate.items():
            for mi, menu in enumerate(QUEUE_MENUS):
                per_order = values[:,mi].mean(axis=0)
                rows.append(dict(cohort=cohort, flow="S", menu=menu, sources_per_queue=size,
                    nominal_records_per_queue=size*10, capacity=.7, seeds=30, arrival_orders=20,
                    J=float(per_order.mean()), arrival_order_min=float(per_order.min()),
                    arrival_order_max=float(per_order.max())))
    write_csv(output/"finite_queue_points.csv", rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--stage",choices=("points","bootstrap","all"),default="all")
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.stage in ("points","all"):
        summaries(a.results,a.output)
        finite_queues(a.results,a.output)
        print("POINTS_AND_FINITE_QUEUES_COMPLETE",flush=True)
    if a.stage in ("bootstrap","all"):
        problems=[]
        for cohort in COHORTS:
            p,point=make_problem(a.results,cohort)
            print(json.dumps(dict(cohort=cohort,panels=len(PANELS),unique_policies=len(p['orders']))),flush=True)
            folder=a.output/cohort
            draws=bootstrap_problem(folder,p,1999,128,a.device)
            save_npz(folder/"points.npz",point=point)
            problems.append((p,point,draws))
        joint_inference(a.output,problems,1999)
        method=json.loads((a.output/"multiplicity_method.json").read_text())
        if method['family_endpoints']!=720:
            raise ValueError("The predeclared interval family differs")
    save_json(a.output/("complete_"+a.stage+".json"),dict(status="EXTENSION_ANALYSIS_COMPLETE",stage=a.stage))


if __name__=="__main__":
    main()
