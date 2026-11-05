"""Summarize existing matched-budget experiments and fixed-seed sensitivity."""
from pathlib import Path
import csv
import json
import numpy as np
from analyze_review_requests import write_csv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "analysis"
R1 = ROOT / "output/neurocomputing_execution_20260910/formal_results_c9b7cb93c34b"


def read(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def export_original_group_indices():
    """Export anonymous released group indices without inventing original IDs."""
    indices=[]
    for ds in ["vqav2", "docvqa", "activitynetqa"]:
        with np.load(R1/f"release/{ds}_primary.npz",allow_pickle=False) as z:
            indices.extend(dict(dataset=ds,cohort="H",record_index=i,source_group_index=int(g))
                           for i,g in enumerate(z["source_index"]))
    write_csv(OUT/"original_H_group_indices.csv",indices)
    print("Exported original H anonymous record/group indices:",len(indices))


def main():
    policies = read(R1/"publication/tables/supplemental_policies.csv")
    comparisons = read(R1/"publication/tables/supplemental_comparisons.csv")
    details, summary = [], []
    for ds in ["vqav2", "docvqa", "activitynetqa"]:
        for frac in [.25, .5, 1.0]:
            variant = f"matched_fit_{frac}"
            rows = [r for r in policies if r["dataset"] == ds and r["variant"] == variant]
            assert len(rows) == 10
            budgets = [json.loads(r["stress_metadata"])["matched_F_record_budget"] for r in rows]
            jr = np.mean([float(r["accepted_loss_per_total_record"]) for r in rows if r["flow"] == "record_random"])
            js = np.mean([float(r["accepted_loss_per_total_record"]) for r in rows if r["flow"] == "source_disjoint"])
            old = next(r for r in comparisons if r["dataset"] == ds and json.loads(r["variant"])["id"] == variant)
            np.testing.assert_allclose(jr-js, float(old["delta_record_minus_source"]), atol=1e-12, rtol=0)
            summary.append(dict(dataset=ds, fitting_fraction=frac, F_records_min=min(budgets), F_records_max=max(budgets),
                                J_R=float(jr), J_S=float(js), delta=float(jr-js), delta_per_1000=float(1000*(jr-js)),
                                native_selections=sum(r["selected_config"].endswith("::native") for r in rows),
                                total_selections=len(rows), seeds=5,
                                scope="EXISTING_EXPLORATORY_F_SIZE_MATCHED_V_UNCHANGED"))
            for r, budget in zip(rows, budgets):
                details.append(dict(dataset=ds, fitting_fraction=frac, seed=int(r["split_seed"]), flow=r["flow"],
                                    fitting_records=budget, selected_config=r["selected_config"],
                                    validation_sources_overlapping_F=int(r["true_F_V_source_overlap"]),
                                    J=float(r["accepted_loss_per_total_record"])))
    write_csv(OUT/"matched_budget_summary.csv", summary)
    write_csv(OUT/"matched_budget_policy_details.csv", details)

    seed_rows = []
    for ds in ["vqav2", "docvqa", "activitynetqa"]:
        rows = [r for r in policies if r["dataset"] == ds and r["variant"] == "primary"]
        # The release's supplemental table need not duplicate all primary seeds.
        seeds = sorted({int(r["split_seed"]) for r in rows})
        if len(seeds) == 30:
            delta = []
            for s in seeds:
                pair={r["flow"]:float(r["accepted_loss_per_total_record"]) for r in rows if int(r["split_seed"])==s}
                delta.append(pair["record_random"]-pair["source_disjoint"])
            a=np.array(delta)
            seed_rows.append(dict(dataset=ds, study="initial", contrast="primary", n_seeds=30, mean=float(a.mean()),
                                  seed_min=float(a.min()),seed_max=float(a.max()),
                                  leave_one_out_min=float(((a.sum()-a)/29).min()), leave_one_out_max=float(((a.sum()-a)/29).max())))
    for ds in ["vqav2", "activitynetqa"]:
        rows=read(ROOT/f"revision/numeric_release/E2_confirmation/{ds}/all_policy_losses.csv")
        rows=[r for r in rows if r["scorer"]=="original" and r["overlap"]==""]
        lookup={(r["variant"],r["flow"],int(r["seed"])):float(r["J"]) for r in rows}
        variants=sorted({r["variant"] for r in rows})
        assert all(v in variants for v in ["Tn","TVn","TV"]), variants
        def deltas(v):
            return np.array([lookup[v,"record_random",s]-lookup[v,"source_disjoint",s] for s in range(5)])
        for name,a in [("TVn",deltas("TVn")),("I1",deltas("TVn")-deltas("TV")),("I2",deltas("TVn")-deltas("Tn"))]:
            seed_rows.append(dict(dataset=ds,study="H2_original_recipe",contrast=name,n_seeds=5,mean=float(a.mean()),
                                  seed_min=float(a.min()),seed_max=float(a.max()),
                                  leave_one_out_min=float(((a.sum()-a)/4).min()),leave_one_out_max=float(((a.sum()-a)/4).max())))
    write_csv(OUT/"initial_and_H2_seed_sensitivity.csv",seed_rows)

    # Include the document source index from the unchanged original release.
    p=R1/"release/docvqa_primary.npz"
    with np.load(p,allow_pickle=False) as z:
        print("Original document array keys:",z.files[:8])
        if "record_ids" in z.files and "source_ids" in z.files:
            doc=[dict(dataset="docvqa",cohort="H",record_id=str(r),source_id=str(s)) for r,s in zip(z["record_ids"],z["source_ids"])]
            indices=read(OUT/"evaluation_source_indices.csv")
            indices=[r for r in indices if r["dataset"]!="docvqa"]+doc
            write_csv(OUT/"evaluation_source_indices.csv",indices)
    export_original_group_indices()
    print(json.dumps({"matched_budget_summary":summary,"seed_sensitivity":seed_rows},indent=2))


if __name__=="__main__":
    main()
