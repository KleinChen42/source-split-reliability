"""F-only reliability fitting and V-only selection for the three extensions.

Actual dataset fitting is deferred until --gpu-ready is set after the user
supplies resources. --self-test uses synthetic arrays only. Saved H/H2 loss
arrays are read only after the complete D-selected policy set is written.
"""
from pathlib import Path
import argparse
import json
import sys
import joblib
import numpy as np
from filelock import FileLock
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from neurocomputing_transfer.statistics.queue_bootstrap import ranked_order
from prepare_extension import original_seed
from extension_layout import candidate_ids,panels as planned_panels,representations,WEIGHTS


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    partial.replace(path)


def read_bank(path,outcomes=False):
    z=np.load(path,allow_pickle=False)
    keys=["record_ids","source_ids","candidates","text","native","visual"]
    if "question_hidden" in z.files:keys.append("question_hidden")
    if outcomes:keys.append("utility")
    result={k:z[k] for k in keys};z.close()
    return result


def fit_geometry(D,ci,F,rep,seed,weight=None):
    use_q="Q" in rep
    extra=D["question_hidden"] if use_q else D["visual"]
    state=dict(rep=rep,weight=weight)
    if rep.startswith("PCA32"):
        vs=StandardScaler().fit(extra[F])
        pca=PCA(n_components=32,svd_solver="randomized",iterated_power=4,random_state=seed).fit(vs.transform(extra[F]))
        state.update(extra_scaler=vs,pca=pca)
        extra=pca.transform(vs.transform(extra))
    raw=raw_features(D,ci,rep,extra)
    scaler=StandardScaler().fit(raw[F]);w=np.ones(raw.shape[1])
    if rep.startswith("block_") or weight is not None:
        lam=1. if weight is None else weight
        for lo,hi in [(0,13),(13,raw.shape[1]-1),(raw.shape[1]-1,raw.shape[1])]:
            dim=max(1,int(np.count_nonzero(scaler.var_[lo:hi]>0)))
            w[lo:hi]=dim**(-lam/2)
    state.update(scaler=scaler,weights=w)
    return state,scaler.transform(raw)*w


def raw_features(bank,ci,rep,extra):
    if rep=="V":return extra
    base=bank["text"][ci]
    if rep not in ["T","Tn"]:base=np.column_stack([base,extra])
    if rep.endswith("n"):base=np.column_stack([base,bank["native"][ci]])
    return base


def apply_geometry(state,B,ci):
    extra=B["question_hidden"] if "Q" in state["rep"] else B["visual"]
    if "pca" in state:extra=state["pca"].transform(state["extra_scaler"].transform(extra))
    return state["scaler"].transform(raw_features(B,ci,state["rep"],extra))*state["weights"]


def value(scores,loss,ids,seed,key):
    order=ranked_order(scores,ids,seed,key)
    return float(loss[order[:int(.7*len(order))]].sum()/len(order)),order


def configuration_key(candidate,kind,param):
    # Keep the original routing/tie key, including the HGB exception used by
    # the fixed nonlinear comparator, when extending its first five seeds.
    return candidate+"::"+kind if kind in ["native","hgb"] else f"{candidate}::{kind}:{param}"


def eligible(kind,param,panel):
    if panel=="native_only":return kind=="native"
    if panel=="constant_only":return kind=="constant"
    if panel=="hgb_only":return kind=="hgb"
    if panel=="strong_mix":return kind in ["hgb","native","constant"]
    if panel=="ridge_only":return kind=="ridge" and param in [.1,1,10]
    if panel.startswith("alpha_"):return kind=="ridge" and param==float(panel.split("_",1)[1])
    if kind=="native":return panel!="common_only"
    if panel in ["original_mix","common_only"]:return kind in ["constant","knn"] or (kind=="ridge" and param in [.1,1,10])
    if panel=="ridge_original":return kind=="ridge" and param in [.1,1,10]
    if panel=="ridge_expanded":return kind=="ridge"
    if panel=="knn":return kind=="knn"
    raise ValueError(panel)


def fit_learner(a, rep, candidate, kind, parameter, features, utility, profiles):
    if kind == "ridge":
        model = Ridge(alpha=float(parameter), solver="lsqr", tol=1e-8)
    elif kind == "knn":
        model = KNeighborsRegressor(n_neighbors=int(parameter), weights="uniform", n_jobs=1)
    else:
        seed = original_seed("R3_HGB_FIXED", a.dataset, a.seed, a.flow, candidate, rep) % (2**32)
        model = HistGradientBoostingRegressor(**profiles["hgb_parameters"], random_state=seed)
    return model.fit(features, utility)


def choose_on_development(a, D, F, V, seed, candidates, rep, label, weight):
    profiles=json.loads((HERE/"fit_parameters.json").read_text())
    if a.task=="strong":
        methods=[("constant",None),("hgb",None),("native",None)]
    elif a.task=="block_weight":
        methods=[("ridge",x) for x in [.1,1,10]]+[("native",None)]
    else:
        expanded=a.task=="seed" and rep in ["Tn","TVn","block_TVn","PCA32_TVn"]
        grid=profiles["ridge_alphas"] if expanded else [.1,1,10]
        methods=[("constant",None)]+[("ridge",x) for x in grid]+[("knn",k) for k in [5,20,50]]+[("native",None)]
    panels=planned_panels(a.task,label)
    configs = []
    transforms = {}
    for ci,c in enumerate(candidates):
        pca_seed=original_seed("R3_PCA32",a.dataset,a.seed,a.flow,c)%(2**32)
        state,X=fit_geometry(D,ci,F,rep,pca_seed,weight)
        transforms[ci] = state
        y=D["utility"][ci]
        X_fit, y_fit = X[F], y[F]
        X_validation = X[V]
        for kind,param in methods:
            key=configuration_key(c,kind,param)
            model = None
            if kind=="native":
                vs = D["native"][ci, V]
            elif kind=="constant":
                model = float(y_fit.mean())
                vs = np.full(len(V), model)
            else:
                model = fit_learner(a, rep, c, kind, param, X_fit, y_fit, profiles)
                vs = np.clip(model.predict(X_validation), 0, 1)
            j,_=value(vs,1-y[V],D["record_ids"][V],seed,key)
            # kNN retains all of F in its estimator. Rebuild its search index
            # from the same F rows when applying a selected kNN configuration.
            configs.append(dict(candidate=c, ci=ci, kind=kind, param=param, key=key,
                                validation_J=j, model=None if kind == "knn" else model))
    selected=[]
    scopes=["ALL"]+candidates
    for panel in panels:
        for scope in scopes:
            subset=[c for c in configs if (scope=="ALL" or c["candidate"]==scope) and eligible(c["kind"],c["param"],panel)]
            chosen=min(subset,key=lambda c:(c["validation_J"],c["key"]))
            selected.append(dict(panel=panel,fixed_candidate=scope,candidate=chosen["candidate"],ci=chosen["ci"],config=chosen["key"],validation_J=chosen["validation_J"],
                                 dataset=a.dataset,seed=a.seed,flow=a.flow,representation=label,routing_seed=seed,fit_records=len(F),validation_records=len(V),
                                 selection_scope="pooled_candidate" if scope=="ALL" else "fixed_candidate"))
    keys = {row["config"] for row in selected}
    chosen_configs = {config["key"]: config for config in configs if config["key"] in keys}
    used = {config["ci"] for config in chosen_configs.values() if config["kind"] in ("ridge", "knn", "hgb")}
    return dict(policies=selected, configs=chosen_configs, transforms={ci: transforms[ci] for ci in used})


def evaluate_holdouts(a, dest, D, F, candidates, seed, checkpoint):
    selected = checkpoint["policies"]
    configs = checkpoint["configs"]
    transforms = checkpoint["transforms"]
    knn_indices = {}
    fit_features = {}
    for key, config in configs.items():
        if config["kind"] != "knn":
            continue
        ci = config["ci"]
        if ci not in fit_features:
            fit_features[ci] = apply_geometry(transforms[ci], D, ci)[F]
        model = KNeighborsRegressor(n_neighbors=int(config["param"]), weights="uniform", n_jobs=1)
        knn_indices[key] = model.fit(fit_features[ci], D["utility"][ci, F])

    arrays = {}
    rows = []
    for part in ("H", "H2"):
        B = read_bank(a.bank_dir / f"{a.dataset}_{part}.npz", False)
        assert B["candidates"].astype(str).tolist() == candidates
        with np.load(a.bank_dir/f"{a.dataset}_{part}.npz",allow_pickle=False) as z:losses=1-z["utility"]
        arrays[part+"_record_ids"]=B["record_ids"];arrays[part+"_source_ids"]=B["source_ids"]
        geometry = {}
        outcomes = {}
        for key, config in configs.items():
            ci, kind = config["ci"], config["kind"]
            if kind == "native":
                scores = B["native"][ci]
            elif kind == "constant":
                scores = np.full(len(B["record_ids"]), config["model"])
            else:
                if ci not in geometry:
                    geometry[ci] = apply_geometry(transforms[ci], B, ci)
                model = knn_indices[key] if kind == "knn" else config["model"]
                scores = np.clip(model.predict(geometry[ci]), 0, 1)
            outcomes[key] = value(scores, losses[ci], B["record_ids"], seed, key)
        for i,selected_policy in enumerate(selected):
            key = selected_policy["config"]
            c = configs[key]
            j, o = outcomes[key]
            arrays[f"{part}_p{i:03d}_order"]=o.astype(np.int32)
            arrays[f"{part}_p{i:03d}_loss"]=losses[c["ci"]]
            rows.append({**selected_policy,"cohort":part,"J":j,"errors_or_loss_units_per_1000":1000*j,"status":"EXPLORATORY_POST_REVIEW_EXTENSION"})
    partial = dest / "evaluation_arrays.npz.part"
    with partial.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    partial.replace(dest / "evaluation_arrays.npz")
    save(dest / "results.json", rows)


def run_unit(a, rep, weight=None):
    label = rep if weight is None else f"weighted_TVn_{weight:.2f}"
    dest = a.output / a.task / a.dataset / f"{a.flow}_{a.seed:02d}" / label
    dest.mkdir(parents=True, exist_ok=True)
    development = dest / "D_fitted_selected.joblib"
    complete = dest / "complete.json"
    select_only = getattr(a, "select_only", False)
    with FileLock(str(dest / ".fit.lock")):
        if complete.is_file() or (select_only and development.is_file()):
            return
        bank_path = a.bank_dir / f"{a.dataset}_D.npz"
        D = read_bank(bank_path, True)
        with np.load(HERE / f"splits/{a.dataset}_30_seeds.npz", allow_pickle=False) as split:
            assert np.array_equal(D["record_ids"], split["record_ids"])
            F = np.flatnonzero(split[a.flow + "_fit"][a.seed])
            V = np.flatnonzero(~split[a.flow + "_fit"][a.seed])
            seed = int(split["routing_seeds"][a.seed])
        candidates = D["candidates"].astype(str).tolist()
        assert candidates == candidate_ids(a.dataset), "Input bank does not contain the declared frozen candidate panel"
        identity = dict(task=a.task, dataset=a.dataset, seed=a.seed, flow=a.flow, representation=label,
                        development_bank=str(bank_path.resolve()), routing_seed=seed, candidates=candidates)
        if development.is_file():
            checkpoint = joblib.load(development)
            if checkpoint["identity"] != identity:
                raise ValueError("Development checkpoint differs from this experiment's input bank or design")
        else:
            checkpoint = choose_on_development(a, D, F, V, seed, candidates, rep, label, weight)
            checkpoint["identity"] = identity
            save(dest / "D_selected_policies.json", checkpoint["policies"])
            partial = development.with_suffix(".joblib.part")
            joblib.dump(checkpoint, partial, compress=3)
            partial.replace(development)
            print(json.dumps(dict(unit=str(dest.relative_to(a.output)), policies=len(checkpoint["policies"]),
                                  status="DEVELOPMENT_SELECTION_COMPLETE")), flush=True)
        if select_only:
            return
        evaluate_holdouts(a, dest, D, F, candidates, seed, checkpoint)
        count = len(checkpoint["policies"])
        save(complete, dict(status="UNIT_COMPLETE", task=a.task, dataset=a.dataset, seed=a.seed, flow=a.flow,
                            representation=label, policies=count, development_checkpoint=development.name,
                            scope="F-only fit, V-only selection and post-review exploratory evaluation on H/H2"))
        print(json.dumps(dict(unit=str(dest.relative_to(a.output)), policies=count, status="UNIT_COMPLETE")), flush=True)


def self_test():
    rng=np.random.default_rng(452031)
    text=rng.normal(size=(1,90,13));text[:,:,0]=1.
    native=rng.normal(size=(1,90));visual=rng.normal(size=(90,80));visual[:,3]=1.
    B=dict(text=text,native=native,visual=visual,question_hidden=rng.normal(size=(90,64)))
    F=np.arange(60)
    for rep in ["Tn","TVn","block_TVn","PCA32_TVn","block_TQn","PCA32_TQn"]:
        s,z=fit_geometry(B,0,F,rep,131)
        altered={k:v.copy() for k,v in B.items()};altered["text"][:,60:]+=1e6;altered["native"][:,60:]+=1e6
        altered["visual"][60:]+=1e6;altered["question_hidden"][60:]+=1e6
        _,zz=fit_geometry(altered,0,F,rep,131)
        np.testing.assert_allclose(z[F],zz[F],atol=1e-10,rtol=0)
        np.testing.assert_allclose(apply_geometry(s,B,0),z,atol=1e-10,rtol=0)
    s0,z0=fit_geometry(B,0,F,"TVn",131,0.)
    _,original=fit_geometry(B,0,F,"TVn",131)
    _,one=fit_geometry(B,0,F,"TVn",131,1.)
    _,block=fit_geometry(B,0,F,"block_TVn",131)
    np.testing.assert_allclose(z0,original,atol=1e-12,rtol=0);np.testing.assert_allclose(one,block,atol=1e-12,rtol=0)
    assert eligible("native",None,"ridge_original") and not eligible("native",None,"ridge_only")
    print("SYNTHETIC_SOFTWARE_CHECKS_COMPLETED: F-only transforms, apply consistency, exact weight endpoints and native eligibility. No dataset experiment run.")


def main():
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true");p.add_argument("--gpu-ready",action="store_true")
    p.add_argument("--select-only",action="store_true",help="Fit on F, select on V and save the selected states before opening H or H2")
    p.add_argument("--task",choices=["seed","question","block_weight","strong"]);p.add_argument("--dataset",choices=["vqav2","activitynetqa"])
    p.add_argument("--seed",type=int);p.add_argument("--flow",choices=["record_random","source_disjoint"])
    p.add_argument("--bank-dir",type=Path,default=HERE/"rebuilt/banks");p.add_argument("--output",type=Path,default=HERE/"extension_results")
    p.add_argument("--threads",type=int,default=4);a=p.parse_args()
    with threadpool_limits(limits=a.threads):
        if a.self_test:self_test();return
        if not a.gpu_ready:raise SystemExit("WAITING_USER_GPU: actual extension experiments remain deferred.")
        if any(x is None for x in [a.task,a.dataset,a.seed,a.flow]):raise SystemExit("--task, --dataset, --seed and --flow required")
        if not 0<=a.seed<30:raise SystemExit("Seed outside prepared 0..29")
        if a.task=="block_weight":
            if a.dataset!="activitynetqa":raise SystemExit("Block-weight path is planned for video")
            for w in WEIGHTS:run_unit(a,"TVn",w)
        else:
            for rep in representations(a.task):run_unit(a,rep)


if __name__=="__main__":main()
