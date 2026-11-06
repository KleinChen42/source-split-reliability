"""Publication figures and tables for the representation-invariance extension."""
from pathlib import Path
import argparse
import csv
import gzip
import json
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_figures import setup_plotting, INK, MUTED, GRID, BLUE, AMBER, GREEN

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data/extension"
COHORTS=("H","H2")
NORMALIZATIONS=("coordinate","coordinate_block","energy")
ENCODINGS=("identity","repeat4","rotate")
LABELS=("Coordinate", "Coordinate + block", "Block energy")


def read(name):
    path=DATA/name
    opener=gzip.open if path.suffix==".gz" else open
    with opener(path,"rt",encoding="utf-8",newline="") as f:return list(csv.DictReader(f))


def export(analysis):
    DATA.mkdir(parents=True,exist_ok=True)
    for name in ["points.csv","invariance_summary.csv","encoding_sensitivity.csv",
                 "development_choices.csv","finite_queue_points.csv","cell_intervals.csv",
                 "pairwise_intervals.csv.gz","multiplicity_method.json"]:
        shutil.copyfile(analysis/name,DATA/name)
    for name in ["latency_qwen.csv","latency_internvl.csv","latency_qwen.json","latency_internvl.json"]:
        shutil.copyfile(analysis.parent/"results/cost"/name,DATA/name)


def clean(axis):
    axis.spines[["top","right"]].set_visible(False)
    axis.tick_params(length=3)
    axis.grid(axis="y",color=GRID,lw=.55)
    axis.set_axisbelow(True)


def draw_encodings(intervals,sensitivity):
    fig,axes=plt.subplots(2,2,figsize=(7.05,4.25),sharex="row",sharey="row")
    fig.subplots_adjust(left=.12,right=.985,bottom=.19,top=.91,hspace=.47,wspace=.13)
    for col,cohort in enumerate(COHORTS):
        h="$H$" if cohort=="H" else "$H_2$"
        ax=axes[0,col];clean(ax)
        ax.set_title(f"({chr(97+col)})  Split contrast: {h}",loc="left",fontsize=8.8,fontweight="bold")
        ax.axhline(0,color=MUTED,lw=.8)
        ax.set(xticks=range(3),xticklabels=("Original","Four copies","Rotation"),xlim=(-.22,2.22))
        for offset,norm,label,color,marker in zip((-.06,0,.06),NORMALIZATIONS,LABELS,("#757E87",BLUE,GREEN),("v","s","o")):
            vals=[intervals[cohort,"delta","expanded/"+norm+"__"+enc] for enc in ENCODINGS]
            y=np.array([float(v["estimate_per_1000"]) for v in vals])
            lo=np.array([float(v["simultaneous95_low_per_1000"]) for v in vals])
            hi=np.array([float(v["simultaneous95_high_per_1000"]) for v in vals])
            ax.errorbar(np.arange(3)+offset,y,yerr=[y-lo,hi-y],label=label,color=color,marker=marker,
                        lw=1.05,markersize=3.4,capsize=2.2)
        ax=axes[1,col];clean(ax)
        ax.set_title(f"({chr(99+col)})  Replaced acceptances: {h}",loc="left",fontsize=8.8,fontweight="bold")
        denom=10000 if cohort=="H" else 9950
        for offset,enc,color,label in [(-.17,"repeat4",AMBER,"Four copies"),(.17,"rotate",BLUE,"Rotation")]:
            vals=[np.mean([float(r["changed_accepted_records"])*1000/denom for r in sensitivity
                           if r["cohort"]==cohort and r["normalization"]==norm and r["encoding"]==enc])
                  for norm in NORMALIZATIONS]
            ax.bar(np.arange(3)+offset,vals,width=.29,color=color,label=label,zorder=3)
            for x,y in zip(np.arange(3)+offset,vals):
                value="<0.1" if 0<y<.1 else ("0" if y==0 else f"{y:.1f}")
                ax.annotate(value,(x,y),xytext=(0,3),textcoords="offset points",ha="center",fontsize=7.2)
        ax.set(xticks=range(3),xticklabels=("Coordinate","Coord. + block","Energy"),xlim=(-.45,2.45))
        ax.margins(y=.2)
    axes[0,0].set_ylabel(r"$\Delta$ per 1,000 arrivals",fontsize=8.3)
    axes[1,0].set_ylabel("Replaced answers\nper 1,000 arrivals",fontsize=8.3)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc="upper center",bbox_to_anchor=(.54,1.01),frameon=False,fontsize=7.7)
    handles,labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,loc="lower center",bbox_to_anchor=(.54,.035),frameon=False,fontsize=8)
    fig.savefig(ROOT/"figures/encoding_sensitivity.pdf")
    plt.close(fig)


def draw_deployment(points,queues):
    index={(r["cohort"],r["panel"],float(r["capacity"])):r
           for r in points if r["scope"]=="ALL"}
    qi={(r["cohort"],r["menu"],int(r["sources_per_queue"])):r for r in queues}
    capacities=(.1,.3,.5,.7,.9)
    fig,axes=plt.subplots(2,2,figsize=(7.05,4.2),sharey=True)
    fig.subplots_adjust(left=.12,right=.985,bottom=.19,top=.91,hspace=.54,wspace=.13)
    for col,cohort in enumerate(COHORTS):
        h="$H$" if cohort=="H" else "$H_2$"
        ax=axes[0,col];clean(ax)
        ax.set_title(f"({chr(97+col)})  Acceptance capacity: {h}",loc="left",fontsize=8.8,fontweight="bold")
        ax.axhline(0,color=MUTED,lw=.75)
        ax.set(xticks=capacities,xticklabels=(".1",".3",".5",".7",".9"),xlabel="Acceptance fraction")
        for menu,color,marker,label in [("combined_visual",BLUE,"o","Visual menu"),("combined_question",GREEN,"D","Question menu")]:
            for selection,line in [("matched","-"),("frozen70","--")]:
                gain=[1000*(float(index[cohort,"matched/native",q]["J_S"])-
                            float(index[cohort,selection+"/"+menu,q]["J_S"])) for q in capacities]
                ax.plot(capacities,gain,color=color,marker=marker,lw=1.2,markersize=3.3,
                        linestyle=line,markerfacecolor=color if line=="-" else "white",
                        label=label+(", matched" if line=="-" else ", selected at .7"))
        ax=axes[1,col];clean(ax)
        ax.set_title(f"({chr(99+col)})  Finite queues at 70%: {h}",loc="left",fontsize=8.8,fontweight="bold")
        ax.axhline(0,color=MUTED,lw=.75)
        sizes=(10,50,100,1000)
        for menu,color,marker in [("combined_visual",BLUE,"o"),("combined_question",GREEN,"D")]:
            gain=[1000*(float(qi[cohort,"native",size]["J"])-float(qi[cohort,menu,size]["J"])) for size in sizes]
            ax.plot(range(4),gain,color=color,marker=marker,lw=1.2,markersize=3.6)
        ax.set(xticks=range(4),xticklabels=("10","50","100","Full cohort"),xlabel="Videos per queue")
    fig.supylabel("Reduction from native selection (errors per 1,000 arrivals)",x=.012,y=.58,fontsize=8.4)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,loc="lower center",bbox_to_anchor=(.54,.005),frameon=False,fontsize=7.8)
    fig.savefig(ROOT/"figures/deployment_replay.pdf")
    plt.close(fig)


def write_tables(intervals):
    panels=[("matched/native","Native only"),("matched/trees",r"Trees, $\T+\Nfeat$"),
        ("expanded/coordinate_block__identity",r"Coordinate-balanced \tvn"),
        ("expanded/energy__identity",r"Energy-normalized \tvn"),
        ("matched/question_coordinate",r"Coordinate-balanced \tqn"),
        ("matched/question_energy",r"Energy-normalized \tqn"),
        ("matched/combined_question","Combined question menu")]
    lines=["% Generated from the new 720-endpoint interval family."]
    for cohort in COHORTS:
        for k,(panel,label) in enumerate(panels):
            r,s,d=[intervals[cohort,e,panel] for e in ("J_R","J_S","delta")]
            h=("$H$" if cohort=="H" else "$H_2$") if k==0 else ""
            vals=[float(x["estimate_per_1000"]) for x in (r,s,d)]
            lo,hi=[float(d[x]) for x in ("simultaneous95_low_per_1000","simultaneous95_high_per_1000")]
            lines.append(f"{h} & {label} & {vals[0]:.2f} & {vals[1]:.2f} & ${vals[2]:.2f}$ & $[{lo:.2f},{hi:.2f}]$"+r" \\")
        if cohort=="H":lines.append(r"\addlinespace[3pt]")
    (ROOT/"sections/extension_policy_rows.tex").write_text("\n".join(lines)+"\n")
    latency=read("latency_qwen.csv")+read("latency_internvl.csv")
    names=[("qwen","answer_native","direct_short","Qwen, direct"),
           ("qwen","answer_native_and_Q","direct_short","Qwen, direct + state capture"),
           ("qwen","answer_native","task_constrained_short","Qwen, constrained"),
           ("internvl","answer_native","direct_short","InternVL, direct"),
           ("internvl","answer_native","task_constrained_short","InternVL, constrained"),
           ("qwen","separate_Q_pass","direct_short","Separate Qwen state pass"),
           ("resnet50","visual_per_source","none","ResNet, eight frames/video")]
    lines=["% GPU-synchronized timings from 64 real questions/sources."]
    for model,mode,prompt,label in names:
        v=np.array([float(r["seconds"])*1000 for r in latency if (r["model"],r["mode"],r["prompt"])==(model,mode,prompt)])
        if len(v)!=64:raise ValueError(f"Missing latency rows: {label}, {len(v)}")
        lines.append(f"{label} & {v.mean():.1f} & {np.median(v):.1f} & {np.quantile(v,.95):.1f}"+r" \\")
    (ROOT/"sections/latency_rows.tex").write_text("\n".join(lines)+"\n")


def draw_summary(intervals):
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.25),sharey=True)
    fig.subplots_adjust(left=.11,right=.985,bottom=.2,top=.8,wspace=.15)
    for ax,cohort in zip(axes,COHORTS):
        clean(ax)
        ax.axhline(0,color=MUTED,lw=.7)
        ax.set_title("Original cohort $H$" if cohort=="H" else "Additional cohort $H_2$",loc="left",fontsize=9)
        for offset,norm,label,color,marker in zip((-.04,.04),("coordinate","energy"),
                ("Coordinate standardization","Block energy"),("#757E87",GREEN),("v","o")):
            rows=[intervals[cohort,"delta","expanded/"+norm+"__"+e] for e in ENCODINGS]
            y=np.array([float(r["estimate_per_1000"]) for r in rows])
            lo=np.array([float(r["simultaneous95_low_per_1000"]) for r in rows])
            hi=np.array([float(r["simultaneous95_high_per_1000"]) for r in rows])
            ax.errorbar(np.arange(3)+offset,y,yerr=[y-lo,hi-y],label=label,color=color,
                        marker=marker,ms=4,lw=1.3,capsize=3)
        ax.set(xticks=range(3),xticklabels=("Original","Four copies","Rotation"),xlim=(-.2,2.2))
    axes[0].set_ylabel(r"$\Delta$ per 1,000 arrivals",fontsize=9)
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,ncol=2,loc="upper center",bbox_to_anchor=(.55,1.03),frameon=False,fontsize=8.5)
    fig.savefig(ROOT/"figures/graphical_summary.pdf")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir",type=Path)
    a=parser.parse_args()
    (ROOT/"figures").mkdir(exist_ok=True)
    (ROOT/"sections").mkdir(exist_ok=True)
    if a.analysis_dir:export(a.analysis_dir)
    setup_plotting()
    intervals={(r["cohort"],r["endpoint"],r["left"]):r for r in read("cell_intervals.csv")}
    draw_encodings(intervals,read("encoding_sensitivity.csv"))
    draw_deployment(read("points.csv"),read("finite_queue_points.csv"))
    draw_summary(intervals)
    write_tables(intervals)


if __name__=="__main__":main()
