from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def line_plot(df, y, ylabel, title, path):
    agg=df.groupby(["method","batch"],as_index=False).agg(mean=(y,"mean"),std=(y,"std"))
    plt.figure(figsize=(9,5))
    for method,g in agg.groupby("method"):
        plt.plot(g.batch,g["mean"],label=method)
        sd=g["std"].fillna(0)
        plt.fill_between(g.batch,g["mean"]-sd,g["mean"]+sd,alpha=.15)
    plt.xlabel("Stream batch"); plt.ylabel(ylabel); plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(path,dpi=200); plt.close()


def main():
    p=Path("results/repeated"); df=pd.read_csv(p/"all_runs.csv"); stream=df[df.batch>=0]
    line_plot(stream,"test_accuracy","Test accuracy","Accuracy across the stream (mean ± SD)",p/"accuracy_mean_std.png")
    line_plot(stream,"pseudo_precision","Pseudo-label precision","Pseudo-label quality",p/"pseudo_precision.png")
    line_plot(stream,"pseudo_coverage","Pseudo-label coverage","Accepted fraction of unlabelled samples",p/"pseudo_coverage.png")
    line_plot(stream,"parameter_change","Relative parameter change","Model change after each batch",p/"parameter_change.png")
    line_plot(stream,"feature_centroid_drift","Feature-centroid drift","Representation drift from trusted semantics",p/"feature_centroid_drift.png")
    dist=pd.read_csv(next(p.glob("distribution_seed*.csv")))
    plt.figure(figsize=(9,4)); plt.plot(dist.batch,dist.total_variation_from_previous,marker="o")
    plt.xlabel("Stream batch"); plt.ylabel("Total variation distance"); plt.title("Class-distribution change"); plt.tight_layout(); plt.savefig(p/"distribution_change.png",dpi=200); plt.close()
    print("Saved plots in",p)

if __name__=="__main__": main()
