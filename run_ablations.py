import subprocess, sys

# Component ablation on identical seed and stream.
methods=["naive","centroid","centroid_ewc","centroid_replay","centroid_replay_ewc"]
subprocess.run([sys.executable,"run_repeated.py","--methods",*methods,
                "--seeds","0","1","2","3","4","--output-dir","results/ablations"],check=True)
