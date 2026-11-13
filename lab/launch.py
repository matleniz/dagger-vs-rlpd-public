"""Lance des runs détachés (ils survivent à la fermeture du terminal) et les inscrit au registre.

    python -m lab.launch rlpd --seeds 0-4 -- --name base --demos data/team4_lighthouse_n0.0.npz

Un processus par graine. Sortie console : ``runs/logs/<méthode>_<name>_seed<k>.log``.
Chaque lancement ajoute une ligne à ``runs/jobs.csv`` ; l'état se lit avec ``python -m lab.registry``.
"""

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

from .metrics import CsvLog

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=("dagger", "rlpd"))
    ap.add_argument("--seeds", default="0-4", help="ex. 0-4 ou 7")
    ap.add_argument("--wait", action="store_true", help="attend la fin des runs (pour enchaîner des lots)")
    a, args = ap.parse_known_args()  # le reste : arguments de la méthode
    args = [x for x in args if x != "--"]
    name = args[args.index("--name") + 1] if "--name" in args else "base"
    first, _, last = a.seeds.partition("-")
    Path("runs/logs").mkdir(parents=True, exist_ok=True)
    jobs = Path("runs/jobs.csv")
    procs = []
    for seed in range(int(first), int(last or first) + 1):
        log = f"runs/logs/{a.method}_{name}_seed{seed}.log"
        cmd = [sys.executable, "-u", "-m", f"lab.{a.method}", "--seed", str(seed), *args]
        p = subprocess.Popen(cmd, stdout=open(log, "a"), stderr=subprocess.STDOUT, start_new_session=True)
        row = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), host=socket.gethostname(), pid=p.pid,
                   run=f"runs/{a.method}/{name}/seed{seed}", log=log, command=" ".join(cmd[2:]))
        with open(jobs, "a") as f:
            if f.tell() == 0:
                f.write(",".join(row) + "\n")
            f.write(",".join(str(v) for v in row.values()) + "\n")
        procs.append(p)
        print(f"lancé pid {p.pid} -> {row['run']}", flush=True)
    if a.wait:
        for p in procs:
            p.wait()
