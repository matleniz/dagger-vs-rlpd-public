"""Push a detached Kaggle kernel that trains some seeds of one configuration.

    python scripts/kaggle_push.py <slug> <method> <seeds> -- <method arguments>
    python scripts/kaggle_push.py rlpd-t4-lh-s01 rlpd 0-1 -- --name t4_lighthouse_kaggle --demos data/team4_lighthouse_n0.0.npz

The kernel embeds lab/ and stk_actor/, reads data/*.npz from the private dataset
matleniz/dagger-vs-rlpd-demos (a copy of the local data/ folder, to update when demos change), writes the
same runs/<method>/<name>/seed<k> folders as a local run, and is fetched back with
``kaggle kernels output matleniz/<slug> -p <dir>``. Run from the repository root.
"""

import base64
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

KERNEL = r'''
import base64, io, os, subprocess, sys, tarfile, time
SLUG, METHOD, SEEDS, ARGS, COMMIT = {slug!r}, {method!r}, {seeds!r}, {args!r}, {commit!r}
print("cpus", os.cpu_count(), sys.version, flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pystk2-gymnasium==0.10.0"], check=True)
os.makedirs("/tmp/repo", exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD))).extractall("/tmp/repo")
import glob, shutil
os.makedirs("/tmp/repo/data", exist_ok=True)
for f in glob.glob("/kaggle/input/**/*.npz", recursive=True):
    shutil.copy(f, "/tmp/repo/data/")
print(len(os.listdir("/tmp/repo/data")), "demo files", flush=True)
os.makedirs("/kaggle/working/runs/logs", exist_ok=True)
os.symlink("/kaggle/working/runs", "/tmp/repo/runs")
os.chdir("/tmp/repo")
open("/kaggle/working/runs/logs/" + SLUG + "_commit.txt", "w").write(COMMIT)
# one env first, alone: downloads the STK data once before the seeds start in parallel
subprocess.run([sys.executable, "-c", "from lab.envs import make_env; e = make_env('lighthouse'); e.reset(seed=0); e.close()"], check=True)
procs = []
for seed in SEEDS:
    log = open(f"runs/logs/{{SLUG}}_seed{{seed}}.log", "w")
    procs.append(subprocess.Popen([sys.executable, "-u", "-m", "lab." + METHOD, "--seed", str(seed), *ARGS], stdout=log, stderr=subprocess.STDOUT))
t0 = time.time()
while any(p.poll() is None for p in procs):
    time.sleep(600)
    print(round(time.time() - t0), "s", [p.poll() for p in procs], flush=True)
for seed in SEEDS:
    print("---- seed", seed, "\n" + "".join(open(f"runs/logs/{{SLUG}}_seed{{seed}}.log").readlines()[-8:]), flush=True)
sys.exit(max(abs(p.returncode) for p in procs))
'''

if __name__ == "__main__":
    slug, method, seeds = sys.argv[1:4]
    args = [x for x in sys.argv[4:] if x != "--"]
    first, _, last = seeds.partition("-")
    seeds = list(range(int(first), int(last or first) + 1))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in [*Path("lab").glob("*.py"), *Path("stk_actor").glob("*.py")]:
            tar.add(p)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = Path("runs/logs/kaggle") / slug  # ignored by git
    out.mkdir(parents=True, exist_ok=True)
    code = KERNEL.format(slug=slug, method=method, seeds=seeds, args=args, commit=commit)
    (out / f"{slug}.py").write_text(f'PAYLOAD = "{base64.b64encode(buf.getvalue()).decode()}"\n' + code)
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"matleniz/{slug}", "title": slug, "code_file": f"{slug}.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": True,
        "dataset_sources": ["matleniz/dagger-vs-rlpd-demos"], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=1))
    subprocess.run(["kaggle", "kernels", "push", "-p", str(out)], check=True)
