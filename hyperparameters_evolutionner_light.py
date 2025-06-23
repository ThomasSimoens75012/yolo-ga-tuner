import random, logging, shutil, yaml
from pathlib import Path
from datetime import datetime
import pandas as pd
from ultralytics import YOLO

# Configuration
GENERATIONS = 15
POPULATION_SIZE = 12
CARRYING_CAPACITY = 20  # population cible (non stricte)
FIXED_EPOCHS = 40

EXPERIMENT_NAME = "exp_001"
EXP_DIR = Path("experiments") / EXPERIMENT_NAME
EXP_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = EXP_DIR / "summary.csv"

SELECTION_METRIC = "mAP_50:95"
MUTATION_PROB = 0.80
MUTATION_STRENGTH = 0.30

HYP_PARAMS = {
    "lr0": {"min": 0.001, "max": 0.1, "type": "float"},
    "batch": {"min": 2, "max": 4, "type": "int"},
    "positive_ratio": {"min": 0.1, "max": 0.9, "type": "float"},
    "dataset_size": {"min": 200, "max": 6000, "type": "int"},
    "imgsz": {"min": 320, "max": 960, "type": "int", "quant": 32},
}
FIXED_PARAMS = {"model": "yolov8n.pt", "epochs": FIXED_EPOCHS}

IMG_ROOT = Path("unified/images")
LBL_ROOT = Path("unified/labels")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("GA")

def _q(v, q): return int(round(v/q)*q) if q else v

class Hyperparam:
    def __init__(self, name, meta, value=None):
        self.name, self.meta = name, meta
        self.value = value if value is not None else self.random()

    def random(self):
        lo, hi, typ = self.meta["min"], self.meta["max"], self.meta["type"]
        v = random.uniform(lo, hi)
        if typ == "int":
            v = int(round(v))
            v = _q(v, self.meta.get("quant"))
        return v

    def mutate(self):
        if random.random() >= MUTATION_PROB:
            return None
        span = self.meta["max"] - self.meta["min"]
        delta = (random.random()*2 - 1) * MUTATION_STRENGTH * span
        nv = self.value + delta
        if self.meta["type"] == "int":
            nv = int(round(nv))
            nv = _q(nv, self.meta.get("quant"))
        nv = max(self.meta["min"], min(self.meta["max"], nv))
        before, self.value = self.value, nv
        return {"name": self.name, "before": before, "after": nv}

    @staticmethod
    def crossover(a, b):
        return a.value if random.random() < 0.5 else b.value

def rand_name():
    colors = "red blue green golden silver".split()
    adjs = "brave sneaky swift lucky tiny giant".split()
    animals = "fox owl bear wolf tiger panda hawk".split()
    return f"{random.choice(colors)}-{random.choice(adjs)}-{random.choice(animals)}"

def fitness(p, r, m):
    eps = 1e-6
    p, r, m = max(p or 0, eps), max(r or 0, eps), max(m or 0, eps)
    f1 = 2*p*r/(p+r)
    return f1*m*max(0, 1-abs(p-r))

_ds_cache = {}
def generate_dataset(size, pos_ratio, out_dir: Path):
    key = (size, pos_ratio)
    if key in _ds_cache:
        return _ds_cache[key]
    pos = list(IMG_ROOT.glob("moustiques_*.jpg"))
    neg = list(IMG_ROOT.glob("negatifs_*.jpg"))
    random.shuffle(pos); random.shuffle(neg)
    n_pos = int(size * pos_ratio); n_neg = size - n_pos
    splits = {"train": lambda l: l[:int(.8*len(l))], "val": lambda l: l[int(.8*len(l)):]}

    for split, sel in splits.items():
        for img in sel(pos[:n_pos]) + sel(neg[:n_neg]):
            dst_img = out_dir/"images"/split/img.name
            dst_lbl = out_dir/"labels"/split/f"{img.stem}.txt"
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img, dst_img)
            lbl_src = LBL_ROOT/f"{img.stem}.txt"
            shutil.copy(lbl_src, dst_lbl) if lbl_src.exists() else dst_lbl.write_text("")
    yaml.safe_dump({"path": str(out_dir), "train": "images/train", "val": "images/val",
                    "nc": 1, "names": ["moustiques"]},
                   open(out_dir/"data.yaml", "w"))
    _ds_cache[key] = out_dir
    return out_dir

SUMMARY_COLS = list(HYP_PARAMS.keys()) + [
    "run_name", "model", "epochs", "run_id", "generation",
    "precision", "recall", "mAP_50:95", "fitness_score",
    "timestamp", "parent1", "parent2", "mutated", "mutation_details"
]
summary = pd.DataFrame(columns=SUMMARY_COLS)

def save_row(row: dict):
    global summary
    row_clean = {k: ("" if v is None else v) for k, v in row.items()}
    mask = (summary.run_id == row_clean["run_id"]) & (summary.generation == row_clean["generation"])
    summary = summary[~mask]
    summary = pd.concat([summary, pd.DataFrame([row_clean])], ignore_index=True)
    summary.to_csv(SUMMARY_PATH, index=False, na_rep="")

def run_trial(params: dict, run_id: int, gen: int):
    run_name = params.get("run_name") or rand_name()
    exp_dir = EXP_DIR / f"run_{run_id:04d}_{run_name}"
    ds_dir = exp_dir / "dataset"
    ds_dir.mkdir(parents=True, exist_ok=True)
    generate_dataset(params["dataset_size"], params["positive_ratio"], ds_dir)

    try:
        YOLO(params["model"]).train(
            data=str(ds_dir/"data.yaml"),
            epochs=params["epochs"],
            imgsz=params["imgsz"],
            batch=params["batch"],
            lr0=params["lr0"],
            project=str(exp_dir), name="yolo",
            exist_ok=True, verbose=False)
    except Exception as e:
        log.error("YOLO training failed: %s", e)
        return None

    res_csv = exp_dir/"yolo"/"results.csv"
    if not res_csv.exists():
        log.warning("results.csv manquant")
        return None
    df = pd.read_csv(res_csv).iloc[-1]
    p, r, m = df["metrics/precision(B)"], df["metrics/recall(B)"], df["metrics/mAP50-95(B)"]

    return {
        **{k: params[k] for k in HYP_PARAMS},
        "run_name": run_name, "model": params["model"], "epochs": params["epochs"],
        "run_id": run_id, "generation": gen,
        "precision": p, "recall": r, "mAP_50:95": m, "fitness_score": fitness(p, r, m),
        "timestamp": datetime.now().isoformat(),
        "parent1": params.get("parent1"), "parent2": params.get("parent2"),
        "mutated": params.get("mutated", False), "mutation_details": params.get("mutation_details", {})
    }

def evaluate(pop, gen, rid_start):
    results = []; rid = rid_start
    for indiv in pop:
        res = run_trial({**indiv, **FIXED_PARAMS}, rid, gen); rid += 1
        if res:
            save_row(res); results.append(res)
    return results, rid

def reproduce(parents):
    total_fitness = sum(p[SELECTION_METRIC] for p in parents) or 1e-6
    budget = CARRYING_CAPACITY
    kids = []

    for p in parents:
        expected_kids = budget * (p[SELECTION_METRIC] / total_fitness)
        num_kids = int(round(expected_kids))

        for _ in range(num_kids):
            mates = [q for q in parents if q["run_id"] != p["run_id"]]
            if not mates:
                continue
            mate = random.choices(mates, weights=[q[SELECTION_METRIC] for q in mates])[0]
            child, muts = {}, {}

            for name, meta in HYP_PARAMS.items():
                base = Hyperparam.crossover(Hyperparam(name, meta, p[name]), Hyperparam(name, meta, mate[name]))
                hp = Hyperparam(name, meta, base)
                rec = hp.mutate()
                child[name] = hp.value
                if rec:
                    muts[rec["name"]] = {"before": rec["before"], "after": rec["after"]}

            child.update(run_name=rand_name(), parent1=p["run_id"], parent2=mate["run_id"],
                         mutated=bool(muts), mutation_details=muts)
            kids.append(child)

    log.info("%d enfants générés (cible = %d)", len(kids), budget)
    return kids

def main():
    random.seed()
    run_id = 0
    pop = [{n: Hyperparam(n, m).value for n, m in HYP_PARAMS.items()} |
           dict(run_name=rand_name()) for _ in range(POPULATION_SIZE)]

    for gen in range(GENERATIONS):
        log.info("=== Génération %d ===", gen)
        evaluated, run_id = evaluate(pop, gen, run_id)
        if not evaluated:
            break
        top_parents = sorted(evaluated, key=lambda r: r[SELECTION_METRIC], reverse=True)
        pop = reproduce(top_parents)

    if not summary.empty:
        best = summary.sort_values(SELECTION_METRIC, ascending=False).iloc[0]
        log.info("BEST run_id=%s  %s=%.4f", best.run_id, SELECTION_METRIC, best[SELECTION_METRIC])

if __name__ == "__main__":
    main()
