
# ===== cell 1
import os
import h5py
import json
import random
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sksurv.metrics import concordance_index_censored
import sys

# ===== cell 2
BASE_DIR = Path("/loctmp/mue37538/dataset/data-transfer-student")

# ===== cell 3
def _get_patch_count(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as f:
        return f["features"].shape[0]

def build_wsi_cohort(cohort_dir: Path, required_slides: set = None) -> pd.DataFrame:
    """Identisch zu SURVPATH.ipynb. Die WSI-Features werden hier NICHT benutzt, aber die Patientenauswahl
    (nur Patienten mit WSI + RNA, 1 Slide pro Patient) bleibt gleich -> dieselben Patienten und Folds wie bei
    den WSI-/multimodalen Modellen."""
    df_clin = pd.read_csv(
        cohort_dir / "clinical/clinical_survival_per_slide.csv")
    wsi_dir = cohort_dir / "wsi_features/conch_v15"
    available_wsi = {p.stem for p in wsi_dir.glob("*.h5")}

    df_clin = df_clin.dropna(subset=['survival_months', 'censorship']).copy()
    df_clin = df_clin[df_clin['survival_months'] > 0].copy()
    df_clin['slide_id'] = df_clin['slide_id'].str.replace(
        '.svs', '', regex=False)
    df_clin = df_clin[df_clin["slide_id"].isin(available_wsi)].copy()
    if required_slides is not None:  # nur Slides, die auch Omics-Daten haben
        df_clin = df_clin[df_clin["slide_id"].isin(required_slides)].copy()

    df_clin["event"] = 1 - df_clin["censorship"] # 1 = Todesfall, 0 = zensiert

    dx_extracted = df_clin["slide_id"].str.extract(r"-DX(\d+)\.", expand=False)
    df_clin["dx_num"] = dx_extracted.fillna(999).astype(int)

    df_clin["n_patches"] = [
        _get_patch_count(wsi_dir / f"{s}.h5") for s in df_clin["slide_id"]
    ]

    df_clin = (
        df_clin.sort_values(by=["case_id", "dx_num", "n_patches"], ascending=[
                            True, True, False])
        .drop_duplicates(subset=["case_id"], keep="first")
        # da wir nur einen Slide pro Patient wollen, wird hier nach case_id gruppiert und der Slide mit der niedrigsten DX-Nummer und den meisten Patches behalten
        .reset_index(drop=True)
    )
    return df_clin

# ===== cell 4
def compute_fold_time_bins(train_df: pd.DataFrame, val_df: pd.DataFrame, n_bins: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Berechnet die Quartil-Grenzen NUR auf den Todesfällen von train_df (verhindert Data Leakage)
    und wendet sie auf train_df und val_df an.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()

    uncensored_train = train_df.loc[train_df["event"] == 1, "survival_months"]
    _, q_bins = pd.qcut(uncensored_train, q=n_bins, retbins=True, labels=False)

    q_bins[0] = -np.inf
    q_bins[-1] = np.inf

    train_df["time_bin"] = pd.cut(
        train_df["survival_months"], bins=q_bins, labels=False, include_lowest=True
    ).astype(int)
    val_df["time_bin"] = pd.cut(
        val_df["survival_months"], bins=q_bins, labels=False, include_lowest=True
    ).astype(int)

    return train_df, val_df

# ===== cell 5
def cross_validate(df: pd.DataFrame, n_splits: int = 5, n_bins: int = 4, min_events_per_fold: int = 5) -> pd.DataFrame:
    best_split = None
    best_score = float("inf")

    # Berechnet die Zeitbins für die Stratifizierung
    uncensored_times = df.loc[df["event"] == 1, "survival_months"]
    _, q_bins = pd.qcut(uncensored_times, q=n_bins, retbins=True, labels=False)
    q_bins[0] = -np.inf
    q_bins[-1] = np.inf
    strat_bins = pd.cut(df["survival_months"], bins=q_bins, labels=False, include_lowest=True)
    strat_target = np.where(df["event"] == 1, strat_bins + 1, 0)

    # testen von verschiedenen Seeds für die Greedy-Gruppenzuweisung
    for seed in range(1000):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        folds = np.full(len(df), -1)
        valid_split = True
        event_counts = []
        fold_sizes = []

        for fold, (_, val_idx) in enumerate(sgkf.split(X=df, y=strat_target, groups=df["site"])):
            folds[val_idx] = fold
            n_events = df.iloc[val_idx]["event"].sum()
            if n_events < min_events_per_fold:
                valid_split = False
                break
            event_counts.append(n_events)
            fold_sizes.append(len(val_idx))

        if valid_split:
            score = np.std(event_counts) * 2.0 + np.std(fold_sizes) * 0.5
            if score < best_score:
                best_score = score
                best_split = folds.copy()

    if best_split is None:
        raise RuntimeError(
            f"Kein gültiger Split mit mind. {min_events_per_fold} Events pro Fold gefunden! "
            "Eine einzelne 'site' dominiert die Kohorte zu stark."
        )

    df["fold"] = best_split

    print("Fold-Verteilung (Größe | Events):")
    for f in range(n_splits):
        sub = df[df["fold"] == f]
        print(f"  Fold {f}: n={len(sub):3d} | events={int(sub['event'].sum()):2d} | sites={sub['site'].nunique()}")

    return df

# ===== cell 6
RNA_SUFFIX = "_rnaseq"

def load_pathways(pathway_csv: Path) -> dict:
    """SurvPath-Pathways (Hallmarks + Reactome, 'combine'): Gen x Pathway 0/1-Matrix -> {pathway: [gene, ...]}."""
    m = pd.read_csv(pathway_csv, index_col=0)
    return {pw: m.index[m[pw] == 1].tolist() for pw in m.columns}


def load_rna(cohort_dir: Path, genes: list) -> pd.DataFrame:
    """Rohe Xena-RNA-seq-Matrix (log2(norm count+1), pan-cancer normalisiert) -> eine Zeile pro Sample
    (15 Zeichen Barcode), eingeschraenkt auf die Pathway-Gene. Spalten: <GEN>_rnaseq."""
    files = sorted((cohort_dir / "genomics/raw").glob("HiSeqV2_PANCAN_*.tsv"))  # GBMLGG: _GBM und _LGG
    rna = pd.concat([pd.read_csv(f, sep="\t", index_col=0) for f in files], axis=1)
    rna = rna[~rna.index.duplicated()]
    rna = rna.loc[rna.index.intersection(genes)].T.astype(np.float32)  # [samples, genes]
    rna.columns = [g + RNA_SUFFIX for g in rna.columns]
    rna.index.name = "rna_sample"
    return rna


def get_omic_columns(columns, pathways: dict) -> list:
    """Genomic-only: jedes Pathway-Gen genau EINMAL (sortiert) als ein flacher Feature-Vektor.
    Gleiche Genmenge wie bei SurvPath (dort pro Pathway gruppiert, Gene in mehreren Pathways mehrfach);
    das entspricht dem 'omics'/'snn'-Modus im SurvPath-Repo."""
    columns = set(columns)
    return sorted({g + RNA_SUFFIX for genes in pathways.values() for g in genes if g + RNA_SUFFIX in columns})


def build_dataset(cohort_dir: Path, pathways: dict, n_bins: int = 4, n_splits: int = 5) -> pd.DataFrame:
    all_genes = sorted({g for genes in pathways.values() for g in genes})
    rna = load_rna(cohort_dir, all_genes)
    clin = pd.read_csv(cohort_dir / "clinical/clinical_survival_per_slide.csv", usecols=["slide_id"])
    clin_slides = clin["slide_id"].str.replace(".svs", "", regex=False)
    required = set(clin_slides[clin_slides.str[:15].isin(rna.index)])  # nur Slides, deren Sample RNA-Daten hat
    df_clin = build_wsi_cohort(cohort_dir, required_slides=required)
    df_clin["rna_sample"] = df_clin["slide_id"].str[:15]
    df_clin = df_clin.merge(rna.reset_index(), on="rna_sample", how="left", validate="many_to_one")
    df_clin = cross_validate(df_clin, n_splits=n_splits, n_bins=n_bins)
    return df_clin

# ===== cell 7
def scale_omics(train_df: pd.DataFrame, val_df: pd.DataFrame, omic_cols: list, clip: float = 5.0):
    """Z-Standardisierung PRO GEN (StandardScaler), NUR auf dem Trainings-Fold gefittet (kein Leakage),
    danach auf [-clip, clip] begrenzt (wie SURVPATH.ipynb). Gibt float32-Arrays [n_samples, n_genes] zurueck."""
    scaler = StandardScaler().fit(train_df[omic_cols].values)
    out = []
    for df in (train_df, val_df):
        out.append(np.clip(scaler.transform(df[omic_cols].values), -clip, clip).astype(np.float32))
    return out[0], out[1]


class SurvivalOmicsDataset(Dataset):
    """Wie SurvivalWSIDataset, aber ohne WSI: gibt (omics [n_genes], target [time_bin, censorship], surv, cens) zurueck."""
    def __init__(self, df: pd.DataFrame, omic_array: np.ndarray, device: torch.device = torch.device("cpu")):
        self.df = df.reset_index(drop=True)
        self.omics = torch.from_numpy(omic_array).to(device)
        self.targets = torch.tensor(self.df[['time_bin', 'censorship']].values, dtype=torch.long)
        self.surv = torch.tensor(self.df['survival_months'].values, dtype=torch.float32)
        self.cens = torch.tensor(self.df['censorship'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.omics[idx], self.targets[idx], self.surv[idx], self.cens[idx]

# ===== cell 8
class NLLSurvLoss(nn.Module):
    """Discrete-Time Survival Loss (Negative Log-Likelihood), identisch zu SURVPATH.ipynb."""
    def __init__(self, alpha: float = 0.15):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # target: [B, 2] -> Spalte 0 = time_bin, Spalte 1 = censorship (1 = zensiert, 0 = Event)
        y = target[:, 0].view(-1, 1)
        c = target[:, 1].view(-1, 1).float()

        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        S_padded = torch.cat([torch.ones_like(c), S], dim=1)

        s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=1e-7)
        h_this = torch.gather(hazards, dim=1, index=y).clamp(min=1e-7)
        s_this = torch.gather(S_padded, dim=1, index=y + 1).clamp(min=1e-7)

        uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
        censored_loss = -c * torch.log(s_this)

        loss = (1.0 - self.alpha) * censored_loss + uncensored_loss
        return loss.mean()

# ===== cell 9
# Original-SNNOmics (Klambauer et al. 2017, wie im SurvPath-Repo) aus dem lokalen Repo
SURVPATH_REPO = "/loctmp/mue37538/models/SURVPATH/SurvPath"
if SURVPATH_REPO not in sys.path:
    sys.path.insert(0, SURVPATH_REPO)
from models.model_SNNOmics import SNNOmics


class SNN(nn.Module):
    """Wrapper um SNNOmics mit der Schnittstelle forward(omics, loss_fn, label) -> (results_dict, None).
    omics: [B, n_genes]. Architektur: 'small' = [n_genes -> 256 -> 256] je Linear + ELU + AlphaDropout(0.25),
    danach Linear(256 -> num_classes)."""
    def __init__(self, omic_input_dim, num_classes=4, model_size_omic="small"):
        super().__init__()
        self.model = SNNOmics(omic_input_dim=omic_input_dim, model_size_omic=model_size_omic, n_classes=num_classes)

    def forward(self, omics, loss_fn=None, label=None):
        logits = self.model(data_omics=omics.float())  # [B, num_classes]
        results_dict = {"logits": logits}
        S = torch.cumprod(1 - torch.sigmoid(logits), dim=1)
        results_dict["risk"] = -S.sum(dim=1).detach()  # hoeher = hoeheres Risiko
        if loss_fn is not None and label is not None:
            results_dict["loss"] = loss_fn(logits, label)
        return results_dict, None

# ===== cell 10
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Ein Optimizer-Schritt pro Batch. Mit batch_size = grad_accum_steps (32) ist das mathematisch identisch zur
    Gradient Accumulation mit batch_size=1 in SURVPATH.ipynb (Loss = Mittel ueber 32 Samples, letzter Rest-Batch
    = Mittel ueber den Rest). Keine BatchNorm im Modell -> keine Abhaengigkeit von der Batchgroesse."""
    model.train()
    total_loss = torch.zeros((), device=device)
    n_samples = 0

    for omics, target, _, _ in loader:
        omics = omics.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        results_dict, _ = model(omics=omics, loss_fn=loss_fn, label=target)
        results_dict['loss'].backward()
        optimizer.step()

        total_loss += results_dict['loss'].detach() * omics.size(0)
        n_samples += omics.size(0)

    return total_loss.item() / n_samples  # mittlerer Loss pro Sample (wie in SURVPATH.ipynb)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = torch.zeros((), device=device)
    n_samples = 0
    all_risk_scores, all_censorships, all_event_times = [], [], []

    for omics, target, surv_months, censorship in loader:
        omics = omics.to(device)
        target = target.to(device)

        results_dict, _ = model(omics=omics, loss_fn=loss_fn, label=target)
        total_loss += results_dict['loss'].detach() * omics.size(0)
        n_samples += omics.size(0)

        all_risk_scores.append(results_dict['risk'])
        all_censorships.extend(censorship.numpy())
        all_event_times.extend(surv_months.numpy())

    all_risk_scores = torch.cat(all_risk_scores).cpu().numpy()

    # sksurv erwartet event_indicator = True für Todesfall (1 - censorship)
    event_indicators = (1 - np.array(all_censorships)).astype(bool)
    c_index = concordance_index_censored(event_indicators, np.array(all_event_times), all_risk_scores)[0]

    return total_loss.item() / n_samples, c_index, all_risk_scores

# ===== cell 11
class ExperimentLogger:
    """Identisches Ausgabeformat wie in SURVPATH.ipynb (config.json, dataset_splits.csv, metrics_per_epoch.csv,
    oof_predictions.csv, summary.json, runs/experiment_summary.csv) -> direkt in evaluation/eval.ipynb vergleichbar."""
    SUMMARY_EXCLUDE = set()

    def __init__(self, base_dir: str, config: dict):
        self.config = config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config['cohort']}_{config['model_type']}_{timestamp}"

        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        with open(self.run_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=4)

        self.epoch_logs = []
        self.oof_records = []

    def log_splits(self, df: pd.DataFrame):
        cols = ["case_id", "slide_id", "site", "survival_months", "censorship", "event", "time_bin", "fold"]
        available_cols = [c for c in cols if c in df.columns]
        df[available_cols].to_csv(self.run_dir / "dataset_splits.csv", index=False)

    def log_genes(self, omic_cols: list):
        with open(self.run_dir / "genes.json", "w") as f:
            json.dump(omic_cols, f, indent=0)

    def log_epoch(self, fold: int, epoch: int, train_loss: float, val_loss: float, val_c_index: float):
        self.epoch_logs.append({"fold": fold, "epoch": epoch, "train_loss": train_loss,
                                "val_loss": val_loss, "val_c_index": val_c_index})

    def log_oof_predictions(self, val_df: pd.DataFrame, risk_scores, fold: int):
        oof_df = val_df[["case_id", "slide_id", "survival_months", "censorship", "event", "time_bin"]].copy()
        oof_df["fold"] = fold
        oof_df["risk_score"] = risk_scores
        self.oof_records.append(oof_df)

    def finalize(self, fold_c_indices: list):
        """fold_c_indices: C-Index der letzten Epoche pro Fold."""
        pd.DataFrame(self.epoch_logs).to_csv(self.run_dir / "metrics_per_epoch.csv", index=False)
        if self.oof_records:
            pd.concat(self.oof_records, ignore_index=True).to_csv(self.run_dir / "oof_predictions.csv", index=False)

        summary = {
            "run_name": self.run_dir.name,
            "model_selection": "last_epoch",
            "mean_c_index": round(float(np.mean(fold_c_indices)), 4),
            "std_c_index": round(float(np.std(fold_c_indices)), 4),
            "fold_c_indices": [round(float(x), 4) for x in fold_c_indices],
        }
        summary.update({k: v for k, v in self.config.items() if k not in self.SUMMARY_EXCLUDE})

        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=4)

        master_csv = self.base_dir / "experiment_summary.csv"
        summary_flat = {k: str(v) if isinstance(v, list) else v for k, v in summary.items()}
        df_row = pd.DataFrame([summary_flat])
        if master_csv.exists():
            df_row = pd.concat([pd.read_csv(master_csv), df_row], ignore_index=True)
        df_row.to_csv(master_csv, index=False)

        print(f"\nAlle Ergebnisse archiviert unter: {self.run_dir}")

# ===== cell 12
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Nutze Device: {device}")

# ===== cell 13
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Alle Trainings-Hyperparameter identisch zu SURVPATH.ipynb (und MCAT/MOTCAT/ABMIL/TransMIL)
config = {
    "cohort": "TCGA-GBMLGG",
    "model_type": "snn",
    "pathways": "xena_all",   # Genmenge: ALLE Gene der Xena-Matrix, keine Vorauswahl
    "omic_scaling": "zscore_per_gene_clip5",  # StandardScaler pro Gen (Fit nur auf Train-Fold), Clip auf [-5, 5]
    "model_size_omic": "small",   # SNNOmics: small = [256, 256]

    "num_classes": 4,
    "epochs": 20,
    "lr": 1e-4,
    "lr_scheduler": "none",
    "weight_decay": 1e-5,
    "grad_accum_steps": 32,  # effektive Batchgroesse; hier direkt als batch_size umgesetzt
    "loss_alpha": 0.15,
    "seed": 85,
}

seed_everything(config["seed"])

raw_files = sorted((BASE_DIR / config["cohort"] / "genomics/raw").glob("HiSeqV2_PANCAN_*.tsv"))
all_xena = sorted(set().union(*[pd.read_csv(f, sep="\t", usecols=[0]).iloc[:, 0] for f in raw_files]))
pathways = {"xena_all": all_xena}
dataset = build_dataset(BASE_DIR / config["cohort"], pathways, n_bins=config["num_classes"], n_splits=5)

omic_cols = get_omic_columns(dataset.columns, pathways)
config["n_omic_features"] = len(omic_cols)
assert len(omic_cols) == len(all_xena), "Nicht alle Xena-Gene im Datensatz"
assert np.isfinite(dataset[omic_cols].values).all(), "NaN/Inf in den RNA-Daten"
print(f"Genomic-only Input: {len(omic_cols)} Gene, {len(dataset)} Patienten")

logger = ExperimentLogger(base_dir="./runs", config=config)
logger.log_splits(dataset)
logger.log_genes(omic_cols)

# ===== cell 14
# Vergleichbarkeits-Check: Patienten und Fold-Zuordnung muessen exakt dieselben sein wie im letzten SurvPath-Run
# derselben Kohorte (Splits sind deterministisch, haengen nur von der Kohorte ab).
ref_runs = sorted(Path("../SURVPATH/runs").glob(f"{config['cohort']}_survpath_*/dataset_splits.csv"))
if ref_runs:
    ref = pd.read_csv(ref_runs[-1]).set_index("case_id")["fold"].sort_index()
    own = dataset.set_index("case_id")["fold"].sort_index()
    assert ref.index.equals(own.index), "Andere Patienten als im SurvPath-Run!"
    assert (ref.values == own.values).all(), "Andere Fold-Zuordnung als im SurvPath-Run!"
    print(f"OK: identische Patienten ({len(own)}) und Folds wie {ref_runs[-1].parent.name}")
else:
    print(f"Kein SurvPath-Run fuer {config['cohort']} gefunden -> Check uebersprungen")

# ===== cell 15
def run_fold(fold, dataset, config, omic_cols, dev):
    """Trainiert und validiert einen Fold. Modellauswahl: immer die LETZTE Epoche (wie SURVPATH.ipynb);
    der Val-Fold wird nur geloggt, nie zur Auswahl genutzt."""
    seed_everything(config["seed"] + fold)
    print(f"==================== FOLD {fold} ====================")

    train_df = dataset[dataset['fold'] != fold]
    val_df = dataset[dataset['fold'] == fold]
    train_df, val_df = compute_fold_time_bins(train_df, val_df, n_bins=config["num_classes"])

    train_omics, val_omics = scale_omics(train_df, val_df, omic_cols)

    train_ds = SurvivalOmicsDataset(train_df, train_omics, device=dev)
    val_ds = SurvivalOmicsDataset(val_df, val_omics, device=dev)
    g = torch.Generator().manual_seed(config["seed"] + fold)
    train_loader = DataLoader(train_ds, batch_size=config["grad_accum_steps"], shuffle=True, generator=g, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    model = SNN(omic_input_dim=len(omic_cols), num_classes=config["num_classes"],
                model_size_omic=config["model_size_omic"]).to(dev)
    if fold == 0:
        print(f"Parameter: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = NLLSurvLoss(alpha=config["loss_alpha"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    epoch_logs = []
    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, dev)
        val_loss, val_c_index, val_risk_scores = evaluate(model, val_loader, loss_fn, dev)  # nur Monitoring
        epoch_logs.append((fold, epoch + 1, train_loss, val_loss, val_c_index))
        print(f"[Fold {fold}] Epoch {epoch+1:02d}/{config['epochs']:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val C-Index: {val_c_index:.4f}")

    print(f"--> Fold {fold} | C-Index (letzte Epoche): {val_c_index:.4f}")
    return {"fold": fold, "epoch_logs": epoch_logs, "c_index": val_c_index, "risk": val_risk_scores}

# ===== cell 16
n_folds = 5
results = [run_fold(fold, dataset, config, omic_cols, device) for fold in range(n_folds)]  # sequenziell, dauert nur Sekunden

fold_c_indices = []
for r in results:
    for (f, ep, tl, vl, vc) in r["epoch_logs"]:
        logger.log_epoch(f, ep, tl, vl, vc)
    _, val_df = compute_fold_time_bins(dataset[dataset['fold'] != r["fold"]], dataset[dataset['fold'] == r["fold"]],
                                       n_bins=config["num_classes"])
    logger.log_oof_predictions(val_df, r["risk"], r["fold"])
    fold_c_indices.append(r["c_index"])

print("\n==================================================")
for r in results:
    print(f"Fold {r['fold']} | C-Index (letzte Epoche): {r['c_index']:.4f}")
print(f"5-Fold CV ({config['cohort']}) - C-Index (letzte Epoche): {np.mean(fold_c_indices):.4f} ± {np.std(fold_c_indices):.4f}")
logger.finalize(fold_c_indices)
