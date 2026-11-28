import os, sys
sys.path.insert(0, os.getcwd())  # wie im Notebook
display = print

# ===== cell 0
import os
import h5py
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")

# ===== cell 1
BASE_DIR = Path("/loctmp/mue37538/dataset/data-transfer-student")

# ===== cell 2
def _get_patch_count(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as f:
        return f["features"].shape[0]

def build_wsi_cohort(cohort_dir: Path, required_slides: set = None) -> pd.DataFrame:
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

# ===== cell 3
from typing import Tuple

def compute_fold_time_bins(train_df: pd.DataFrame, val_df: pd.DataFrame, n_bins: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Berechnet die Quartil-Grenzen NUR auf den Todesfällen von train_df (verhindert Data Leakage)
    und wendet sie auf train_df und val_df an.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()

    # 1. Fit nur auf den unzensierten Trainingsdaten dieses Folds
    uncensored_train = train_df.loc[train_df["event"] == 1, "survival_months"]
    _, q_bins = pd.qcut(uncensored_train, q=n_bins, retbins=True, labels=False)

    # 2. Äußere Grenzen auf +/- unendlich setzen, damit Val-Samples außerhalb
    #    der Trainings-Range sauber in Bin 0 bzw. Bin (n_bins - 1) fallen (keine NaNs!)
    q_bins[0] = -np.inf
    q_bins[-1] = np.inf

    # 3. Transform auf Train und Val mit denselben Trainings-Grenzen
    train_df["time_bin"] = pd.cut(
        train_df["survival_months"], bins=q_bins, labels=False, include_lowest=True
    ).astype(int)
    val_df["time_bin"] = pd.cut(
        val_df["survival_months"], bins=q_bins, labels=False, include_lowest=True
    ).astype(int)

    return train_df, val_df

# ===== cell 4
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

        for fold, (_, val_idx) in enumerate(sgkf.split(X=df, y=strat_target, groups=df["site"])): # fold ist die aktuelle Fold-Nummer, val_idx sind die Indizes der Validierungsdaten für diesen Fold
            folds[val_idx] = fold
            n_events = df.iloc[val_idx]["event"].sum()
            if n_events < min_events_per_fold:
                valid_split = False
                break
            event_counts.append(n_events)
            fold_sizes.append(len(val_idx))

        if valid_split:
            # Optimiere auf möglichst gleiche Anzahl an Events UND ähnliche Fold-Größe
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
    
    # Kurzer Check-Print Log:
    print("Fold-Verteilung (Größe | Events):")
    for f in range(n_splits):
        sub = df[df["fold"] == f]
        print(f"  Fold {f}: n={len(sub):3d} | events={int(sub['event'].sum()):2d} | sites={sub['site'].nunique()}")
        
    return df

# ===== cell 5
RNA_SUFFIX = "_rnaseq"

def load_pathways(pathway_csv: Path) -> dict:
    """SurvPath-Pathways (Hallmarks + Reactome, 'combine'): survpath_combine_comps.csv ist eine Gen x Pathway 0/1-Matrix.
    Gibt {pathway_name: [gene, ...]} zurueck."""
    m = pd.read_csv(pathway_csv, index_col=0)
    return {pw: m.index[m[pw] == 1].tolist() for pw in m.columns}


def load_rna(cohort_dir: Path, genes: list) -> pd.DataFrame:
    """Rohe Xena-RNA-seq-Matrix (Gene x Samples, log2(norm count+1), pan-cancer normalisiert) -> eine Zeile pro Sample
    (15 Zeichen Barcode, z.B. TCGA-2K-A9WE-01), eingeschraenkt auf die Pathway-Gene. Spalten: <GEN>_rnaseq."""
    files = sorted((cohort_dir / "genomics/raw").glob("HiSeqV2_PANCAN_*.tsv"))  # GBMLGG: _GBM und _LGG
    rna = pd.concat([pd.read_csv(f, sep="\t", index_col=0) for f in files], axis=1)
    rna = rna[~rna.index.duplicated()]
    rna = rna.loc[rna.index.intersection(genes)].T.astype(np.float32)  # [samples, genes]
    rna.columns = [g + RNA_SUFFIX for g in rna.columns]
    rna.index.name = "rna_sample"
    return rna


def get_omic_groups(columns, pathways: dict):
    """Pro Pathway eine sortierte Liste der verfuegbaren <GEN>_rnaseq-Spalten (wie _series_intersection im SurvPath-Original).
    Pathways ohne verfuegbares Gen werden verworfen. Gibt (groups, pathway_names) zurueck."""
    columns = set(columns)
    groups, names = [], []
    for pw, genes in pathways.items():
        cols = sorted(g + RNA_SUFFIX for g in genes if g + RNA_SUFFIX in columns)
        if cols:
            groups.append(cols)
            names.append(pw)
        else:
            print(f"Pathway '{pw}' verworfen (kein Gen in der RNA-Matrix)")
    return groups, names


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


# ===== cell 6
def preload_wsi_features(df: pd.DataFrame, wsi_dir: Path) -> dict:
    """Lädt alle eindeutigen WSI-Feature-Dateien einmalig auf das Device (GPU-VRAM bei 'cuda', sonst RAM)."""
    wsi_dir = Path(wsi_dir)
    unique_slides = df['slide_id'].unique()
    cache = {}
    
    print(f"Lade {len(unique_slides)} WSI-Feature-Dateien auf {device}...")
    for slide_id in unique_slides:
        h5_path = wsi_dir / f"{slide_id}.h5"
        with h5py.File(h5_path, 'r') as f:
            # Direkt als float32-Tensor auf dem Device ablegen
            cache[slide_id] = torch.from_numpy(f['features'][:]).to(device, dtype=torch.float32) #device laedt die Daten auf die GPU, falls device='cuda', sonst auf die CPU
            
    mem_mb = sum(t.element_size() * t.nelement() for t in cache.values()) / (1024 ** 2)
    mem_name = "VRAM" if device.type == "cuda" else "RAM"
    print(f"Fertig! Belegter {mem_name} durch Feature-Cache: {mem_mb:.1f} MB")
    return cache


def scale_omics(train_df: pd.DataFrame, val_df: pd.DataFrame, omic_groups: list, clip: float = 5.0):
    """Z-Standardisierung PRO GEN (StandardScaler), NUR auf dem Trainings-Fold gefittet (kein Leakage), danach auf
    [-clip, clip] begrenzt. Die Xena-PANCAN-Daten sind pro Gen bereits um 0 zentriert (0 = durchschnittlich exprimiert,
    keine NaNs), daher keine Sonderbehandlung von 0-Werten.
    Gibt pro Fold EIN float32-Array [n_samples, sum(omic_sizes)] zurueck: die Pathway-Gruppen hintereinander
    (Gene, die in mehreren Pathways vorkommen, entsprechend mehrfach). Aufgeteilt wird im Modell-Wrapper per torch.split."""
    all_cols = sorted({c for g in omic_groups for c in g})  # eindeutig: Gene sind in mehreren Pathways
    flat_cols = [c for g in omic_groups for c in g]        # Reihenfolge = Pathway-Gruppen hintereinander
    scaler = StandardScaler().fit(train_df[all_cols].values)  # Gene mit Varianz 0 -> scale=1 (kein Div. durch 0)
    out = []
    for df in (train_df, val_df):
        scaled = np.clip(scaler.transform(df[all_cols].values), -clip, clip)
        scaled = pd.DataFrame(scaled, columns=all_cols, index=df.index)
        out.append(scaled[flat_cols].values.astype(np.float32))
    return out[0], out[1]


class SurvivalWSIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, wsi_dir: Path, omic_array: np.ndarray, max_patches: int = None,
                 feature_cache: dict = None, device: torch.device = torch.device("cpu")):
        self.df = df.reset_index(drop=True)
        self.wsi_dir = Path(wsi_dir)
        self.max_patches = max_patches
        self.feature_cache = feature_cache
        # Pro-Sample-Daten einmalig vorbereiten (df.iloc im __getitem__ ist langsam)
        self.slide_ids = self.df['slide_id'].tolist()
        # Omics einmalig als EIN Tensor [n_samples, sum(omic_sizes)] auf das Device (wenige MB) -> kein Transfer pro Schritt
        self.omics = torch.from_numpy(omic_array).to(device)
        self.targets = torch.tensor(self.df[['time_bin', 'censorship']].values, dtype=torch.long)
        self.surv = torch.tensor(self.df['survival_months'].values, dtype=torch.float32)
        self.cens = torch.tensor(self.df['censorship'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        
        # 1. Aus dem Feature-Cache lesen (falls vorhanden), sonst Fallback auf SSD
        if self.feature_cache is not None and slide_id in self.feature_cache:
            features_tensor = self.feature_cache[slide_id]
        else:
            h5_path = self.wsi_dir / f"{slide_id}.h5"
            with h5py.File(h5_path, 'r') as f:
                features_tensor = torch.from_numpy(f['features'][:]).to(torch.float32)

        # 2. Optionales Subsampling direkt mit PyTorch auf dem Tensor
        if self.max_patches is not None and features_tensor.size(0) > self.max_patches:
            indices = torch.randperm(features_tensor.size(0))[:self.max_patches]
            features_tensor = features_tensor[indices]
        
        return features_tensor, self.omics[idx], self.targets[idx], self.surv[idx], self.cens[idx]

# ===== cell 7
from sksurv.metrics import concordance_index_censored

class NLLSurvLoss(nn.Module):
    """
    Discrete-Time Survival Loss (Negative Log-Likelihood).
    """
    def __init__(self, alpha: float = 0.15):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # target: [B, 2] -> Spalte 0 = time_bin, Spalte 1 = censorship (1 = zensiert, 0 = Event)
        y = target[:, 0].view(-1, 1)
        c = target[:, 1].view(-1, 1).float()

        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1) # kumulatives Produkt: Tensor, bei dem jeder Eintrag die Wahrscheinlichkeit angibt, bis zu diesem spezifischen Zeitpunkt (und allen vorherigen) überlebt zu haben. 
        S_padded = torch.cat([torch.ones_like(c), S], dim=1)

        s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=1e-7)
        h_this = torch.gather(hazards, dim=1, index=y).clamp(min=1e-7)
        s_this = torch.gather(S_padded, dim=1, index=y + 1).clamp(min=1e-7)

        uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
        censored_loss = -c * torch.log(s_this)

        loss = (1.0 - self.alpha) * censored_loss + uncensored_loss
        return loss.mean()


# ===== cell 8
# Original-SurvPath-Modell (Jaume et al., CVPR 2024) aus dem lokalen Repo
SURVPATH_REPO = "/loctmp/mue37538/models/SURVPATH/SurvPath"
if SURVPATH_REPO not in sys.path:
    sys.path.insert(0, SURVPATH_REPO)
from models.model_SurvPath import SurvPath


class SURVPATH(nn.Module):
    """Wrapper um SurvPath mit der Schnittstelle forward(h, omics, loss_fn, label) -> (results_dict, None).
    h: [1, N, in_dim] WSI-Patches, omics: EIN Tensor [1, sum(omic_sizes)] (Pathway-Gruppen hintereinander),
    der hier per torch.split in n_pathways Vektoren [d_i] zerlegt wird. Batchgroesse 1.
    Die Pathway-Token (SNN pro Pathway) und die Patch-Token werden vom Original-Modell in einem multimodalen
    Transformer (MMAttention) fusioniert."""
    def __init__(self, omic_sizes, in_dim=768, num_classes=4, wsi_projection_dim=256, dropout=0.1):
        super().__init__()
        self.omic_sizes = list(omic_sizes)
        self.model = SurvPath(omic_sizes=omic_sizes, wsi_embedding_dim=in_dim, dropout=dropout,
                              num_classes=num_classes, wsi_projection_dim=wsi_projection_dim)

    def forward(self, h, omics, loss_fn=None, label=None):
        chunks = torch.split(omics.squeeze(0).float(), self.omic_sizes)  # n_pathways Views, je [d_i]
        kwargs = {f"x_omic{i+1}": o for i, o in enumerate(chunks)}
        logits = self.model(x_path=h.float(), return_attn=False, **kwargs)  # [1, num_classes]
        results_dict = {"logits": logits}
        S = torch.cumprod(1 - torch.sigmoid(logits), dim=1)
        results_dict["risk"] = -S.sum(dim=1).detach()  # hoeher = hoeheres Risiko
        if loss_fn is not None and label is not None:
            results_dict["loss"] = loss_fn(logits, label)
        return results_dict, None


# ===== cell 9
def train_one_epoch(model, loader, loss_fn, optimizer, device, grad_accum_steps=8):
    model.train()
    total_loss = torch.zeros((), device=device)
    optimizer.zero_grad()
    num_batches = len(loader)

    for step, (features, omics, target, _, _) in enumerate(loader):
        features = features.to(device)  # [1, M, 768]
        omics = omics.to(device)        # [1, sum(omic_sizes)] (liegt bereits auf dem Device -> No-op)
        target = target.to(device)      # [1, 2]

        rem_batches = num_batches - (step // grad_accum_steps) * grad_accum_steps
        actual_accum_steps = min(grad_accum_steps, rem_batches)

        results_dict, _ = model(h=features, omics=omics, loss_fn=loss_fn, label=target)
        loss = results_dict['loss'] / actual_accum_steps
        loss.backward()

        total_loss += results_dict['loss'].detach() # Nur für Logging; kein .item() hier, das würde pro Schritt die GPU synchronisieren

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == num_batches:
            optimizer.step()
            optimizer.zero_grad()

    return total_loss.item() / len(loader)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = torch.zeros((), device=device)
    all_risk_scores = []
    all_censorships = []
    all_event_times = []

    for features, omics, target, surv_months, censorship in loader:
        features = features.to(device)
        omics = omics.to(device)
        target = target.to(device)

        results_dict, _ = model(h=features, omics=omics, loss_fn=loss_fn, label=target)
        total_loss += results_dict['loss'].detach()

        risk = results_dict['risk']  # -sum(S), gemittelt ueber die Micro-Batches (hoeher = hoeheres Risiko)

        all_risk_scores.append(risk.detach())  # bleibt auf der GPU, ein Transfer am Ende
        all_censorships.extend(censorship.numpy())
        all_event_times.extend(surv_months.numpy())

    all_risk_scores = torch.cat(all_risk_scores).cpu().numpy()

    # sksurv erwartet event_indicator = True für Todesfall (1 - censorship)
    event_indicators = (1 - np.array(all_censorships)).astype(bool)
    c_index = concordance_index_censored(
        event_indicators,
        np.array(all_event_times),
        np.array(all_risk_scores)
    )[0]

    return total_loss.item() / len(loader), c_index, all_risk_scores

# ===== cell 10
import json
from datetime import datetime

class ExperimentLogger:
    # Nur in config.json, nicht in summary.json / experiment_summary.csv (zu lang)
    SUMMARY_EXCLUDE = {"omic_sizes"}

    def __init__(self, base_dir: str, config: dict):
        self.config = config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config['cohort']}_{config['model_type']}_{timestamp}"
        
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Config als JSON speichern (inkl. vollstaendiger omic_sizes-Liste)
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=4)
            
        self.epoch_logs = []
        self.oof_records = []

    def log_splits(self, df: pd.DataFrame):
        # 2. Exakte Splits und Bins sichern
        cols = ["case_id", "slide_id", "site", "survival_months", "censorship", "event", "time_bin", "fold"]
        available_cols = [c for c in cols if c in df.columns]
        df[available_cols].to_csv(self.run_dir / "dataset_splits.csv", index=False)

    def log_pathways(self, pathway_names: list, omic_groups: list):
        # Pathway-Reihenfolge = Reihenfolge der Pathway-Token im Modell (x_omic1, x_omic2, ...) -> fuer Attention-Interpretation
        records = [{"token_idx": i, "pathway": pw, "n_genes": len(g), "genes": g}
                   for i, (pw, g) in enumerate(zip(pathway_names, omic_groups))]
        with open(self.run_dir / "pathways.json", "w") as f:
            json.dump(records, f, indent=2)

    def log_epoch(self, fold: int, epoch: int, train_loss: float, val_loss: float, val_c_index: float):
        # 3. Verlauf pro Epoche merken (nur Monitoring, keine Modellauswahl)
        self.epoch_logs.append({
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_c_index": val_c_index
        })

    def log_oof_predictions(self, val_df: pd.DataFrame, risk_scores: list, fold: int):
        # 4. Validierungs-Vorhersagen der LETZTEN Epoche dieses Folds für spätere Kaplan-Meier-Analysen speichern
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
            
        # Master-CSV: bestehende Datei einlesen und per concat zusammenfuehren, damit neue/entfallene Spalten
        # (z.B. omic_scaling, n_pathways) nicht gegen einen alten Header verrutschen.
        master_csv = self.base_dir / "experiment_summary.csv"
        summary_flat = {k: str(v) if isinstance(v, list) else v for k, v in summary.items()}
        df_row = pd.DataFrame([summary_flat])
        if master_csv.exists():
            df_row = pd.concat([pd.read_csv(master_csv), df_row], ignore_index=True)
        df_row.to_csv(master_csv, index=False)
        
        print(f"\nAlle Ergebnisse archiviert unter: {self.run_dir}")

# ===== cell 11
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Nutze Device: {device}")

# ===== cell 12
import random
# 1. Zentrale Konfiguration definieren

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

config = {
    "cohort": "TCGA-KIRC",
    "model_type": "survpath",
    "pathways": "combine",   # Hallmarks + Reactome (survpath_combine_comps.csv)
    "omic_scaling": "zscore_per_gene_clip5",  # StandardScaler pro Gen (Fit nur auf Train-Fold), Clip auf [-5, 5]
    "in_dim": 768, # equal to CONCH v1.5 embedding dimension
    "wsi_projection_dim": 256,
    "dropout": 0.1,
    "num_classes": 4,
    "epochs": 10,
    "lr": 1e-4,
    "lr_scheduler": "none",
    "weight_decay": 1e-5,
    "grad_accum_steps": 32,
    "max_patches_train": None,   # kein Patch-Subsampling: alle Patches im Training
    "loss_alpha": 0.15,
    "seed": 85,
    "n_jobs": 5   # Folds parallel in eigenen Prozessen (1 = nacheinander, mit Feature-Cache auf der GPU)
}

seed_everything(config["seed"])

pathways = load_pathways(BASE_DIR / "shared/gene_sets/survpath_combine_comps.csv")
dataset = build_dataset(BASE_DIR / config["cohort"], pathways, n_bins=config["num_classes"], n_splits=5)

# Pathway-Gruppen (SurvPath): Spaltennamen der RNA-Features je Pathway
omic_groups, pathway_names = get_omic_groups(dataset.columns, pathways)
omic_sizes = [len(g) for g in omic_groups]
print(f"{len(omic_groups)} Pathways, Gene pro Pathway: min={min(omic_sizes)}, max={max(omic_sizes)}, Summe={sum(omic_sizes)}")
config["n_pathways"] = len(omic_sizes)
config["n_omic_features"] = sum(omic_sizes)
config["omic_sizes"] = omic_sizes   # nur in config.json, nicht in experiment_summary.csv

logger = ExperimentLogger(base_dir="./runs", config=config)
logger.log_splits(dataset)
logger.log_pathways(pathway_names, omic_groups)   # -> pathways.json (Token-Index -> Pathway + Gene)
WSI_DIR = BASE_DIR / config["cohort"] / "wsi_features/conch_v15"


# ===== cell 13
from joblib import Parallel, delayed


def _fold_log(msg, log_path, echo):
    with open(log_path, "a") as f:
        f.write(msg + "\n")
    if echo:
        print(msg)


def run_fold(fold, dataset, config, omic_groups, wsi_dir, log_path, feature_cache=None, device_str="cuda"):
    """Trainiert und validiert einen Fold. Laeuft im Hauptprozess (n_jobs=1) oder in einem eigenen Prozess (n_jobs>1).
    Ausgaben gehen in log_path. Gibt nur Ergebnisse zurueck.
    Modellauswahl: immer die LETZTE Epoche (feste Epochenzahl aus der Config). Der Val-Fold wird nur geloggt,
    nie zur Auswahl genutzt -> unverzerrte Schaetzung des C-Index."""
    echo = feature_cache is not None  # sequenziell -> zusaetzlich im Notebook ausgeben
    dev = torch.device(device_str)
    torch.set_num_threads(max(1, os.cpu_count() // max(1, config["n_jobs"])))  # CPU nicht ueberbuchen
    seed_everything(config["seed"] + fold)
    _fold_log(f"==================== FOLD {fold} ====================", log_path, echo)

    train_df = dataset[dataset['fold'] != fold]
    val_df = dataset[dataset['fold'] == fold]
    train_df, val_df = compute_fold_time_bins(train_df, val_df, n_bins=config["num_classes"])

    # Omics: Scaler nur auf dem Trainings-Fold fitten
    train_omics, val_omics = scale_omics(train_df, val_df, omic_groups)

    # Omics liegen danach als ein Tensor pro Dataset auf dem Device (dev)
    train_ds = SurvivalWSIDataset(train_df, wsi_dir=wsi_dir, omic_array=train_omics,
                                  max_patches=config["max_patches_train"], feature_cache=feature_cache, device=dev)
    val_ds = SurvivalWSIDataset(val_df, wsi_dir=wsi_dir, omic_array=val_omics,
                                max_patches=None, feature_cache=feature_cache, device=dev)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    model = SURVPATH(
        omic_sizes=[len(g) for g in omic_groups],
        in_dim=config["in_dim"],
        num_classes=config["num_classes"],
        wsi_projection_dim=config["wsi_projection_dim"],
        dropout=config["dropout"],
    ).to(dev)

    loss_fn = NLLSurvLoss(alpha=config["loss_alpha"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    epoch_logs = []

    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, dev, grad_accum_steps=config["grad_accum_steps"])
        val_loss, val_c_index, val_risk_scores = evaluate(model, val_loader, loss_fn, dev)  # nur Monitoring

        epoch_logs.append((fold, epoch + 1, train_loss, val_loss, val_c_index))
        _fold_log(f"[Fold {fold}] Epoch {epoch+1:02d}/{config['epochs']:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val C-Index: {val_c_index:.4f}", log_path, echo)

    # Keine Checkpoints: es werden nur Metriken und OOF-Risk-Scores gespeichert (Baseline-Vergleich)

    _fold_log(f"--> Fold {fold} | C-Index (letzte Epoche): {val_c_index:.4f}", log_path, echo)
    return {"fold": fold, "epoch_logs": epoch_logs, "c_index": val_c_index, "risk": val_risk_scores}

# ===== cell 14
n_jobs = config["n_jobs"]
n_folds = 5

# Sequenziell: alle Features einmalig auf die GPU. Parallel: jeder Prozess liest die h5-Dateien selbst
# (geteilter OS-Seitencache), damit der VRAM nicht n_jobs-fach belegt wird.
wsi_cache = preload_wsi_features(dataset, WSI_DIR) if n_jobs == 1 else None

# Worker-Prozesse muessen das SurvPath-Repo importieren koennen
os.environ["PYTHONPATH"] = SURVPATH_REPO + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ["PYTHONWARNINGS"] = "ignore:enable_nested_tensor is True"  # Warnung auch in den Worker-Prozessen unterdruecken


import threading, time

def _tail_logs(log_files, stop_event, interval=2.0):
    """Druckt neue Zeilen der fold_*.log live ins Notebook (Worker haben kein eigenes stdout)."""
    pos = {f: 0 for f in log_files}
    while True:
        done = stop_event.is_set()  # nach dem Stop noch einmal vollstaendig lesen
        for f in log_files:
            if f.exists():
                with open(f) as fh:
                    fh.seek(pos[f])
                    for line in fh:
                        if line.startswith("[Fold") or line.startswith("-->"):
                            print(line.rstrip())
                    pos[f] = fh.tell()
        if done:
            return
        time.sleep(interval)

if n_jobs > 1:
    print(f"Starte {n_folds} Folds mit {n_jobs} parallelen Prozessen. Fortschritt: tail -f {logger.run_dir}/fold_*.log")

log_files = [logger.run_dir / f"fold_{fold}.log" for fold in range(n_folds)]
stop_tail = threading.Event()
tail_thread = None
if n_jobs > 1:
    tail_thread = threading.Thread(target=_tail_logs, args=(log_files, stop_tail), daemon=True)
    tail_thread.start()

try:
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_fold)(fold, dataset, config, omic_groups, WSI_DIR,
                          log_files[fold], wsi_cache, str(device))
        for fold in range(n_folds)
    )
finally:
    stop_tail.set()
    if tail_thread is not None:
        tail_thread.join()
results = sorted(results, key=lambda r: r["fold"])

fold_c_indices = []
for r in results:
    for (f, ep, tl, vl, vc) in r["epoch_logs"]:
        logger.log_epoch(f, ep, tl, vl, vc)
    # val_df mit denselben Fold-Bins wie im Training fuer die OOF-Vorhersagen
    _, val_df = compute_fold_time_bins(dataset[dataset['fold'] != r["fold"]], dataset[dataset['fold'] == r["fold"]],
                                       n_bins=config["num_classes"])
    logger.log_oof_predictions(val_df, r["risk"], r["fold"])
    fold_c_indices.append(r["c_index"])
    print(f"Fold {r['fold']} | C-Index (letzte Epoche): {r['c_index']:.4f}")

# GESAMTERGEBNIS & ARCHIVIERUNG
print("\n==================================================")
print(f"5-Fold CV ({config['cohort']}) - C-Index (letzte Epoche): {np.mean(fold_c_indices):.4f} ± {np.std(fold_c_indices):.4f}")
logger.finalize(fold_c_indices)

# ===== cell 15
torch.cuda.empty_cache()
