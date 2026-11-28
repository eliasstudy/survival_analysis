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
import json
from sklearn.preprocessing import StandardScaler

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
    """Pathway-Gene (Hallmarks + Reactome, 'combine'): survpath_combine_comps.csv ist eine Gen x Pathway 0/1-Matrix.
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


def get_omic_columns(columns, pathways: dict) -> list:
    """Jedes Pathway-Gen genau EINMAL (sortiert) als ein flacher Feature-Vektor (gleiche Genmenge wie SNN/MLP/SurvPath).
    HEALNet bekommt die Genomics als EINE tabellarische Modalitaet, ohne Pathway-Gruppierung."""
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

# ===== cell 6
def check_split_consistency(dataset: pd.DataFrame, cohort: str, models_dir: Path = Path("..")):
    """Vergleicht Patienten, Slides und Fold-Zuordnung mit den gespeicherten Splits der anderen Modelle.
    Bricht bei Abweichungen ab (z.B. durch eine andere scikit-learn-Version), da die C-Indizes sonst nicht vergleichbar sind."""
    key = ["case_id", "slide_id", "fold"]
    own = dataset[key].sort_values("case_id").reset_index(drop=True)
    checked = []
    for model_dir in sorted(p for p in models_dir.resolve().iterdir() if p.name.lower() != "healnet"):
        runs = sorted(model_dir.glob(f"runs/{cohort}_*/dataset_splits.csv"), key=lambda p: p.parent.name[-15:])
        if not runs:
            continue
        other = pd.read_csv(runs[-1])[key].sort_values("case_id").reset_index(drop=True)
        if not own.equals(other):
            raise AssertionError(f"Split weicht von {runs[-1]} ab!")
        checked.append(model_dir.name)
    print(f"Split identisch mit: {', '.join(checked) if checked else '(keine Vergleichs-Runs gefunden)'}")

# ===== cell 7
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


def scale_omics(train_df: pd.DataFrame, val_df: pd.DataFrame, omic_cols: list, clip: float = 5.0):
    """Z-Standardisierung PRO GEN (StandardScaler), NUR auf dem Trainings-Fold gefittet (kein Leakage), danach auf
    [-clip, clip] begrenzt. Gibt pro Fold EIN float32-Array [n_samples, n_genes] zurueck."""
    scaler = StandardScaler().fit(train_df[omic_cols].values)  # Gene mit Varianz 0 -> scale=1 (kein Div. durch 0)
    return tuple(np.clip(scaler.transform(df[omic_cols].values), -clip, clip).astype(np.float32)
                 for df in (train_df, val_df))


class SurvivalWSIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, wsi_dir: Path, omic_array: np.ndarray, max_patches: int = None,
                 feature_cache: dict = None, device: torch.device = torch.device("cpu")):
        self.df = df.reset_index(drop=True)
        self.wsi_dir = Path(wsi_dir)
        self.max_patches = max_patches
        self.feature_cache = feature_cache
        # Pro-Sample-Daten einmalig vorbereiten (df.iloc im __getitem__ ist langsam)
        self.slide_ids = self.df['slide_id'].tolist()
        # Omics einmalig als EIN Tensor [n_samples, n_genes] auf das Device (wenige MB) -> kein Transfer pro Schritt
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

# ===== cell 8
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

# ===== cell 9
from healnet import HealNet


class HEALNetSurv(nn.Module):
    """Wrapper um HealNet mit der Schnittstelle forward(h, omics, loss_fn, label) -> (results_dict, None).
    h: [1, N, in_dim] WSI-Patches, omics: [1, n_genes]. Batchgroesse 1."""
    N_MODALITIES = 2

    def __init__(self, n_genes: int, config: dict):
        super().__init__()
        self.model = HealNet(
            n_modalities=self.N_MODALITIES,
            channel_dims=[n_genes, config["in_dim"]],   # Reihenfolge: [Genomics, WSI] (wie sources im Original)
            num_spatial_axes=[1, 1],
            out_dims=config["num_classes"],
            depth=config["depth"],
            num_freq_bands=config["num_freq_bands"],
            max_freq=config["max_freq"],
            l_c=config["num_latents"],
            l_d=config["latent_dim"],
            x_heads=config["cross_heads"],
            l_heads=config["latent_heads"],
            cross_dim_head=config["cross_dim_head"],
            latent_dim_head=config["latent_dim_head"],
            attn_dropout=config["attn_dropout"],
            ff_dropout=config["ff_dropout"],
            weight_tie_layers=config["weight_tie_layers"],
            fourier_encode_data=config["fourier_encode_data"],
            self_per_cross_attn=config["self_per_cross_attn"],
            final_classifier_head=True,
            snn=config["snn"],
        )
        # Schutz gegen das stille try/except in HealNet.forward: Hooks feuern nur bei erfolgreichem Aufruf
        self._n_cross = 0
        self._expected_cross = len(self.model.layers) * self.N_MODALITIES
        for layer in self.model.layers:
            for m in range(self.N_MODALITIES):
                layer[2 * m].register_forward_hook(self._count_cross)

    def _count_cross(self, module, inputs, output):
        self._n_cross += 1

    def forward(self, h, omics, loss_fn=None, label=None):
        self._n_cross = 0
        # Die Liste wird von HealNet.forward veraendert -> bei jedem Aufruf neu erzeugen
        logits = self.model([omics.float().unsqueeze(1), h.float()])  # [1, num_classes]
        if self._n_cross != self._expected_cross:
            raise RuntimeError(f"Nur {self._n_cross}/{self._expected_cross} Cross-Attention-Aufrufe erfolgreich "
                               "(Fehler wurde in HealNet.forward verschluckt, z.B. CUDA Out-of-Memory).")
        results_dict = {"logits": logits}
        S = torch.cumprod(1 - torch.sigmoid(logits), dim=1)
        results_dict["risk"] = -S.sum(dim=1).detach()  # hoeher = hoeheres Risiko
        if loss_fn is not None and label is not None:
            results_dict["loss"] = loss_fn(logits, label)
        return results_dict, None

# ===== cell 10
def train_one_epoch(model, loader, loss_fn, optimizer, device, grad_accum_steps=8, l1=0.0):
    """l1: L1-Strafterm aus HEALNet, einmal pro Optimizer-Schritt auf den Gradienten addiert.
    Der geloggte Loss enthaelt nur den Survival-Loss (vergleichbar mit Val Loss und anderen Modellen)."""
    model.train()
    total_loss = torch.zeros((), device=device)
    optimizer.zero_grad()
    num_batches = len(loader)

    for step, (features, omics, target, _, _) in enumerate(loader):
        features = features.to(device)  # [1, M, 768]
        omics = omics.to(device)        # [1, n_genes] (liegt bereits auf dem Device -> No-op)
        target = target.to(device)      # [1, 2]

        rem_batches = num_batches - (step // grad_accum_steps) * grad_accum_steps
        actual_accum_steps = min(grad_accum_steps, rem_batches)

        results_dict, _ = model(h=features, omics=omics, loss_fn=loss_fn, label=target)
        loss = results_dict['loss'] / actual_accum_steps
        loss.backward()

        total_loss += results_dict['loss'].detach() # Nur für Logging; kein .item() hier, das würde pro Schritt die GPU synchronisieren

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == num_batches:
            if l1 > 0:
                (l1 * sum(p.abs().sum() for p in model.parameters())).backward()
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

        risk = results_dict['risk']  # -sum(S) (hoeher = hoeheres Risiko)

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

# ===== cell 11
from datetime import datetime

class ExperimentLogger:
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

    def log_genes(self, omic_cols: list):
        # Reihenfolge = Reihenfolge der Gen-Kanaele der Omics-Modalitaet -> fuer Attention-Interpretation
        with open(self.run_dir / "genes.json", "w") as f:
            json.dump([c.removesuffix("_rnaseq") for c in omic_cols], f)

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

        summary.update(self.config)
        
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=4)
            
        # Master-CSV: bestehende Datei einlesen und per concat zusammenfuehren, damit neue/entfallene Spalten
        # (z.B. neue Hyperparameter) nicht gegen einen alten Header verrutschen.
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
import random
# 1. Zentrale Konfiguration definieren

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

config = {
    "cohort": "TCGA-GBMLGG",
    "model_type": "healnet",
    "pathways": "combine",   # Gene aus Hallmarks + Reactome (survpath_combine_comps.csv)
    "omic_scaling": "zscore_per_gene_clip5",  # StandardScaler pro Gen (Fit nur auf Train-Fold), Clip auf [-5, 5]
    "in_dim": 768, # equal to CONCH v1.5 embedding dimension

    # HEALNet-Architektur: config/best_hyperparams.yml, Eintrag "kirp"
    "depth": 5,
    "num_latents": 17,
    "latent_dim": 62,
    "cross_heads": 1,
    "cross_dim_head": 27,
    "latent_heads": 8,        # ohne Wirkung bei self_per_cross_attn = 0
    "latent_dim_head": 113,   # ohne Wirkung bei self_per_cross_attn = 0
    "self_per_cross_attn": 0,
    "num_freq_bands": 2,
    "max_freq": 2.0,
    "fourier_encode_data": True,
    "weight_tie_layers": False,
    "snn": True,
    "attn_dropout": 0.31789955176609086,
    "ff_dropout": 0.04735283995174411,
    "l1": 0.00004627399922284714,

    # Training: identisch zu den anderen Modellen
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
check_split_consistency(dataset, config["cohort"])

omic_cols = get_omic_columns(dataset.columns, pathways)
config["n_omic_features"] = len(omic_cols)
print(f"{len(dataset)} Patienten, {len(omic_cols)} Gene")

logger = ExperimentLogger(base_dir="./runs", config=config)
logger.log_splits(dataset)
logger.log_genes(omic_cols)   # -> genes.json
WSI_DIR = BASE_DIR / config["cohort"] / "wsi_features/conch_v15"

# ===== cell 14
def sanity_check():
    sample_df = dataset.iloc[:1]
    omics, _ = scale_omics(dataset, sample_df, omic_cols)
    omics = torch.from_numpy(omics[:1]).to(device)  # Werte hier nur zum Testen der Shapes
    with h5py.File(WSI_DIR / f"{sample_df['slide_id'].iloc[0]}.h5", "r") as f:
        h = torch.from_numpy(f["features"][:]).float().unsqueeze(0).to(device)
    model = HEALNetSurv(len(omic_cols), config).to(device)
    print(f"Genomics {tuple(omics.unsqueeze(1).shape)} | WSI {tuple(h.shape)} | "
          f"Parameter: {sum(p.numel() for p in model.parameters()):,}")

    model.eval()
    out = model(h=h, omics=omics)[0]["logits"]
    assert out.shape == (1, config["num_classes"])
    assert not torch.allclose(out, model(h=h, omics=torch.zeros_like(omics))[0]["logits"]), "Genomics ohne Einfluss"
    assert not torch.allclose(out, model(h=torch.zeros_like(h), omics=omics)[0]["logits"]), "WSI ohne Einfluss"

    model(h=h, omics=omics)[0]["logits"].sum().backward()
    for m, name in enumerate(["Genomics", "WSI"]):
        grad = sum(layer[2 * m].fn.to_kv.weight.grad.abs().sum().item() for layer in model.model.layers)
        assert grad > 0, f"Kein Gradient fuer {name}"
    print("OK: beide Modalitaeten fliessen in die Vorhersage ein.")

sanity_check()
torch.cuda.empty_cache()

# ===== cell 15
from joblib import Parallel, delayed


def _fold_log(msg, log_path, echo):
    with open(log_path, "a") as f:
        f.write(msg + "\n")
    if echo:
        print(msg)


def run_fold(fold, dataset, config, omic_cols, wsi_dir, log_path, feature_cache=None, device_str="cuda"):
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
    train_omics, val_omics = scale_omics(train_df, val_df, omic_cols)

    # Omics liegen danach als ein Tensor pro Dataset auf dem Device (dev)
    train_ds = SurvivalWSIDataset(train_df, wsi_dir=wsi_dir, omic_array=train_omics,
                                  max_patches=config["max_patches_train"], feature_cache=feature_cache, device=dev)
    val_ds = SurvivalWSIDataset(val_df, wsi_dir=wsi_dir, omic_array=val_omics,
                                max_patches=None, feature_cache=feature_cache, device=dev)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    model = HEALNetSurv(len(omic_cols), config).to(dev)

    loss_fn = NLLSurvLoss(alpha=config["loss_alpha"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    epoch_logs = []

    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, dev, grad_accum_steps=config["grad_accum_steps"], l1=config["l1"])
        val_loss, val_c_index, val_risk_scores = evaluate(model, val_loader, loss_fn, dev)  # nur Monitoring

        epoch_logs.append((fold, epoch + 1, train_loss, val_loss, val_c_index))
        _fold_log(f"[Fold {fold}] Epoch {epoch+1:02d}/{config['epochs']:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val C-Index: {val_c_index:.4f}", log_path, echo)

    # Keine Checkpoints: es werden nur Metriken und OOF-Risk-Scores gespeichert (Baseline-Vergleich)

    _fold_log(f"--> Fold {fold} | C-Index (letzte Epoche): {val_c_index:.4f}", log_path, echo)
    return {"fold": fold, "epoch_logs": epoch_logs, "c_index": val_c_index, "risk": val_risk_scores}

# ===== cell 16
n_jobs = config["n_jobs"]
n_folds = 5

# Sequenziell: alle Features einmalig auf die GPU. Parallel: jeder Prozess liest die h5-Dateien selbst
# (geteilter OS-Seitencache), damit der VRAM nicht n_jobs-fach belegt wird.
wsi_cache = preload_wsi_features(dataset, WSI_DIR) if n_jobs == 1 else None

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
        delayed(run_fold)(fold, dataset, config, omic_cols, WSI_DIR,
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

# ===== cell 17
torch.cuda.empty_cache()
