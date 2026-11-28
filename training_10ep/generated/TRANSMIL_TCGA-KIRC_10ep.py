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

# ===== cell 1
BASE_DIR = Path("/loctmp/mue37538/dataset/data-transfer-student")

# ===== cell 2
def _get_patch_count(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as f:
        return f["features"].shape[0]

def build_wsi_cohort(cohort_dir: Path) -> pd.DataFrame:
    df_clin = pd.read_csv(
        cohort_dir / "clinical/clinical_survival_per_slide.csv")
    wsi_dir = cohort_dir / "wsi_features/conch_v15"
    available_wsi = {p.stem for p in wsi_dir.glob("*.h5")}

    df_clin = df_clin.dropna(subset=['survival_months', 'censorship']).copy()
    df_clin = df_clin[df_clin['survival_months'] > 0].copy()
    df_clin['slide_id'] = df_clin['slide_id'].str.replace(
        '.svs', '', regex=False)
    df_clin = df_clin[df_clin["slide_id"].isin(available_wsi)].copy()

    
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
def build_dataset(cohort_dir: Path, n_bins: int = 4, n_splits: int = 5) -> pd.DataFrame:
    df_clin = build_wsi_cohort(cohort_dir)
    df_clin = cross_validate(df_clin, n_splits=n_splits, n_bins=n_bins)
    return df_clin

# ===== cell 6
def preload_wsi_features(df: pd.DataFrame, wsi_dir: Path) -> dict:
    """Lädt alle eindeutigen WSI-Feature-Dateien einmalig in den RAM."""
    wsi_dir = Path(wsi_dir)
    unique_slides = df['slide_id'].unique()
    cache = {}
    
    print(f"Lade {len(unique_slides)} WSI-Feature-Dateien in den RAM...")
    for slide_id in unique_slides:
        h5_path = wsi_dir / f"{slide_id}.h5"
        with h5py.File(h5_path, 'r') as f:
            # Direkt als float32-Tensor im RAM ablegen
            cache[slide_id] = torch.from_numpy(f['features'][:]).to(device, dtype=torch.float32) #device laedt die Daten auf die GPU, falls device='cuda', sonst auf die CPU
            
    ram_mb = sum(t.element_size() * t.nelement() for t in cache.values()) / (1024 ** 2)
    print(f"Fertig! Belegter RAM durch Feature-Cache: {ram_mb:.1f} MB")
    return cache


class SurvivalWSIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, wsi_dir: Path, max_patches: int = None, feature_cache: dict = None):
        self.df = df.reset_index(drop=True)
        self.wsi_dir = Path(wsi_dir)
        self.max_patches = max_patches
        self.feature_cache = feature_cache
        # Pro-Sample-Daten einmalig vorbereiten (df.iloc im __getitem__ ist langsam)
        self.slide_ids = self.df['slide_id'].tolist()
        self.targets = torch.tensor(self.df[['time_bin', 'censorship']].values, dtype=torch.long)
        self.surv = torch.tensor(self.df['survival_months'].values, dtype=torch.float32)
        self.cens = torch.tensor(self.df['censorship'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        
        # 1. Aus dem RAM-Cache lesen (falls vorhanden), sonst Fallback auf SSD
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
        
        return features_tensor, self.targets[idx], self.surv[idx], self.cens[idx]


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
# Original-Code aus dem TransMIL-Repo: TransLayer und PPEG werden direkt von dort geladen
# (Datei per Pfad laden, damit models/__init__.py mit pytorch_lightning nicht importiert wird)
import math
import importlib.util

TRANSMIL_FILE = "/loctmp/mue37538/models/TransMIL/models/TransMIL.py"
_spec = importlib.util.spec_from_file_location("transmil_orig", TRANSMIL_FILE)
_orig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_orig)
TransLayer, PPEG = _orig.TransLayer, _orig.PPEG


class TransMIL(nn.Module):
    """Wie TransMIL aus dem Repo, aber mit konfigurierbarem in_dim (CONCH: 768), ohne hartcodiertes .cuda()
    und mit ABMIL-Schnittstelle: forward(h, loss_fn, label) -> (results_dict, None)."""
    def __init__(self, in_dim=768, embed_dim=512, num_classes=4):
        super().__init__()
        self.pos_layer = PPEG(dim=embed_dim)
        self._fc1 = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.layer1 = TransLayer(dim=embed_dim)
        self.layer2 = TransLayer(dim=embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self._fc2 = nn.Linear(embed_dim, num_classes)

    def forward(self, h, loss_fn=None, label=None):
        h = self._fc1(h.float())  # [B, N, D]

        # Auf quadratisches Grid auffuellen
        N = h.shape[1]
        side = math.ceil(math.sqrt(N))
        add_length = side * side - N
        h = torch.cat([h, h[:, :add_length, :]], dim=1)

        cls_tokens = self.cls_token.expand(h.shape[0], -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        h = self.layer1(h)
        h = self.pos_layer(h, side, side)
        h = self.layer2(h)
        h = self.norm(h)[:, 0]

        logits = self._fc2(h)  # [B, num_classes]
        results_dict = {"logits": logits}
        if loss_fn is not None and label is not None:
            results_dict["loss"] = loss_fn(logits, label)
        return results_dict, None


# ===== cell 9
def train_one_epoch(model, loader, loss_fn, optimizer, device, grad_accum_steps=8):
    model.train()
    total_loss = torch.zeros((), device=device)
    optimizer.zero_grad()
    num_batches = len(loader)

    for step, (features, target, _, _) in enumerate(loader):
        features = features.to(device)  # [1, M, 768]
        target = target.to(device)      # [1, 2]

        rem_batches = num_batches - (step // grad_accum_steps) * grad_accum_steps
        actual_accum_steps = min(grad_accum_steps, rem_batches)

        results_dict, _ = model(h=features, loss_fn=loss_fn, label=target)
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

    for features, target, surv_months, censorship in loader:
        features = features.to(device)
        target = target.to(device)

        results_dict, _ = model(h=features, loss_fn=loss_fn, label=target)
        total_loss += results_dict['loss'].detach()

        # Risk Score aus den 4 Survival-Bins berechnen
        hazards = torch.sigmoid(results_dict['logits'])
        survival = torch.cumprod(1 - hazards, dim=1) #Berechnet die kumulative Überlebenswahrscheinlichkeit über die Zeit, indem das Produkt von 1 - hazards gebildet wird.
        risk = -torch.sum(survival, dim=1) # Berechnet den Risikowert als negative Summe der Überlebenswahrscheinlichkeiten (höherer Wert = höheres Risiko)


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
from pathlib import Path
import pandas as pd
import numpy as np

class ExperimentLogger:
    def __init__(self, base_dir: str, config: dict):
        self.config = config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config['cohort']}_{config['model_type']}_{timestamp}"
        
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / run_name
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Config als JSON speichern
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=4)
            
        self.epoch_logs = []
        self.oof_records = []

    def log_splits(self, df: pd.DataFrame):
        # 2. Exakte Splits und Bins sichern
        cols = ["case_id", "slide_id", "site", "survival_months", "censorship", "event", "time_bin", "fold"]
        available_cols = [c for c in cols if c in df.columns]
        df[available_cols].to_csv(self.run_dir / "dataset_splits.csv", index=False)

    def log_epoch(self, fold: int, epoch: int, train_loss: float, val_loss: float, val_c_index: float):
        # 3. Verlauf pro Epoche merken
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
            
        # Master-CSV: einlesen und per concat zusammenfuehren, damit neue/entfallene Spalten
        # nicht gegen einen alten Header verrutschen.
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
    "model_type": "transmil",
    "in_dim": 768, # equal to CONCH v1.5 embedding dimension
    "embed_dim": 512,
    "num_classes": 4,
    "epochs": 10,
    "lr": 1e-4,
    "lr_scheduler": "none",
    "weight_decay": 1e-5,
    "grad_accum_steps": 32, # Wie in Papern Mcat, Motcat
    "max_patches_train": None,
    "loss_alpha": 0.15,
    "seed": 85
}

seed_everything(config["seed"])

dataset = build_dataset(BASE_DIR / config["cohort"], n_bins=config["num_classes"], n_splits=5)

logger = ExperimentLogger(base_dir="./runs", config=config)
logger.log_splits(dataset)
WSI_DIR = BASE_DIR / config["cohort"] / "wsi_features/conch_v15"

# Einmalig in den System-RAM laden
wsi_cache = preload_wsi_features(dataset, WSI_DIR)

fold_c_indices = []

use_pin_memory = (device.type == "cuda")

for fold in range(5):
    print(f"\n==================== FOLD {fold} ====================")
    seed_everything(config["seed"] + fold)
    train_df = dataset[dataset['fold'] != fold]
    val_df = dataset[dataset['fold'] == fold]

    train_df, val_df = compute_fold_time_bins(
        train_df, val_df, n_bins=config["num_classes"]
    )
    # Cache an beide Datasets übergeben
    train_ds = SurvivalWSIDataset(
        train_df, wsi_dir=WSI_DIR, max_patches=config["max_patches_train"], feature_cache=wsi_cache
    )
    val_ds = SurvivalWSIDataset(
        val_df, wsi_dir=WSI_DIR, max_patches=None, feature_cache=wsi_cache
    )
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    
    model = TransMIL(
        in_dim=config["in_dim"],
        embed_dim=config["embed_dim"],
        num_classes=config["num_classes"],
    ).to(device)
    
    loss_fn = NLLSurvLoss(alpha=config["loss_alpha"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    
    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, grad_accum_steps=config["grad_accum_steps"]
        )
        val_loss, val_c_index, val_risk_scores = evaluate(model, val_loader, loss_fn, device)
        
        logger.log_epoch(fold, epoch + 1, train_loss, val_loss, val_c_index)
        print(f"Epoch {epoch+1:02d}/{config['epochs']:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val C-Index: {val_c_index:.4f}")

    # Modellauswahl: immer die LETZTE Epoche; der Val-Fold wird nie zur Auswahl genutzt
    torch.save(model.state_dict(), logger.ckpt_dir / f"last_model_fold_{fold}.pt")
    print(f"--> Fold {fold} | C-Index (letzte Epoche): {val_c_index:.4f}")
    logger.log_oof_predictions(val_df, val_risk_scores, fold)
    fold_c_indices.append(val_c_index)

# GESAMTERGEBNIS & ARCHIVIERUNG
print("\n==================================================")
print(f"5-Fold CV ({config['cohort']}) - C-Index (letzte Epoche): {np.mean(fold_c_indices):.4f} ± {np.std(fold_c_indices):.4f}")
logger.finalize(fold_c_indices)

# ===== cell 13
torch.cuda.empty_cache()
