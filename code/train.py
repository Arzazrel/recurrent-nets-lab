# -*- coding: utf-8 -*-
"""
train.py
========
Motore di addestramento condiviso dalle tre reti e dai due task.

Metriche:
- REGRESSIONE NUMERICA: l'obiettivo e' deterministico, quindi ha senso una "accuracy": consideriamo corretta una predizione entro una tolleranza dal
    valore vero (de-normalizzato). La soglia (es. 98%) e' raggiungibile e informativa -> si misura "quante epoche per arrivare alla soglia".
- CLASSIFICAZIONE TESTO (next-token): predire la parola/carattere esatto su testo naturale plafona molto in basso; l'accuracy e' fuorviante. 
    La metrica corretta e' la CROSS-ENTROPY e la sua esponenziale, la PERPLEXITY: perplexity = exp(cross_entropy)
    Interpretazione: il "numero medio di scelte equiprobabili" fra cui il modello esita. perplexity=1 e' perfetto; perplexity=|vocab| e' il caso casuale. 
    Piu' bassa e' meglio. E' la metrica standard dei modelli linguistici proprio perche' cattura la QUALITA' della distribuzione predetta, non solo il colpo esatto.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import RecurrentModel
from utils import timer
from utils import get_device, set_seed, project_paths, count_parameters
from windows import (NumericWindowDataset, TokenWindowDataset, train_val_split)
from models import ModelConfig, build_model
from make_numeric_datasets import load_txt

# Iperparametri dell'addestramento, raccolti e serializzabili.
@dataclass
class TrainConfig:
    max_epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0                 # clip del gradiente: argina l'esplosione
    
    # early stopping sulla loss di validazione
    patience: int = 5                      # epoche senza miglioramento prima di fermarsi
    min_delta: float = 1e-4                # miglioramento minimo per "contare"
    
    # soglia di accuracy (solo regressione): None = non usata come criterio di stop
    acc_threshold: Optional[float] = None  # es. 0.98
    acc_tolerance: float = 0.5             # |pred - vero| <= tol => "corretto" (regressione)

# Esito dell'addestramento: storie e sintesi per il benchmark.
@dataclass
class TrainResult:
    history: dict = field(default_factory=lambda: {"train_loss": [], "val_loss": [], "val_metric": []})
    epochs_run: int = 0
    epochs_to_threshold: Optional[int] = None   # regressione: epoca in cui si tocca la soglia
    best_val_loss: float = float("inf")
    best_val_metric: float = float("nan")       # best accuracy (regr.) o perplexity (classif.)
    train_time_s: float = 0.0

# Frazione di predizioni entro la tolleranza (spazio normalizzato).
def _regression_accuracy(pred: torch.Tensor, target: torch.Tensor,tol: float) -> float:
    correct = (torch.abs(pred - target) <= tol).float().mean().item()
    return correct

"""
Valuta su un loader. Ritorna (loss_media, metrica).
La metrica e' accuracy per la regressione, perplexity per la classificazione.
"""
@torch.no_grad()
def evaluate(model: RecurrentModel, loader: DataLoader, criterion,task: str, device, tol: float) -> tuple[float, float]:
    # start initialize
    model.eval()
    total_loss, n_batches = 0.0, 0
    acc_sum = 0.0
    # scorre per ogni elemento dato
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)       # mette nel dispositivo disponibile
        out = model(xb)
        if task == "classification":                # text dataset (prediction next token)
            loss = criterion(out, yb)               # yb: indici di classe
        else:                                       # mathematic regression
            loss = criterion(out, yb)               # yb: valori reali
            acc_sum += _regression_accuracy(out, yb, tol)
        # update   
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    if task == "classification":
        metric = math.exp(min(avg_loss, 20))       # perplexity (cap per stabilita')
    else:
        metric = acc_sum / max(n_batches, 1)       # accuracy media
    return avg_loss, metric

"""
Addestra un modello e restituisce il resoconto completo.
Criterio di perdita:
- classificazione: CrossEntropyLoss (confronta logit e classe vera)
- regressione    : MSELoss (errore quadratico medio sul valore)
Ottimizzatore: Adam (robusto e standard per reti ricorrenti).
Stabilizzazione: clip della norma del gradiente, cruciale per le RNN dove il gradiente puo' esplodere su sequenze lunghe.
"""
def train_model(model: RecurrentModel, train_loader: DataLoader,val_loader: DataLoader, task: str, cfg: TrainConfig,device, verbose: bool = True) -> TrainResult:
    
    model.to(device)
    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,weight_decay=cfg.weight_decay)

    result = TrainResult()
    epochs_no_improve = 0

    with timer(sync_cuda=True) as t:
        for epoch in range(1, cfg.max_epochs + 1):
            # ---------------- fase di addestramento ----------------
            model.train()
            running, n_batches = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                # clip della norma: evita passi distruttivi da gradienti enormi
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                running += loss.item()
                n_batches += 1
            train_loss = running / max(n_batches, 1)

            # ---------------- fase di validazione ----------------
            val_loss, val_metric = evaluate(model, val_loader, criterion,
                                            task, device, cfg.acc_tolerance)

            result.history["train_loss"].append(train_loss)
            result.history["val_loss"].append(val_loss)
            result.history["val_metric"].append(val_metric)
            result.epochs_run = epoch

            if verbose:
                mname = "ppl" if task == "classification" else "acc"
                print(f"epoca {epoch:3d} | train_loss={train_loss:.4f} "
                      f"val_loss={val_loss:.4f} val_{mname}={val_metric:.4f}")

            # ---------------- tracciamento del migliore ----------------
            if val_loss < result.best_val_loss - cfg.min_delta:
                result.best_val_loss = val_loss
                result.best_val_metric = val_metric
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # ---------------- epoche-a-soglia (solo regressione) ----------------
            if (task == "regression" and cfg.acc_threshold is not None
                    and result.epochs_to_threshold is None
                    and val_metric >= cfg.acc_threshold):
                result.epochs_to_threshold = epoch
                if verbose:
                    print(f"  -> soglia accuracy {cfg.acc_threshold:.2f} "
                          f"raggiunta all'epoca {epoch}")

            # ---------------- early stopping ----------------
            if epochs_no_improve >= cfg.patience:
                if verbose:
                    print(f"  -> early stopping (nessun miglioramento da "
                          f"{cfg.patience} epoche)")
                break

    result.train_time_s = t["elapsed"]
    return result



# CLI minimale: addestra UNA rete su UN dataset (per prove rapide)
def _cli() -> None:
    p = argparse.ArgumentParser(description="Addestra una singola rete ricorrente.")
    p.add_argument("--cell", choices=["rnn", "lstm", "gru"], default="lstm")
    p.add_argument("--task", choices=["regression", "classification"], required=True)
    p.add_argument("--data", required=True,
                   help="per regressione: file .txt numerico; per classificazione: file .npy di token-id")
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--acc-threshold", type=float, default=None)
    p.add_argument("--vocab-size", type=int, default=None,
                   help="classificazione: |vocab| (letto dal vocab se omesso)")
    p.add_argument("--save-to", default=None,
                   help="se indicato, salva un checkpoint .pt usabile da generate.py")
    args = p.parse_args()

    set_seed(42)
    device = get_device()

    # costruzione del dataset a seconda del task
    if args.task == "regression":
        series = load_txt(args.data)
        ds = NumericWindowDataset(series, seq_len=args.seq_len, stride=args.stride)
        cfg = ModelConfig(cell_type=args.cell, task="regression",
                          input_size=1, hidden_size=args.hidden, num_layers=args.layers)
    else:
        ids = np.load(args.data)
        vocab = args.vocab_size or int(ids.max()) + 1
        ds = TokenWindowDataset(ids, seq_len=args.seq_len, stride=args.stride)
        cfg = ModelConfig(cell_type=args.cell, task="classification",
                          vocab_size=vocab, hidden_size=args.hidden, num_layers=args.layers)

    train_ds, val_ds = train_val_split(ds, val_frac=0.2)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    model = build_model(cfg)
    print(f"[train] {args.cell} su {os.path.basename(args.data)} | "
          f"parametri={count_parameters(model):,}")

    tcfg = TrainConfig(max_epochs=args.epochs, batch_size=args.batch,
                       acc_threshold=args.acc_threshold)
    res = train_model(model, train_loader, val_loader, cfg.task, tcfg, device)

    mname = "perplexity" if cfg.task == "classification" else "accuracy"
    print(f"\n[risultato] epoche={res.epochs_run} tempo={res.train_time_s:.1f}s "
          f"best_val_loss={res.best_val_loss:.4f} best_{mname}={res.best_val_metric:.4f}")
    if res.epochs_to_threshold:
        print(f"[risultato] epoche alla soglia: {res.epochs_to_threshold}")

    # salvataggio opzionale del checkpoint (pesi + configurazione) per l'inferenza
    if args.save_to:
        from generate import save_checkpoint
        save_checkpoint(model, cfg, args.save_to)

if __name__ == "__main__":
    _cli()                  # richiama la CLI
