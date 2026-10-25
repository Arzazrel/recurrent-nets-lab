# -*- coding: utf-8 -*-
"""
benchmark.py

Confronto fra RNN, LSTM e GRU.
Si verifica:
1. dato un dataset e i suoi parametri (seq_len, stride) e un budget di addestramento (epoche max, early stopping), che accuracy/perplexity raggiunge ciascuna rete?
2. quante epoche impiega ciascuna a toccare una soglia di accuracy (solo task numerici)?
3. qual e' la complessita' di ciascuna in PARAMETRI e in TEMPO DI INFERENZA?

Tutte e tre le reti condividono dataset, split, iperparametri e seme: le differenze osservate sono quindi attribuibili alla cella.
Output in results/:
- <nome>_curve.png    : curve di loss/metrica di validazione per epoca
- <nome>_summary.csv  : tabella riassuntiva (parametri, tempi, metriche)
- stampa a schermo della tabella
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from make_numeric_datasets import load_txt
from models import ModelConfig, build_model
from train import TrainConfig, train_model
from utils import (count_parameters, get_device, project_paths, set_seed, timer)
from windows import NumericWindowDataset, TokenWindowDataset, train_val_split

CELLS = ("rnn", "lstm", "gru")  # tipi di cella

"""
Misura il tempo medio di inferenza per batch (in millisecondi).
Dettagli metodologici per una misura corretta su GPU:
- si esegue un "warm-up" ignorato (la prima esecuzione CUDA paga la compilazione/allocazione dei kernel e falserebbe la media);
- si usa il timer con cuda.synchronize() (vedi utils) perche' le chiamate CUDA sono asincrone;
- si media su piu' ripetizioni per ridurre il rumore.
"""
@torch.no_grad()
def measure_inference_time(model, loader, device, repeats: int = 3) -> float:
   
    model.eval()
    batch = next(iter(loader))[0].to(device)

    # warm-up (non misurato)
    for _ in range(2):
        _ = model(batch)

    times = []
    for _ in range(repeats):
        with timer(sync_cuda=True) as t:
            _ = model(batch)
        times.append(t["elapsed"])
    return float(np.mean(times) * 1000.0)        # ms

# Costruisce (train_loader, val_loader, cfg_parziale) secondo il task.
def build_datasets(task: str, data_path: str, seq_len: int, stride: int, log_transform: bool = False):
    # discrimina in base al tipo di task (numerico o testuale)
    if task == "regression":            # dataset numerici
        series = load_txt(data_path)
        ds = NumericWindowDataset(series, seq_len=seq_len, stride=stride, log_transform=log_transform)
        extra = {"input_size": 1}
    else:                               # dataset testuali
        ids = np.load(data_path)
        vocab = int(ids.max()) + 1
        ds = TokenWindowDataset(ids, seq_len=seq_len, stride=stride)
        extra = {"vocab_size": vocab}
    return ds, extra

# Esegue il confronto completo sulle tre reti e ritorna una lista di record.
def run_benchmark(task: str, data_path: str, seq_len: int, stride: int, hidden: int, layers: int, epochs: int, batch: int, acc_threshold: Optional[float], seed: int = 42, log_transform: bool = False) -> list[dict]:

    device = get_device()
    paths = project_paths()
    name = os.path.splitext(os.path.basename(data_path))[0]

    records = []
    histories = {}

    for cell in CELLS:
        set_seed(seed)      # reset del seme PRIMA di ogni rete: stessa inizializzazione e stesso ordine di shuffling -> confronto equo
     
        ds, extra = build_datasets(task, data_path, seq_len, stride, log_transform) # get dataset
        train_ds, val_ds = train_val_split(ds, val_frac=0.2, seed=seed)             # get training and validation set
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)         # initialize train loader
        val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)            # initialize validation loader

        cfg = ModelConfig(cell_type=cell, task=task, hidden_size=hidden,num_layers=layers, **extra) # config the model
        model = build_model(cfg)
        n_params = count_parameters(model)

        print(f"\n{'#'*60}\n# {cell.upper()} su {name}  (params={n_params:,})\n{'#'*60}")
        tcfg = TrainConfig(max_epochs=epochs, batch_size=batch,acc_threshold=acc_threshold) # set paameters
        res = train_model(model, train_loader, val_loader, task, tcfg, device)              # train model

        infer_ms = measure_inference_time(model, val_loader, device)

        records.append({
            "dataset": name, "cell": cell, "params": n_params,
            "epochs_run": res.epochs_run,
            "epochs_to_threshold": res.epochs_to_threshold,
            "best_val_loss": round(res.best_val_loss, 5),
            "best_val_metric": round(res.best_val_metric, 5),
            "metric_name": "perplexity" if task == "classification" else "accuracy",
            "train_time_s": round(res.train_time_s, 2),
            "infer_ms_per_batch": round(infer_ms, 3),
        })
        histories[cell] = res.history

    _plot_curves(histories, task, name, paths["results"])
    _write_csv(records, name, paths["results"])
    _print_table(records)
    return records


# ------------------------------- Report: grafici, CSV, tabella a schermo -------------------------------

# Salva le curve di validazione (loss e metrica) per le tre reti.
def _plot_curves(histories: dict, task: str, name: str, out_dir: str) -> None:

    try:
        import matplotlib
        matplotlib.use("Agg")                    # backend senza display (headless/WSL)
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib non installato: salto i grafici")
        return

    metric_label = "perplexity" if task == "classification" else "accuracy"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for cell, hist in histories.items():
        epochs = range(1, len(hist["val_loss"]) + 1)
        ax1.plot(epochs, hist["val_loss"], marker="o", label=cell.upper())
        ax2.plot(epochs, hist["val_metric"], marker="o", label=cell.upper())
    ax1.set_title(f"Validation loss - {name}")
    ax1.set_xlabel("epoca"); ax1.set_ylabel("loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_title(f"Validation {metric_label} - {name}")
    ax2.set_xlabel("epoca"); ax2.set_ylabel(metric_label); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, f"{name}_curve.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[plot] salvato {out}")

# Salva la tabella riassuntiva in CSV, per analisi successive.
def _write_csv(records: list[dict], name: str, out_dir: str) -> None:
    out = os.path.join(out_dir, f"{name}_summary.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"[csv] salvato {out}")

# Stampa a schermo un confronto leggibile.
def _print_table(records: list[dict]) -> None:
    print("\n" + "=" * 78)
    print(f"{'cella':6s}{'params':>12s}{'epoche':>8s}{'->soglia':>10s}"
          f"{'val_loss':>11s}{'metrica':>11s}{'train_s':>10s}{'infer_ms':>10s}")
    print("-" * 78)
    for r in records:
        thr = r["epochs_to_threshold"] if r["epochs_to_threshold"] else "-"
        print(f"{r['cell'].upper():6s}{r['params']:>12,}{r['epochs_run']:>8d}"
              f"{str(thr):>10s}{r['best_val_loss']:>11.4f}"
              f"{r['best_val_metric']:>11.4f}{r['train_time_s']:>10.2f}"
              f"{r['infer_ms_per_batch']:>10.3f}")
    print("=" * 78)
    print(f"(metrica = {records[0]['metric_name']}; "
          f"{'piu bassa e meglio' if records[0]['metric_name']=='perplexity' else 'piu alta e meglio'})")


def main() -> None:
    p = argparse.ArgumentParser(description="Confronta RNN, LSTM e GRU sullo stesso dataset.")
    p.add_argument("--task", choices=["regression", "classification"], required=True)
    p.add_argument("--data", required=True,
                   help="regressione: .txt numerico; classificazione: .npy di token-id")
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--acc-threshold", type=float, default=0.98,
                   help="solo regressione: soglia di accuracy per 'epoche-a-soglia'")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-transform", action="store_true",
                   help="regressione: applica log1p (per serie esponenziali, es. Fibonacci)")
    args = p.parse_args()

    run_benchmark(task=args.task, data_path=args.data, seq_len=args.seq_len,
                  stride=args.stride, hidden=args.hidden, layers=args.layers,
                  epochs=args.epochs, batch=args.batch,
                  acc_threshold=args.acc_threshold if args.task == "regression" else None,
                  seed=args.seed, log_transform=args.log_transform)

if __name__ == "__main__":
    main()
