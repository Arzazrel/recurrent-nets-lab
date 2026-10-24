# -*- coding: utf-8 -*-
"""
make_numeric_datasets.py

Generazione dei dataset numerici sintetici e loro salvataggio in formato .txt.

I dataset vengono salvati come .txt (un valore per riga, con intestazione di commento) cosi' da essere ispezionabili a occhio e ricaricabili quando serve.
"""

from __future__ import annotations

import argparse
import os
from typing import Callable
import numpy as np

from utils import project_paths, set_seed

# -------------------------------------- Generatori di sequenza --------------------------------------

# Sequenza aritmetica crescente: a[t] = start + t*step. Dipendenza ordine 1.
def seq_increasing(n: int, step: int = 1, start: int = 0) -> np.ndarray:
    return start + step * np.arange(n, dtype=np.float64)

# Sequenza aritmetica decrescente: a[t] = start - t*step. Dipendenza ordine 1.
def seq_decreasing(n: int, step: int = 1, start: int = 0) -> np.ndarray:
    return start - step * np.arange(n, dtype=np.float64)

"""
Fibonacci: a[t] = a[t-1] + a[t-2]. Richiede memoria di due passi.
Cresce esponenzialmente: senza normalizzazione i valori esplodono e saturano la rete, qui salviamo i valori grezzi per ispezionabilita'.
"""
def seq_fibonacci(n: int) -> np.ndarray:
    
    a = np.zeros(n, dtype=np.float64)
    if n > 1:
        a[1] = 1.0
    for t in range(2, n):
        a[t] = a[t - 1] + a[t - 2]
    return a

"""
Onda sinusoidale (opz. con rumore gaussiano). 
Dipendenza periodica: limitato in ampiezza e quindi stabile numericamente. Rappresenta un segnale continuo.
    """
def seq_sine(n: int, period: float = 20.0, noise: float = 0.0) -> np.ndarray:
    
    t = np.arange(n, dtype=np.float64)
    s = np.sin(2.0 * np.pi * t / period)
    if noise > 0.0:
        s = s + np.random.normal(0.0, noise, size=n)
    return s

"""
Copy task a ritardo fisso: la rete deve riprodurre il valore visto 'delay' passi prima. 
E' il banco di prova classico per la memoria a lungo termine: aumentando delay si mette in crisi la RNN semplice (gradiente che svanisce) mentre LSTM/GRU,
grazie ai gate, mantengono il segnale. E' il task che meglio mostra *perche'* LSTM e GRU sono state inventate.

Costruzione: una sorgente casuale in [0, vocab); l'obiettivo e' la sorgente ritardata di 'delay'. 
Salviamo la sola sorgente; il ritardo e' applicato a livello di finestra in fase di addestramento.
"""
def seq_copy_task(n: int, delay: int = 10, vocab: int = 10) -> np.ndarray:
    return np.random.randint(0, vocab, size=n).astype(np.float64)

# -------------------------------------- Salvataggio --------------------------------------
"""
Salva una sequenza in .txt, un valore per riga, con intestazione di commento.
Le righe che iniziano con '#' sono metadati (regola generatrice, parametri): il loader le ignora. 
"""
def save_txt(path: str, values: np.ndarray, header: str) -> None:
    # apre o crea il documento dato come parametro
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {header}\n")
        f.write(f"# count={len(values)}\n")
        for v in values:                        # scorri tutti i valori
            # interi quando possibile, per leggibilita'
            if float(v).is_integer():
                f.write(f"{int(v)}\n")
            else:
                f.write(f"{v:.6f}\n")
    print(f"[save] {os.path.basename(path):28s} ({len(values)} valori)")    # control print

# Ricarica una sequenza salvata con save_txt, ignorando i commenti '#'.
def load_txt(path: str) -> np.ndarray:
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals.append(float(line))
    return np.asarray(vals, dtype=np.float64)

# -------------------------------------- Catalogo dei dataset da generare --------------------------------------
"""
Definisce l'insieme dei dataset numerici del laboratorio.
Ogni voce e' (array, descrizione). La descrizione finisce nell'header del .txt e documenta la regola generatrice. 'n' e' la lunghezza della serie.
"""
def build_catalog(n: int) -> dict[str, tuple[np.ndarray, str]]:
    
    return {
        "num_increasing_1": (seq_increasing(n, step=1),
                             "crescente a passo +1 (markov ordine 1)"),
        "num_increasing_N": (seq_increasing(n, step=3),
                             "crescente a passo +3 (markov ordine 1)"),
        "num_decreasing_1": (seq_decreasing(n, step=1, start=n),
                             "decrescente a passo -1 (markov ordine 1)"),
        "num_fibonacci":    (seq_fibonacci(n),
                             "Fibonacci a[t]=a[t-1]+a[t-2] (memoria 2 passi)"),
        "num_sine":         (seq_sine(n, period=20.0, noise=0.02),
                             "sinusoide periodo=20 + rumore 0.02 (periodica)"),
        "num_copy_delay10": (seq_copy_task(n, delay=10, vocab=10),
                             "copy task sorgente, ritardo=10 (memoria lunga)"),
    }

# funzione main che prende gli argomenti passati e genera i dataset
def main() -> None:
    parser = argparse.ArgumentParser(description="Genera i dataset numerici sintetici.")
    parser.add_argument("--n", type=int, default=2000,
                        help="lunghezza di ogni serie (default: 2000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    paths = project_paths()
    out_dir = os.path.join(paths["dataset"], "numeric") # set the dataset folder name
    os.makedirs(out_dir, exist_ok=True)                 # create folder

    print(f"[make] genero i dataset numerici (n={args.n}) in {out_dir}")
    catalog = build_catalog(args.n)
    for name, (values, desc) in catalog.items():
        save_txt(os.path.join(out_dir, f"{name}.txt"), values, header=desc)
    print("[make] completato.")


if __name__ == "__main__":
    main()
