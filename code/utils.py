# -*- coding: utf-8 -*-
"""
utils.py
Funzioni di utilità condivise da tutti i moduli del laboratorio.
"""

from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch


 """
Restituisce il device di calcolo migliore disponibile.
La priorita' e' la GPU CUDA, se non e' disponibile si ripiega sulla CPU, cosi' il codice resta eseguibile ovunque.
"""
def get_device(verbose: bool = True) -> torch.device:
   
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            name = torch.cuda.get_device_name(0)
            # capability = (major, minor), es. (12, 0) per Blackwell
            cap = torch.cuda.get_device_capability(0)
            print(f"[device] CUDA attivo: {name} (sm_{cap[0]}{cap[1]})")
            print(f"[device] torch {torch.__version__}, CUDA runtime {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("[device] CUDA non disponibile: uso la CPU")
    return device


"""
Fissa i generatori pseudo-casuali di Python, NumPy e PyTorch. Fissiamo i semi per avee riproducibilità per confronti migliori.
Con deterministic=True si forzano anche gli algoritmi cuDNN deterministici: elimina l'ultima fonte di non-determinismo su GPU al costo di un po' di velocita'.
"""
def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


"""
Conta i parametri (scalari) di un modello.
E' una delle metriche di confronto del laboratorio: a parita' di capacita' nominale (hidden_size, numero di layer) RNN, LSTM e GRU hanno un numero di parametri diverso perche' le celle hanno un numero diverso di "gate".
Regola pratica per un singolo layer con input I e stato H:
- RNN  : ~ 1 * (I*H + H*H + H)   (una trasformazione)
- GRU  : ~ 3 * (I*H + H*H + H)   (gate reset, update, candidato)
- LSTM : ~ 4 * (I*H + H*H + H)   (gate input, forget, output, cella)
"""
def count_parameters(model: torch.nn.Module, only_trainable: bool = True) -> int:
    
    params = model.parameters()             # get the parameters
    if only_trainable:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)   # return the sum of the parameters


"""
Context manager per misurare il tempo di un blocco di codice.
Uso:
    with timer("training") as t:
        ...                     # codice da misurare
    print(t["elapsed"])         # secondi trascorsi

Dettaglio su GPU: le chiamate CUDA sono asincrone. Senza una torch.cuda.synchronize() misureremmo solo il tempo di *accodamento* dei kernel, non quello di esecuzione. 
sync_cuda=True garantisce che la GPU abbia finito prima di fermare il cronometro: indispensabile per misurare correttamente i tempi di inferenza richiesti dal benchmark.
"""
@contextmanager
def timer(label: str = "", sync_cuda: bool = True) -> Iterator[dict]:
    
    result: dict = {}
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield result
    finally:
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        result["elapsed"] = time.perf_counter() - start
        if label:
            print(f"[timer] {label}: {result['elapsed']:.4f} s")


"""
Restituisce i percorsi assoluti delle cartelle del progetto.
Risale di un livello da code/ per raggiungere la radice del repository e da li' individua dataset/ e results/. Evita percorsi relativi fragili che dipenderebbero dalla directory di lavoro corrente.
"""
def project_paths() -> dict:
    
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    paths = {
        "root": root,
        "code": here,
        "dataset": os.path.join(root, "dataset"),
        "results": os.path.join(root, "results"),
    }
    for key in ("dataset", "results"):
        os.makedirs(paths[key], exist_ok=True)
    return paths

# eseguendo direttamente questo file se ne verifica il corretto funzionamento isolato, utile per controllare l'ambiente PyTorch/CUDA.
if __name__ == "__main__":
    dev = get_device()                              # get device
    set_seed(0)                                     # set seed
    print("[utils] percorsi:", project_paths())     # see the paths
    with timer("matmul di prova"):                  # timer and parameters
        a = torch.randn(1024, 1024, device=dev)
        _ = a @ a
