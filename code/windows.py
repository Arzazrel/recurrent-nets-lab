# -*- coding: utf-8 -*-
"""
windows.py
==========
Costruzione delle finestre scorrevoli (sliding windows) e wrapper Dataset.

Una rete ricorrente non consuma l'intera serie in un colpo solo: la si divide in finestre di lunghezza fissa 'seq_len' che scorrono con passo 'stride'. 
Ogni finestra e' un campione di addestramento. Questo file e' il punto unico in cui avviene questa divisione, condiviso da task numerico e task testuale.
'seq_len' e' la lunghezza del contesto che la rete vede prima di predire, e 'stride' il passo di scorrimento. 
E' questo il parametro che il laboratorio rende configurabile per adattare la rete ai vari scenari.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

"""
Normalizzazione numerica

Standardizza una serie: (x - media) / deviazione_standard.

Motivo: alcune serie (Fibonacci su tutte) crescono in modo esplosivo. Dati di input con magnitudine enorme saturano le attivazioni (tanh/sigmoid) e
rendono instabile la discesa del gradiente. Portando media 0 e varianza 1 si stabilizza l'addestramento. 
Restituiamo anche media e std per poter de-normalizzare le predizioni al momento dell'inferenza.
"""
def standardize(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    
    mu = float(x.mean())
    sigma = float(x.std()) or 1.0       # evita divisione per zero su serie costanti
    return (x - mu) / sigma, mu, sigma


"""
Dataset di finestre per la regressione sul valore successivo.

Dato l'array x e seq_len=L, il campione i-esimo e':
    input  = x[i : i+L]          (L valori, feature scalare)           <- finestra in input
    target = x[i+L]              (il valore immediatamente successivo) <- output desiderato per la finestra

Per il copy task con ritardo k l'obiettivo e' invece x[i+L-1-k], cioe' il valore visto k passi prima della fine della finestra: parametro 'delay'.

Forma dei tensori restituiti:
    input  : (seq_len, 1)   -> 1 e' la dimensione della feature
    target : (1,)
La dimensione di feature esplicita serve perche' nn.RNN/LSTM/GRU si aspettano input (seq, batch, feature) oppure (batch, seq, feature).
"""
class NumericWindowDataset(Dataset):
    # inizializzazione
    def __init__(self, x: np.ndarray, seq_len: int, stride: int = 1,delay: Optional[int] = None, normalize: bool = True):
        self.seq_len = seq_len                          # windows len
        self.mu, self.sigma = 0.0, 1.0                  # mean 0 and vaiance 1
        if normalize:                                   # check if normalise
            x, self.mu, self.sigma = standardize(x.astype(np.float64))
        self.x = torch.tensor(x, dtype=torch.float32)
        self.delay = delay                              # windows stride

        # Indici di partenza validi delle finestre, pre-calcolati una volta sola.
        last_start = len(x) - seq_len - 1
        self.starts = list(range(0, last_start + 1, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = self.starts[idx]
        window = self.x[s: s + self.seq_len].unsqueeze(-1)  # (L, 1)
        if self.delay is None:
            target = self.x[s + self.seq_len].unsqueeze(0)  # valore successivo
        else:
            # copy task: il target e' il valore a 'delay' passi dalla fine
            target = self.x[s + self.seq_len - 1 - self.delay].unsqueeze(0)
        return window, target

"""
Dataset di finestre per la predizione del token successivo. (classificazione)
Dato un array di token-id interi e seq_len=L:
    input  = ids[i : i+L]        (L id -> entreranno in un nn.Embedding)
    target = ids[i+L]            (l'id del token successivo, da predire)

A differenza del caso numerico il target e' un indice di classe (uno fra |vocab| possibili), quindi il modello produrra' logit su |vocab| classi e si usera' la cross-entropy. 
Forma dei tensori:
    input  : (seq_len,)   dtype long   (indici, non valori)
    target : ()           dtype long
"""
class TokenWindowDataset(Dataset):
    # inizializzazione
    def __init__(self, ids: np.ndarray, seq_len: int, stride: int = 1):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len
        last_start = len(ids) - seq_len - 1
        self.starts = list(range(0, last_start + 1, stride))

    def __len__(self) -> int:
        return len(self.starts)

    # restituisce la finestra di input e il token di output
    def __getitem__(self, idx: int):
        s = self.starts[idx]
        window = self.ids[s: s + self.seq_len]         # finestra di input (L,)
        target = self.ids[s + self.seq_len]            # output (scalare)
        return window, target

# divide un dataset in addestramento/validazione in modo riproducibile.
def train_val_split(dataset: Dataset, val_frac: float = 0.2, seed: int = 42):
    n = len(dataset)
    n_val = int(n * val_frac)
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n_train, n_val], generator=gen)
