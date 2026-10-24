# -*- coding: utf-8 -*-
"""
models.py

Le tre reti ricorrenti complete, dietro UN'UNICA classe parametrizzabile.

Scelta architetturale centrale: invece di tre classi separate, una sola `RecurrentModel` che seleziona il tipo di cella (RNN/LSTM/GRU) e la testa di output (regressione/classificazione). 

Parametri esposti (adattano la rete ai vari scenari):
    cell_type      : "rnn" | "lstm" | "gru"   -> quale cella ricorrente
    task           : "regression" | "classification"
    input_size     : feature in ingresso (1 per numerico; ignorato se c'e' embedding)
    vocab_size     : |vocab| (solo classificazione: attiva nn.Embedding + testa |vocab|)
    embed_dim      : dimensione degli embedding (solo classificazione)
    hidden_size    : ampiezza dello stato nascosto (capacita' della cella)
    num_layers     : profondita' (celle impilate)
    dropout        : regolarizzazione fra i layer ricorrenti
    bidirectional  : doppio verso (SOLO per task non causali; vedi nota sotto)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from utils import count_parameters, set_seed

import torch
import torch.nn as nn


# Mappa nome -> classe PyTorch del layer ricorrente. L'interfaccia quasi identica dei tre layer permette un unica classe.
_RNN_CLASSES = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

"""
Configurazione del modello, raccolta in un unico oggetto serializzabile.
Usare una dataclass (anziche' una lista di argomenti) rende la configurazione salvabile insieme ai pesi e riproducibile.
"""
@dataclass
class ModelConfig:
    
    cell_type: str = "lstm"             # RNN type
    task: str = "classification"        # "regression" | "classification"
    input_size: int = 1                 # usato solo in regressione
    vocab_size: Optional[int] = None    # usato solo in classificazione
    embed_dim: int = 64                 # usato solo in classificazione
    hidden_size: int = 128              # neuron of hidden layer (head)
    num_layers: int = 1                 # number of hidden layer
    dropout: float = 0.0                # from 0 to 1
    bidirectional: bool = False

    def __post_init__(self):
        assert self.cell_type in _RNN_CLASSES, f"cell_type non valido: {self.cell_type}"
        assert self.task in ("regression", "classification")
        if self.task == "classification":
            assert self.vocab_size is not None, "classificazione richiede vocab_size"

"""
Modello ricorrente unificato. Flusso in avanti:
1. se classificazione: id -> Embedding -> vettori densi
   se regressione    : si usa direttamente la feature (batch, seq, input_size)
2. la sequenza passa nella cella ricorrente scelta
3. si prende lo stato dell'ULTIMO passo temporale (predizione causale del "prossimo" valore/token)
4. una testa lineare proietta sullo spazio di output
"""
class RecurrentModel(nn.Module):
    
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # --- 1. Encoder dell'input -------------------------------------------
        if cfg.task == "classification":
            # testo: tabella di embedding apprendibile (id -> vettore)
            self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
            rnn_input_size = cfg.embed_dim
        else:
            # numerico: nessun embedding, la feature entra cosi' com'e'
            self.embedding = None
            rnn_input_size = cfg.input_size

        # --- 2. Cella ricorrente (selezionata dinamicamente) -----------------
        rnn_cls = _RNN_CLASSES[cfg.cell_type]
        self.rnn = rnn_cls(
            input_size=rnn_input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,                                   # forma (batch, seq, feature)
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0, # dropout ha effetto solo con num_layers>1 (agisce TRA i layer)
            bidirectional=cfg.bidirectional,
        )

        # --- 3. Testa di output ----------------------------------------------
        head_in = cfg.hidden_size * (2 if cfg.bidirectional else 1) # con bidirezionalita' lo stato finale concatena i due versi -> x2
        # add the output layer
        if cfg.task == "classification":
            self.head = nn.Linear(head_in, cfg.vocab_size)          # logit per classe (classification, next token prediction)
        else:
            self.head = nn.Linear(head_in, 1)                       # un valore reale

    """
    Estrae la rappresentazione dell'ultimo passo temporale da 'output'.

    'output' ha forma (batch, seq, hidden*dir). Per la predizione causale ci interessa l'ultimo passo. Usiamo output[:, -1, :] invece di h_n perche':
    (a) e' uniforme fra RNN/LSTM/GRU (h_n ha forme/firme diverse: la LSTM restituisce anche c_n), cosi' il codice resta identico per le tre;
    (b) con bidirectional include correttamente entrambi i versi gia' concatenati nell'ultimo passo.
    """
    def _last_hidden(self, output: torch.Tensor) -> torch.Tensor:
        
        return output[:, -1, :]

    """
    x:
        classificazione -> (batch, seq) di id interi (long)
        regressione     -> (batch, seq, input_size) di valori reali
    ritorna:
        classificazione -> (batch, vocab_size) logit
        regressione     -> (batch, 1) valore predetto
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.embedding is not None:
            x = self.embedding(x)                  # (batch, seq) -> (batch, seq, embed)

        # nn.RNN/GRU ritornano (output, h_n); nn.LSTM ritorna (output, (h_n, c_n)).
        output, _ = self.rnn(x)
        last = self._last_hidden(output)           # (batch, hidden*dir)
        return self.head(last)

# Factory esplicita: punto unico di creazione, comodo per i test e il log.
def build_model(cfg: ModelConfig) -> RecurrentModel:
    return RecurrentModel(cfg)

# main and simple test
if __name__ == "__main__":
    
    set_seed(0)

    print("=== classificazione (testo, |vocab|=50) ===")
    for ct in ("rnn", "lstm", "gru"):
        cfg = ModelConfig(cell_type=ct, task="classification",vocab_size=50, embed_dim=32, hidden_size=64, num_layers=2)
        m = build_model(cfg)
        out = m(torch.randint(0, 50, (4, 20)))     # (batch=4, seq=20)
        print(f"{ct:5s}: out={tuple(out.shape)}  params={count_parameters(m):,}")

    print("\n=== regressione (numerico, input_size=1) ===")
    for ct in ("rnn", "lstm", "gru"):
        cfg = ModelConfig(cell_type=ct, task="regression",input_size=1, hidden_size=64, num_layers=2)
        m = build_model(cfg)
        out = m(torch.randn(4, 20, 1))
        print(f"{ct:5s}: out={tuple(out.shape)}  params={count_parameters(m):,}")
