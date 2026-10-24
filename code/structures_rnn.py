# -*- coding: utf-8 -*-
"""
structures_rnn.py

Costruzione DIDATTICA della RNN semplice (Elman).

Questo file non e' pensato per il training: e' un percorso guidato che mostra, con print di verifica a ogni passo, le strutture PyTorch utili fino a comporre una RNN completa.
Eseguirlo (`python structures_rnn.py`) stampa forme dei tensori e output intermedi, cosi' da capire COSA fa ogni componente prima di usarlo nel modello vero (models_rnn.py).

Idea della RNN (Elman): a ogni passo temporale t lo stato nascosto si aggiorna:  h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)
cioe' una sola trasformazione che mescola input corrente e stato precedente.
Da qui il "problema" della RNN semplice: su sequenze lunghe il gradiente che torna indietro viene moltiplicato molte volte per W_hh e tende a svanire (o esplodere).
LSTM e GRU nascono per risolvere questo -> vedi structures_lstm.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from utils import set_seed

"""
nn.Embedding: dalla sequenza di id a vettori densi.
Questo e' nn.Embedding(|vocab|, d): una matrice (|vocab| x d) indicizzata.
"""
def step_1_embedding() -> None:
    
    print("\n--- Passo 1: nn.Embedding ---")
    vocab_size, embed_dim = 20, 8
    emb = nn.Embedding(vocab_size, embed_dim)
    ids = torch.tensor([[1, 5, 7, 7]])              # (batch=1, seq=4)
    vecs = emb(ids)                                 # (1, 4, 8)
    print(f"input id         : {tuple(ids.shape)}  -> {ids.tolist()}")
    print(f"output embedding : {tuple(vecs.shape)}  (batch, seq, embed_dim)")
    print(f"la tabella ha    : {emb.weight.shape[0]}x{emb.weight.shape[1]} parametri")

"""
nn.RNNCell: UN singolo passo temporale. RNNCell calcola h_t da (x_t, h_{t-1}). 
Iterando sulla sequenza si vede esplicitamente la ricorrenza: lo stato viene propagato di passo in passo. E' il cuore concettuale della rete.
"""
def step_2_rnn_cell() -> None:
    
    print("\n--- Passo 2: nn.RNNCell (ricorrenza manuale) ---")
    input_size, hidden_size = 8, 16
    cell = nn.RNNCell(input_size, hidden_size)
    seq = torch.randn(4, input_size)                # 4 passi temporali, 1 esempio
    h = torch.zeros(hidden_size)                    # stato iniziale a zero
    for t, x_t in enumerate(seq):
        h = cell(x_t.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
        print(f"passo t={t}: |h|={h.norm():.3f}  (lo stato evolve nel tempo)")

"""
nn.RNN: l'intero layer che itera sulla sequenza per noi.
nn.RNN fa internamente il ciclo del passo 2, in modo ottimizzato (ed eseguibile su GPU). 
Restituisce:
- output : lo stato nascosto a OGNI passo temporale (per task seq2seq)
- h_n    : lo stato nascosto FINALE (ci serve per predire 1 valore)
Il flag batch_first=True impone la forma (batch, seq, feature), piu' comoda.
"""
def step_3_rnn_layer() -> None:
    
    print("\n--- Passo 3: nn.RNN (layer completo) ---")
    rnn = nn.RNN(input_size=8, hidden_size=16, num_layers=1, batch_first=True)
    x = torch.randn(2, 4, 8)                         # (batch=2, seq=4, feat=8)
    output, h_n = rnn(x)
    print(f"output (tutti i passi): {tuple(output.shape)}  (batch, seq, hidden)")
    print(f"h_n (stato finale)    : {tuple(h_n.shape)}  (num_layers, batch, hidden)")
    print("per predire UN valore si usa l'ultimo passo: output[:, -1, :]")

"""
impilamento e bidirezionalita': i parametri che rendono la rete piu' capace (e piu' pesante).
- num_layers>1 : impila RNN; l'output di una alimenta la successiva -> piu' capacita' di astrazione, piu' parametri.
- bidirectional=True : due passaggi, avanti e indietro; lo stato finale concatena entrambe le direzioni -> utile se il futuro aiuta a interpretare il presente (es. classificazione di sequenze intere) NON per la generazione causale.
"""
def step_4_stacked_bidirectional() -> None:
    
    print("\n--- Passo 4: stacking + bidirezionalita' ---")
    for nl, bidir in [(1, False), (3, False), (2, True)]:
        rnn = nn.RNN(8, 16, num_layers=nl, bidirectional=bidir, batch_first=True)
        x = torch.randn(1, 4, 8)
        out, h_n = rnn(x)
        n_par = sum(p.numel() for p in rnn.parameters())
        print(f"layers={nl} bidir={bidir}: out={tuple(out.shape)} "
              f"h_n={tuple(h_n.shape)} parametri={n_par}")

"""
la "testa" di output: dallo stato finale alla predizione.
- regressione : Linear(hidden, 1)                       -> un valore reale
- classificazione (next-token): Linear(hidden, |vocab|) -> logit per classe
Qui componiamo il pezzo finale che chiude la rete.
"""
def step_5_compose_head() -> None:
    
    print("\n--- Passo 5: testa di output (nn.Linear) ---")
    hidden_size = 16
    rnn = nn.RNN(8, hidden_size, batch_first=True)
    head_reg = nn.Linear(hidden_size, 1)
    head_cls = nn.Linear(hidden_size, 20)           # 20 = |vocab| d'esempio

    x = torch.randn(1, 4, 8)
    _, h_n = rnn(x)
    last = h_n[-1]                                   # (batch, hidden): ultimo layer
    print(f"predizione regressione    : {tuple(head_reg(last).shape)}  (batch, 1)")
    print(f"predizione classificazione: {tuple(head_cls(last).shape)}  (batch, |vocab|)")
    print("\nMessa insieme: Embedding? -> RNN -> prendi ultimo stato -> Linear.")
    print("Questa catena e' cio' che models_rnn.py implementa in modo parametrico.")


if __name__ == "__main__":
    set_seed(0)
    print("=" * 70)
    print("COSTRUZIONE DIDATTICA DELLA RNN SEMPLICE")
    print("=" * 70)
    step_1_embedding()
    step_2_rnn_cell()
    step_3_rnn_layer()
    step_4_stacked_bidirectional()
    step_5_compose_head()
