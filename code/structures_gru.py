# -*- coding: utf-8 -*-
"""
structures_gru.py

Costruzione DIDATTICA della GRU (Gated Recurrent Unit), passo per passo.
La GRU e' una semplificazione della LSTM: mantiene la capacita' di controllare il flusso di memoria tramite gate, ma con DUE gate invece di tre e UN SOLO stato (niente stato di cella separato).
I suoi gate sono:
- reset gate  r_t : quanto dello stato passato "dimenticare" nel calcolo del candidato (quanto il passato influenza il presente)
- update gate z_t : interpola fra stato precedente e candidato nuovo (fa il lavoro combinato di forget+input della LSTM)
Conseguenza: ~3 trasformazioni interne -> ~3x parametri rispetto alla RNN semplice (contro ~4x della LSTM). 
Spesso ottiene accuratezza paragonabile alla LSTM con meno parametri e addestramento piu' rapido.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils import set_seed

"""
un SOLO stato, come la RNN (a differenza della LSTM).
La GRU non ha stato di cella: tutta la memoria vive nello stato nascosto h, ma il suo aggiornamento e' regolato dai gate. 
E' il motivo per cui e' piu' leggera della LSTM pur restando "gated".
"""
def step_1_single_state() -> None:
    
    print("\n--- Passo 1: un unico stato nascosto (niente cella) ---")
    hidden_size = 16
    h0 = torch.zeros(1, 1, hidden_size)
    print(f"h0: {tuple(h0.shape)}  (unico stato; la LSTM ne aveva due)")

"""
nn.GRUCell: stato singolo che evolve.
""" 
def step_2_gru_cell() -> None:
    
    print("\n--- Passo 2: nn.GRUCell (ricorrenza manuale) ---")
    cell = nn.GRUCell(input_size=8, hidden_size=16)
    seq = torch.randn(5, 8)
    h = torch.zeros(1, 16)
    for t, x_t in enumerate(seq):
        h = cell(x_t.unsqueeze(0), h)
        print(f"passo t={t}: |h|={h.norm():.3f}")

"""
L'update gate z_t in [0,1] mescola stato vecchio e candidato nuovo: h_t = (1 - z_t) * h_{t-1} + z_t * h_candidato
    z~0 -> conserva il passato (memoria stabile); 
    z~1 -> adotta il nuovo stato.
E' il meccanismo che, come la cella della LSTM, offre una via quasi diretta al gradiente e contrasta il suo svanire.
"""
def step_3_update_gate_intuition() -> None:
    
    print("\n--- Passo 3: update gate come interpolatore (1-z)*h + z*cand ---")
    z = torch.sigmoid(torch.randn(5))               # gate d'esempio
    h_prev = torch.ones(5)
    h_cand = torch.full((5,), 2.0)
    h_new = (1 - z) * h_prev + z * h_cand
    print(f"z (update gate): {z.tolist()}")
    print(f"h risultante   : {h_new.tolist()}  (fra 1.0 e 2.0 secondo z)")

"""
nn.GRU: layer completo. Interfaccia come nn.RNN (un solo h_n), non come nn.LSTM (che restituiva anche c_n). 
Questa differenza nella firma va gestita nel modello: lo facciamo in models.py normalizzando l'output.
"""
def step_4_gru_layer() -> None:
    
    print("\n--- Passo 4: nn.GRU (layer completo) ---")
    gru = nn.GRU(input_size=8, hidden_size=16, num_layers=2, batch_first=True)
    x = torch.randn(2, 5, 8)
    output, h_n = gru(x)
    print(f"output: {tuple(output.shape)}  (batch, seq, hidden)")
    print(f"h_n   : {tuple(h_n.shape)}  (num_layers, batch, hidden) -- un solo stato")

"""
parametri RNN vs GRU vs LSTM: verifica dei rapporti ~1:3:4.
"""
def step_5_param_comparison() -> None:
    
    print("\n--- Passo 5: parametri RNN vs GRU vs LSTM (atteso ~1:3:4) ---")
    rnn = nn.RNN(8, 16, batch_first=True)
    gru = nn.GRU(8, 16, batch_first=True)
    lstm = nn.LSTM(8, 16, batch_first=True)
    p = {k: sum(x.numel() for x in m.parameters())
         for k, m in {"RNN": rnn, "GRU": gru, "LSTM": lstm}.items()}
    base = p["RNN"]
    for k, v in p.items():
        print(f"{k:5s}: {v:5d} parametri  ({v / base:.2f}x rispetto a RNN)")


if __name__ == "__main__":
    set_seed(0)
    print("=" * 70)
    print("COSTRUZIONE DIDATTICA DELLA GRU")
    print("=" * 70)
    step_1_single_state()
    step_2_gru_cell()
    step_3_update_gate_intuition()
    step_4_gru_layer()
    step_5_param_comparison()
