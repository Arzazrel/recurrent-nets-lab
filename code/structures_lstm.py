# -*- coding: utf-8 -*-
"""
structures_lstm.py

Costruzione DIDATTICA della LSTM (Long Short-Term Memory), passo per passo.
La LSTM risolve il problema del gradiente che svanisce della RNN semplice introducendo:
- uno STATO DI CELLA c_t (la "memoria a lungo termine"), che scorre quasi inalterato lungo il tempo;
- tre GATE (cancelli) che regolano il flusso di informazione:
    * forget gate f_t : quanto della memoria passata conservare
    * input  gate i_t : quanto della nuova informazione scrivere
    * output gate o_t : quanto della memoria esporre nello stato nascosto
Da qui il fattore ~4x di parametri rispetto alla RNN: quattro trasformazioni (tre gate + candidato) invece di una.
Eseguendo il file si vedono forme e comportamento dei due stati (h e c).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils import set_seed

"""
i DUE stati della LSTM: nascosto (h) e di cella (c).
A differenza della RNN (un solo h), la LSTM propaga una coppia (h, c).
- c e' la memoria protetta dai gate; 
- h e' cio' che la rete "mostra" all'esterno a ogni passo. 
Entrambi hanno dimensione hidden_size.
"""
def step_1_two_states() -> None:
    
    print("\n--- Passo 1: stato nascosto h e stato di cella c ---")
    hidden_size = 16
    h0 = torch.zeros(1, 1, hidden_size)
    c0 = torch.zeros(1, 1, hidden_size)
    print(f"h0: {tuple(h0.shape)}  (stato nascosto, 'cio' che si mostra')")
    print(f"c0: {tuple(c0.shape)}  (stato di cella, 'la memoria protetta')")

"""
nn.LSTMCell: un passo, con la coppia di stati che evolve.
Si itera per vedere che a ogni passo entrambi gli stati vengono aggiornati. Il confronto |c| vs |h| mostra come la cella accumuli memoria.
"""
def step_2_lstm_cell() -> None:
    
    print("\n--- Passo 2: nn.LSTMCell (un passo alla volta) ---")
    cell = nn.LSTMCell(input_size=8, hidden_size=16)
    seq = torch.randn(5, 8)
    h = torch.zeros(1, 16)
    c = torch.zeros(1, 16)
    for t, x_t in enumerate(seq):
        h, c = cell(x_t.unsqueeze(0), (h, c))
        print(f"passo t={t}: |h|={h.norm():.3f}  |c|={c.norm():.3f}")

"""
intuizione dei gate: cosa calcola internamente la cella.
nn.LSTM non espone i gate singolarmente, ma li ricostruiamo a scopo illustrativo. 
I gate sono sigmoidi in [0,1]: 0 = "chiudi" (blocca il flusso), 1 = "apri" (lascia passare). E' il meccanismo che permette alla rete di
DECIDERE cosa ricordare, invece di sovrascrivere tutto come la RNN.
"""
def step_3_gates_intuition() -> None:
    
    print("\n--- Passo 3: i gate come 'rubinetti' in [0,1] ---")
    torch.manual_seed(0)
    x_t = torch.randn(8)
    h_prev = torch.randn(16)
    W = nn.Linear(8 + 16, 16)                        # trasformazione generica
    z = torch.cat([x_t, h_prev])
    forget_gate = torch.sigmoid(W(z))
    print(f"esempio di forget gate (primi 5): {forget_gate[:5].tolist()}")
    print("valori ~1 conservano la memoria, valori ~0 la cancellano")

"""
nn.LSTM: il layer completo, interfaccia gemella di nn.RNN.
Stessa firma di nn.RNN ma restituisce la coppia (h_n, c_n).
"""
def step_4_lstm_layer() -> None:
   
    print("\n--- Passo 4: nn.LSTM (layer completo) ---")
    lstm = nn.LSTM(input_size=8, hidden_size=16, num_layers=2, batch_first=True)
    x = torch.randn(2, 5, 8)
    output, (h_n, c_n) = lstm(x)
    print(f"output: {tuple(output.shape)}  (batch, seq, hidden)")
    print(f"h_n   : {tuple(h_n.shape)}  (num_layers, batch, hidden)")
    print(f"c_n   : {tuple(c_n.shape)}  (memoria finale di cella)")

"""
verifica empirica del fattore ~4x di parametri.
"""
def step_5_param_comparison() -> None:
    
    print("\n--- Passo 5: parametri RNN vs LSTM (atteso ~4x) ---")
    rnn = nn.RNN(8, 16, batch_first=True)
    lstm = nn.LSTM(8, 16, batch_first=True)
    p_rnn = sum(p.numel() for p in rnn.parameters())
    p_lstm = sum(p.numel() for p in lstm.parameters())
    print(f"RNN : {p_rnn} parametri")
    print(f"LSTM: {p_lstm} parametri   (rapporto {p_lstm / p_rnn:.2f}x)")


if __name__ == "__main__":
    set_seed(0)
    print("=" * 70)
    print("COSTRUZIONE DIDATTICA DELLA LSTM")
    print("=" * 70)
    step_1_two_states()
    step_2_lstm_cell()
    step_3_gates_intuition()
    step_4_lstm_layer()
    step_5_param_comparison()
