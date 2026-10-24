# -*- coding: utf-8 -*-
"""
generate.py

Inferenza interattiva: dato un seme (una parola/carattere o un numero), il modello addestrato genera la continuazione, con statistiche di inferenza.

Due modalita', coerenti con i due task:
- TESTO (classificazione): generazione testo. Si parte da un input, si predice il token successivo, lo si ri-accoda e si ripete. 
    Si puo'regolare la "temperatura" del campionamento:
    - temperatura -> 0  : scelta quasi deterministica (sempre il piu' probabile)
    - temperatura = 1   : campionamento dalla distribuzione cosi' com'e'
    - temperatura > 1   : piu' casualita'/creativita'
- NUMERICO (regressione): estensione della serie. Dato un contesto di seq_len valori, si predice il successivo, lo si accoda e si prosegue.

Statistiche riportate: latenza media per passo (ms) e throughput (token/s), misurate con sincronizzazione CUDA corretta.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from models import ModelConfig, build_model
from utils import get_device, timer


# --------------------------------- Salvataggio / caricamento del modello ---------------------------------
# Salva pesi + configurazione: entrambi servono per ricostruire il modello.
def save_checkpoint(model, cfg: ModelConfig, path: str) -> None:
    torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__}, path)
    print(f"[save] checkpoint -> {path}")

# Ricostruisce il modello dalla configurazione salvata e carica i pesi.
def load_checkpoint(path: str, device):
    ckpt = torch.load(path, map_location=device)
    cfg = ModelConfig(**ckpt["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


# --------------------------------- generazione ---------------------------------

"""
Generazione testuale
Genera 'n_tokens' caratteri a partire da 'seed_text'. Restituisce testo e statistiche.
"""
@torch.no_grad()
def generate_text(model, cfg: ModelConfig, vocab_path: str, seed_text: str, n_tokens: int, seq_len: int, temperature: float, device) -> dict:
    
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    stoi = vocab["stoi"]
    itos = {int(i): c for c, i in stoi.items()}

    # codifica del seme; si usano solo caratteri noti al vocabolario
    context = [stoi[c] for c in seed_text if c in stoi]
    if not context:
        raise ValueError("il seme non contiene caratteri presenti nel vocabolario")

    generated, latencies = [], []
    for _ in range(n_tokens):
        # si usa come contesto al piu' gli ultimi seq_len token
        window = torch.tensor([context[-seq_len:]], dtype=torch.long, device=device)
        with timer(sync_cuda=True) as t:
            logits = model(window)                 # (1, |vocab|)
        latencies.append(t["elapsed"])

        # campionamento con temperatura
        probs = F.softmax(logits.squeeze(0) / max(temperature, 1e-6), dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        context.append(next_id)
        generated.append(next_id)

    text = "".join(itos[i] for i in generated)
    mean_ms = float(np.mean(latencies) * 1000.0)
    return {
        "seed": seed_text,
        "generated": text,
        "mean_latency_ms": round(mean_ms, 3),
        "throughput_tok_s": round(1000.0 / mean_ms, 1) if mean_ms > 0 else float("inf"),
        "n_tokens": n_tokens,
    }



"""
Continuazione della sequenza numerica di 'n_steps' valori. 
- 'seed_values' e' il contesto iniziale (in spazio originale); 
- mu/sigma sono media/dev.std usate in training per de-normalizzare le predizioni.
"""
@torch.no_grad()
def generate_numeric(model, cfg: ModelConfig, seed_values: list[float],n_steps: int, seq_len: int, device,mu: float = 0.0, sigma: float = 1.0) -> dict:
    
    # porta il seme nello spazio normalizzato in cui il modello opera
    context = [(v - mu) / sigma for v in seed_values]

    preds, latencies = [], []
    for _ in range(n_steps):
        window = torch.tensor([context[-seq_len:]], dtype=torch.float32,
                              device=device).unsqueeze(-1)  # (1, L, 1)
        with timer(sync_cuda=True) as t:
            out = model(window)                    # (1, 1)
        latencies.append(t["elapsed"])
        next_norm = float(out.squeeze().item())
        context.append(next_norm)
        preds.append(next_norm * sigma + mu)       # de-normalizzazione

    mean_ms = float(np.mean(latencies) * 1000.0)
    return {
        "seed": seed_values,
        "predicted": [round(p, 4) for p in preds],
        "mean_latency_ms": round(mean_ms, 3),
        "throughput_step_s": round(1000.0 / mean_ms, 1) if mean_ms > 0 else float("inf"),
        "n_steps": n_steps,
    }

# --------------------------------- main ---------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Genera con un modello ricorrente addestrato.")
    p.add_argument("--checkpoint", required=True, help="file .pt salvato con save_checkpoint")
    p.add_argument("--seq-len", type=int, default=20)
    sub = p.add_subparsers(dest="mode", required=True)

    pt = sub.add_parser("text", help="generazione testuale")
    pt.add_argument("--vocab", required=True, help="file .vocab.json del corpus")
    pt.add_argument("--seed", required=True, help="testo iniziale")
    pt.add_argument("--n", type=int, default=200, help="caratteri da generare")
    pt.add_argument("--temperature", type=float, default=0.8)

    pn = sub.add_parser("numeric", help="estensione di serie numerica")
    pn.add_argument("--seed", required=True, help="valori iniziali separati da virgola")
    pn.add_argument("--n", type=int, default=10, help="passi da predire")
    pn.add_argument("--mu", type=float, default=0.0)
    pn.add_argument("--sigma", type=float, default=1.0)

    args = p.parse_args()
    device = get_device()
    model, cfg = load_checkpoint(args.checkpoint, device)

    if args.mode == "text":
        res = generate_text(model, cfg, args.vocab, args.seed, args.n,
                            args.seq_len, args.temperature, device)
        print(f"\n[seme] {res['seed']!r}")
        print(f"[generato]\n{res['seed']}{res['generated']}")
        print(f"\n[stats] latenza media {res['mean_latency_ms']} ms/token | "
              f"{res['throughput_tok_s']} token/s")
    else:
        seed_values = [float(x) for x in args.seed.split(",")]
        res = generate_numeric(model, cfg, seed_values, args.n, args.seq_len,
                               device, mu=args.mu, sigma=args.sigma)
        print(f"\n[seme] {res['seed']}")
        print(f"[predetto] {res['predicted']}")
        print(f"\n[stats] latenza media {res['mean_latency_ms']} ms/passo | "
              f"{res['throughput_step_s']} passi/s")


if __name__ == "__main__":
    main()
