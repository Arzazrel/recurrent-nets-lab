# -*- coding: utf-8 -*-
"""
preprocess_text.py

Pipeline di preprocessing del testo: dal testo grezzo agli id dei token.
Tokenizer char-level ce ha vocabolario di poche decine di simboli, l'embedding e' trascurabile e i parametri si concentrano nella cella, rendendo il confronto interpretabile. 
tiktoken resta disponibile come opzione (--tokenizer tiktoken) per chi vuole vedere l'effetto del sotto-parola.

La pipeline, per ogni passaggio intermedio:
- stampa la dimensione del dato (n. caratteri, n. token, n. token-id, |vocab|)
- stampa una porzione iniziale per mostrare la trasformazione in corso
- salva su file il risultato intermedio, cosi' da poterlo ispezionare e ricaricare (il file finale di token-id e' pronto per il windowing).

"""

from __future__ import annotations

import argparse
import json
import os
from typing import Protocol
import numpy as np

from utils import project_paths

# Interfaccia comune del tokenizer contenente le funzioni minime che deve offrire un tokenizer
class Tokenizer(Protocol):
    vocab_size: int
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...

"""
Tokenizer a livello di carattere: ogni carattere distinto e' un token.
Il vocabolario e' costruito dal corpus stesso (chiuso sul testo dato), ordinato per determinismo. E' il tokenizer piu' semplice possibile.
"""
class CharTokenizer:
    
    def __init__(self, text: str):
        chars = sorted(set(text))                           # vocabolario ordinato -> riproducibile
        self.stoi = {c: i for i, c in enumerate(chars)}     # string -> int
        self.itos = {i: c for c, i in self.stoi.items()}    # int -> string
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)



# Wrapper su tiktoken (encoding 'gpt2'). Sotto-parola BPE, vocabolario ~50k.
class TiktokenTokenizer:
    
    def __init__(self, encoding_name: str = "gpt2"):
        import tiktoken                                     # import locale: dipendenza opzionale
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)

# seleziona il tokenizer in base alla scelta dell'utente.
def build_tokenizer(kind: str, text: str) -> Tokenizer:
    # sceglie il tokenizer in base alla preferenza dell'utente
    if kind == "char":
        return CharTokenizer(text)
    if kind == "tiktoken":
        return TiktokenTokenizer()
    raise ValueError(f"tokenizer sconosciuto: {kind}")


# ---------------------------------- Pipeline ----------------------------------
# Restituisce una porzione iniziale leggibile di una stringa o lista.
def preview(s, n: int = 80) -> str:
    
    if isinstance(s, str):
        snippet = s[:n].replace("\n", "\\n")
        return f"'{snippet}...'"
    return f"{list(s[:n])}..."

"""
Esegue l'intera pipeline su un file di testo, stampando e salvando i passaggi.
Restituisce un dizionario di metadati (percorsi dei file prodotti, |vocab|,conteggi) utile ai passi successivi e ai test automatici.
"""
def run_pipeline(text_path: str, tokenizer_kind: str, out_dir: str) -> dict:
    
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(text_path))[0]
    print(f"\n{'='*70}\n[preprocess] corpus: {name}  |  tokenizer: {tokenizer_kind}\n{'='*70}")

    # --- Passo 0: testo grezzo ------------------------------------------------
    with open(text_path, "r", encoding="utf-8") as f:
        raw = f.read()
    print(f"[0] testo grezzo        : {len(raw):>8d} caratteri")
    print(f"    campione            : {preview(raw)}")

    # --- Passo 1: costruzione del tokenizer / vocabolario --------------------
    tok = build_tokenizer(tokenizer_kind, raw)
    print(f"[1] vocabolario         : {tok.vocab_size:>8d} token distinti")

    # --- Passo 2: tokenizzazione -> lista di token-id ------------------------
    ids = tok.encode(raw)
    print(f"[2] token-id            : {len(ids):>8d} id")
    print(f"    campione            : {preview(ids)}")

    # --- Passo 3: verifica di invertibilita' (encode->decode) ----------------
    # Controllo di sanita': il decode degli id deve restituire il testo.
    # Per il char-level e' esatto; per tiktoken e' esatto a meno di edge-case.
    roundtrip_ok = tok.decode(ids[:200]) == raw[:200] if tokenizer_kind == "char" else True
    print(f"[3] round-trip (primi 200): {'OK' if roundtrip_ok else 'ATTENZIONE: mismatch'}")

    # --- Salvataggi intermedi -------------------------------------------------
    # 3a) vocabolario (per decodificare le predizioni in fase di inferenza)
    vocab_path = os.path.join(out_dir, f"{name}.{tokenizer_kind}.vocab.json")
    if tokenizer_kind == "char":
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump({"stoi": tok.stoi, "vocab_size": tok.vocab_size},
                      f, ensure_ascii=False, indent=0)
    else:
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump({"encoding": "gpt2", "vocab_size": tok.vocab_size}, f)
    print(f"[save] vocabolario      -> {os.path.basename(vocab_path)}")

    # 3b) token-id finali in .npy (compatto, veloce da ricaricare per il windowing)
    ids_path = os.path.join(out_dir, f"{name}.{tokenizer_kind}.ids.npy")
    np.save(ids_path, np.asarray(ids, dtype=np.int64))
    print(f"[save] token-id         -> {os.path.basename(ids_path)}")

    # 3c) una versione .txt dei primi id, solo per ispezione umana
    ids_txt = os.path.join(out_dir, f"{name}.{tokenizer_kind}.ids.preview.txt")
    with open(ids_txt, "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, ids[:500])))
    print(f"[save] anteprima id     -> {os.path.basename(ids_txt)}")

    return {
        "name": name, "tokenizer": tokenizer_kind, "vocab_size": tok.vocab_size,
        "n_chars": len(raw), "n_ids": len(ids),
        "ids_path": ids_path, "vocab_path": vocab_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocessing del testo in token-id.")
    parser.add_argument("--input", type=str, default=None,
                        help="percorso del file di testo; se omesso, processa i corpora di esempio")
    parser.add_argument("--tokenizer", choices=["char", "tiktoken"], default="char",
                        help="char (default, didattico) oppure tiktoken (sotto-parola)")
    args = parser.parse_args()

    paths = project_paths()
    text_dir = os.path.join(paths["dataset"], "text")
    out_dir = os.path.join(paths["dataset"], "text_processed")

    if args.input:
        run_pipeline(args.input, args.tokenizer, out_dir)
    else:
        # processa entrambi i corpora di esempio (corto e lungo)
        for fname in ("corpus_short.txt", "corpus_long.txt"):
            fpath = os.path.join(text_dir, fname)
            if os.path.exists(fpath):
                run_pipeline(fpath, args.tokenizer, out_dir)
            else:
                print(f"[warn] non trovato: {fpath} (esegui prima make_text_datasets.py)")


if __name__ == "__main__":
    main()
