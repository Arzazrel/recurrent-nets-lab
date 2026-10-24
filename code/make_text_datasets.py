# -*- coding: utf-8 -*-
"""
make_text_datasets.py

Produce i due corpora di testo del laboratorio: uno corto e uno lungo.

Requisiti del corpus (dall'impostazione del progetto):
- realistico e sensato (prosa in inglese ben formata)
- privo di vincoli di copyright: usiamo testo di pubblico dominio (Project Gutenberg) oppure testo generato proceduralmente qui, cosi' il repository e' ridistribuibile liberamente come portfolio
- abbastanza piccolo da addestrare in tempi fattibili in locale su GPU

Strategia: si tenta prima di scaricare un classico di pubblico dominio da Project Gutenberg (inglese, prosa reale). 
Se la rete non e' disponibile si ripiega su un corpus generato proceduralmente (prosa inglese ripetuta con variazioni): meno realistico ma sufficiente a far funzionare la pipeline offline.
In entrambi i casi il testo viene troncato a una lunghezza gestibile.

Il corpus "corto" e' pensato per iterazioni rapide (debug, verifica che le reti apprendano); quello "lungo" per un confronto piu' realistico della capacita'.
"""

from __future__ import annotations

import argparse
import os
import urllib.request

from utils import project_paths


# Classici di pubblico dominio su Project Gutenberg (testo semplice, inglese).
# URL "cache" dei .txt. Scelti per prosa pulita e lunghezza adeguata.
GUTENBERG = {
    # ~ 170 KB: adatto come corpus "corto"
    "short": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",   # Sherlock Holmes
    # ~ 1.2 MB: adatto come corpus "lungo" (lo tronchiamo)
    "long": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",    # Moby Dick
}

"""
Tenta il download di un testo. Restituisce None in caso di fallimento (assenza di rete, timeout).
"""
def try_download(url: str, timeout: int = 20) -> str | None:
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "recurrent-nets-lab"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception as e:  # noqa: BLE001  (vogliamo catturare qualsiasi errore di rete)
        print(f"[download] fallito ({e.__class__.__name__}): uso il fallback procedurale")
        return None

"""
Rimuove l'intestazione e la licenza standard di Project Gutenberg, conservando solo il corpo dell'opera.
"""
def strip_gutenberg_boilerplate(text: str) -> str:
    
    start_marker = "*** START OF"
    end_marker = "*** END OF"
    start = text.find(start_marker)
    if start != -1:
        start = text.find("\n", start) + 1
    else:
        start = 0
    end = text.find(end_marker)
    if end == -1:
        end = len(text)
    return text[start:end].strip()

"""
Corpus di riserva generato proceduralmente: prosa inglese plausibile ottenuta concatenando frasi da un piccolo repertorio con lievi variazioni.
Non e' letteratura, ma ha struttura lessicale e sintattica regolare, sufficiente perche' le reti imparino dipendenze a breve raggio.
"""
def procedural_corpus(n_chars: int) -> str:
    
    sentences = [
        "the old lighthouse stood at the edge of the rocky shore.",
        "every evening the keeper climbed the narrow spiral stairs.",
        "a pale light swept across the dark and restless water.",
        "ships in the distance answered with a single steady lamp.",
        "the wind carried the smell of salt and distant rain.",
        "gulls circled above the harbour and called to one another.",
        "in the morning the fishermen returned with heavy nets.",
        "children gathered on the pier to watch the boats arrive.",
        "the market filled with voices and the scent of fresh bread.",
        "by noon the sun had burned away the last of the fog.",
    ]
    out = []
    i = 0
    while sum(len(s) for s in out) < n_chars:
        out.append(sentences[i % len(sentences)])
        i += 1
        if i % len(sentences) == 0:
            out.append("\n")  # paragrafi, per dare struttura di a-capo
    return " ".join(out)[:n_chars]

# Tronca a 'limit' caratteri e salva il corpus con newline normalizzati.
def write_corpus(path: str, text: str, limit: int) -> None:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text[:limit]
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[save] {os.path.basename(path):20s} ({len(text)} caratteri)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera i corpora di testo (corto e lungo).")
    parser.add_argument("--short-chars", type=int, default=40_000,
                        help="lunghezza del corpus corto (default: 40k char)")
    parser.add_argument("--long-chars", type=int, default=400_000,
                        help="lunghezza del corpus lungo (default: 400k char)")
    parser.add_argument("--offline", action="store_true",
                        help="salta il download e usa solo il fallback procedurale")
    args = parser.parse_args()

    paths = project_paths()
    out_dir = os.path.join(paths["dataset"], "text")
    os.makedirs(out_dir, exist_ok=True)

    for tag, limit in (("short", args.short_chars), ("long", args.long_chars)):
        text = None
        if not args.offline:
            raw = try_download(GUTENBERG[tag])
            if raw is not None:
                text = strip_gutenberg_boilerplate(raw)
        if text is None:
            text = procedural_corpus(limit)
        write_corpus(os.path.join(out_dir, f"corpus_{tag}.txt"), text, limit)

    print("[make] corpora di testo pronti in", out_dir)


if __name__ == "__main__":
    main()
