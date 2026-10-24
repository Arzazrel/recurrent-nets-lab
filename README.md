# recurrent-nets-lab

**Laboratorio pratico per imparare, implementare e confrontare le reti neurali ricorrenti — RNN semplice, LSTM e GRU — con PyTorch.**

Un ambiente autocontenuto in cui costruire le tre architetture dai mattoncini di base, addestrarle su task controllati e metterle a confronto in modo equo su:
**qualità** (accuracy/perplexity), **velocità di apprendimento** (epoche per raggiungere una soglia) e **costo** (numero di parametri e tempo di inferenza).

## 1. Scopo del laboratorio

Prima dell'avvento dei transformer, le reti ricorrenti erano lo strumento standard per i dati sequenziali. 
Capirle a fondo, perché LSTM e GRU esistono e quando conviene l'una o l'altra, resta un prerequisito per comprendere l'evoluzione verso i modelli moderni.
Questo repository nasce con tre obiettivi:

1. **Imparare costruendo.** Ogni architettura è ricostruita passo per passo, dal singolo `RNNCell` fino alla rete completa, con verifiche intermedie.
2. **Sperimentare in modo rigoroso.** Le tre reti condividono dataset, split, iperparametri e seme casuale, così che ogni differenza osservata sia attribuibile all'architettura della cella e non al contorno.
3. **Quantificare i compromessi.** Non basta sapere che «la LSTM ricorda meglio»: il laboratorio *misura* di quanto, a quale costo in parametri e con quale penalità sui tempi.

## 2. Cosa si è fatto

- Implementate le tre celle ricorrenti dietro **un'unica classe parametrizzabile** (`models.py`): tipo di cella, larghezza dello stato, profondità, dropout, bidirezionalità e lunghezza del contesto sono tutti configurabili, così la stessa rete si adatta a scenari diversi.
- Realizzati **due task complementari**:
  - **predizione del token successivo** su testo (il classico task pre-transformer), con corpus corto e lungo;
  - **regressione su sequenze numeriche** generate sinteticamente, dove la regola è nota e la correttezza è verificabile analiticamente.
- Costruita una **pipeline di preprocessing del testo** trasparente, che a ogni passaggio stampa dimensioni e anteprime e salva i risultati intermedi (vocabolario, token-id) pronti per il windowing.
- Scritto un **motore di confronto** (`benchmark.py`) che addestra le tre reti nelle stesse condizioni e produce tabelle e grafici su qualità, epoche-a-soglia, parametri e latenza di inferenza.
- Aggiunta l'**inferenza interattiva** (`generate.py`): dato un seme, il modello genera la continuazione con statistiche di latenza e throughput.

## 3. Cosa si impara / si osserva

Eseguendo gli esperimenti si rendono misurabili alcuni fatti strutturali:

- **Costo in parametri.** A parità di capacità nominale, i parametri stanno in rapporto circa **1 : 3 : 4** fra RNN, GRU e LSTM, perché la cella contiene rispettivamente 1, 3 e 4 trasformazioni interne (i «gate»). 
  Lo si verifica empiricamente nei file di strutture.
- **Memoria a lungo termine.** Sui task a dipendenza breve (sequenze a passo costante) tutte e tre convergono rapidamente. 
  È sul **copy task a ritardo** e su Fibonacci che la RNN semplice si stacca in negativo: il suo gradiente svanisce sulle dipendenze lunghe, mentre i gate di LSTM/GRU preservano il
  segnale. È la dimostrazione pratica del motivo per cui queste varianti sono state inventate.
- **Compromesso GRU.** La GRU tende a ottenere qualità vicina alla LSTM con meno parametri e addestramento più rapido: il benchmark quantifica questo equilibrio caso per caso.

> I numeri esatti dipendono dall'hardware e dagli iperparametri; il valore del
> laboratorio è il **metodo di confronto**, non una singola tabella.

---

## 4. Struttura del repository

```
recurrent-nets-lab/
├── README.md
├── requirements.txt
├── code/                       # tutto il codice
│   ├── utils.py                # device, seed, conteggio parametri, timer CUDA-safe
│   ├── windows.py              # finestre scorrevoli + Dataset PyTorch
│   ├── make_numeric_datasets.py# genera le sequenze numeriche (→ dataset/numeric)
│   ├── make_text_datasets.py   # scarica/genera i corpora di testo (→ dataset/text)
│   ├── preprocess_text.py      # testo → token → token-id (→ dataset/text_processed)
│   ├── structures_rnn.py       # costruzione didattica della RNN, passo per passo
│   ├── structures_lstm.py      # costruzione didattica della LSTM
│   ├── structures_gru.py       # costruzione didattica della GRU
│   ├── models.py               # la rete completa parametrizzabile (le 3 celle)
│   ├── train.py                # motore di addestramento + early stopping
│   ├── benchmark.py            # confronto fra le 3 reti + grafici/CSV
│   └── generate.py             # inferenza interattiva (testo e numerica)
├── dataset/                    # dataset iniziali e preprocessati
│   ├── numeric/                # sequenze numeriche in .txt (ispezionabili)
│   ├── text/                   # corpus corto e lungo
│   └── text_processed/         # vocabolario + token-id pronti per il training
└── results/                    # grafici e CSV prodotti dai benchmark
```

---

## 5. Perché queste scelte di progetto

### 5.1 Una sola classe per tre reti
`nn.RNN`, `nn.LSTM` e `nn.GRU` hanno interfacce quasi identiche. Sfruttando questo, `RecurrentModel` seleziona la cella dinamicamente e normalizza la gestione degli stati (la LSTM restituisce anche lo stato di cella). 
Così il confronto misura l'architettura, non differenze accidentali di codice.

### 5.2 Tokenizzazione a livello di carattere come default
Il tokenizer predefinito è **char-level**, non `tiktoken`. 
Con un vocabolario sotto-parola da decine di migliaia di voci, la matrice di embedding (`|vocab| × d`) dominerebbe il conteggio dei parametri, **oscurando proprio ciò che vogliamo confrontare**: la complessità della cella ricorrente. 
Con poche decine di caratteri l'embedding è trascurabile e i parametri si concentrano nella cella, rendendo il confronto interpretabile. `tiktoken` resta disponibile come opzione (`--tokenizer tiktoken`) per chi voglia osservare l'effetto del sotto-parola.

### 5.3 «Finestra», non «attenzione»
RNN/LSTM/GRU **non hanno un meccanismo di attenzione** quello è proprio dei transformer.
Il parametro configurabile è la **lunghezza del contesto** (`seq_len`) con cui si costruiscono le finestre scorrevoli, assieme al passo (`stride`). 
È questo a determinare quanto passato la rete vede prima di predire.

### 5.4 Metriche: accuracy *oppure* perplexity, non a caso
La scelta della metrica è dettata dalla natura del task..

- **Sequenze numeriche (regressione) → accuracy a soglia.** L'obiettivo è
  deterministico: esiste una risposta corretta e unica. Si definisce «corretta»
  una predizione entro una tolleranza dal valore vero. 
  La soglia è raggiungibile e informativa.

- **Testo (next-token) → perplexity.** Sul linguaggio naturale predire la parola
  *esatta* plafona molto in basso: anche modelli forti raramente superano il
  30–40% di accuracy, perché a ogni passo esistono molte continuazioni
  plausibili. La metrica corretta è la **cross-entropy** e la sua esponenziale, la **perplexity**:

  ```
  perplexity = exp(cross_entropy)
  ```

  Interpretazione intuitiva: la perplexity è il **numero medio di alternative equiprobabili** fra cui il modello è indeciso a ogni passo. 
  `perplexity = 1` è la predizione perfetta; `perplexity = |vocab|` equivale a tirare a caso.
  **Più bassa è, meglio è.** È la metrica standard dei modelli linguistici proprio perché valuta la *qualità dell'intera distribuzione predetta*,
  quanto il modello concentra probabilità sulle continuazioni corrette e non solo se indovina il colpo singolo. 
  Due modelli con la stessa accuracy possono avere perplexity molto diverse: quello con perplexity minore ha imparato meglio la struttura del linguaggio.

## 6. Installazione

Il laboratorio è pensato per una GPU NVIDIA sotto **WSL2** (Linux), ambiente più stabile del Windows nativo per le build CUDA recenti.

### Requisito critico per RTX serie 50 (Blackwell, es. RTX 5070 Ti)
La RTX 5070 Ti ha compute capability **sm_120** e richiede una build di PyTorch con **kernel CUDA 12.8** (wheel `cu128`).
Una build precedente rileva la scheda ma fallisce al primo kernel con l'errore *«sm_120 is not compatible»*.

```bash
# 1) PyTorch con CUDA 12.8 (obbligatorio per la RTX 5070 Ti)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2) il resto delle dipendenze
pip install -r requirements.txt

# 3) verifica: 'sm_120' deve comparire nell'elenco
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"
```

Su schede più vecchie es RTX 3060 (Ampere, sm_86) va bene anche una build `cu121`/`cu124`: lo stesso codice gira su entrambe le macchine senza modifiche.

## 7. Uso rapido

### 7.1 Esplorare le architetture (didattico)
```bash
cd code
python structures_rnn.py      # costruisce la RNN passo per passo, con print
python structures_lstm.py     # idem per la LSTM (gate + stato di cella)
python structures_gru.py      # idem per la GRU (reset/update gate)
```

### 7.2 Generare i dataset
```bash
python make_numeric_datasets.py --n 2000      # sequenze numeriche → dataset/numeric
python make_text_datasets.py                  # corpus corto e lungo → dataset/text
python preprocess_text.py --tokenizer char    # token-id → dataset/text_processed
```

### 7.3 Addestrare una singola rete
```bash
# regressione su Fibonacci
python train.py --task regression --cell lstm \
    --data ../dataset/numeric/num_fibonacci.txt \
    --seq-len 10 --acc-threshold 0.98 --save-to ../results/lstm_fib.pt

# testo (next-char) sul corpus corto
python train.py --task classification --cell gru \
    --data ../dataset/text_processed/corpus_short.char.ids.npy \
    --seq-len 50 --hidden 128 --save-to ../results/gru_short.pt
```

### 7.4 Confrontare le tre reti
```bash
# confronto su un task numerico (epoche-a-soglia 98%)
python benchmark.py --task regression \
    --data ../dataset/numeric/num_copy_delay10.txt \
    --seq-len 20 --acc-threshold 0.98 --epochs 60

# confronto su testo (perplexity finale)
python benchmark.py --task classification \
    --data ../dataset/text_processed/corpus_long.char.ids.npy \
    --seq-len 64 --hidden 256 --layers 2 --epochs 30
```
I grafici e il CSV riassuntivo finiscono in `results/`.

### 7.5 Generare con un modello addestrato
```bash
# continuazione testuale
python generate.py --checkpoint ../results/gru_short.pt --seq-len 50 \
    text --vocab ../dataset/text_processed/corpus_short.char.vocab.json \
    --seed "the old " --n 200 --temperature 0.8

# estensione di una serie numerica
python generate.py --checkpoint ../results/lstm_fib.pt --seq-len 10 \
    numeric --seed "0,1,1,2,3,5,8,13,21,34" --n 10
```

## 8. Dataset inclusi

**Sequenze numeriche** (generate, salvate in `.txt` leggibili):

| file | regola | memoria richiesta |
|---|---|---|
| `num_increasing_1` | +1 a passo costante | ordine 1 (banale) |
| `num_increasing_N` | +N a passo costante | ordine 1 |
| `num_decreasing_1` | −1 a passo costante | ordine 1 |
| `num_fibonacci` | aₜ = aₜ₋₁ + aₜ₋₂ | 2 passi |
| `num_sine` | sinusoide + rumore | periodica |
| `num_copy_delay10` | copia il valore di 10 passi prima | **lunga** |

Il `num_copy_delay10` è il banco di prova chiave: è qui che la differenza fra
RNN semplice e reti gated emerge in modo netto e misurabile.

**Testo**: un corpus **corto** (iterazioni rapide) e uno **lungo** (confronto più
realistico), scaricati da fonti di pubblico dominio con fallback procedurale
offline, così il repository resta liberamente ridistribuibile.

# Autore
Alessandro Diana

Programmato con aiuto di Claude


