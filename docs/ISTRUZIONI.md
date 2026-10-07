# Istruzioni operative

## Percorsi

Il progetto non contiene percorsi legati al computer sul quale è stato creato.
Il notebook determina `PROJECT_ROOT` automaticamente; in alternativa accetta
la variabile d'ambiente `PIPELINE_EMG_ROOT`. Gli script ricevono file e cartelle
come argomenti da terminale, quindi possono usare sia percorsi assoluti sia
percorsi relativi alla radice del progetto.

## Dataset

Il dataset incluso si trova in `dataset/DATASET_FILTRATO`. Ogni CSV usato dal
training deve contenere `CH1`-`CH8` e `label`. Il notebook ignora i file che
contengono `_aug_` nel nome.

## Copia delle label

Quando file raw e filtrato hanno lo stesso numero e ordine di campioni:

```bash
python scripts/copy_labels.py \
  --source-csv /percorso/raw_labeled.csv \
  --target-csv /percorso/filtered.csv \
  --output /percorso/filtered_labeled.csv
```

Aggiungere `--overwrite` solo quando si vuole sovrascrivere l'output.

## Butterworth fixed point

```bash
python scripts/filter_butterworth_fixed.py \
  /percorso/input.csv \
  --output /percorso/output.csv \
  --fs 1000
```

Consultare `--help` perché frequenze, canali e parametri possono essere
personalizzati.

## Inferenza

Creare, per esempio, `models/mio_modello/` e inserirvi:

```text
models/mio_modello/model_state_dict.pt
models/mio_modello/config.json
```

Avviare `eeg-emg`, premere `Load model` e selezionare il file `.pt`. La
configurazione accanto al checkpoint descrive finestra, encoding e filtraggio
usati durante il training e non deve essere separata dal modello.

Il pannello CSV replay consente di provare l'inferenza su un CSV senza acquisire
dalla scheda. Impostare la frequenza di campionamento corretta prima del replay.

## Aggiornamento delle dipendenze

Le dipendenze sono definite in `pyproject.toml`. `requirements.txt` installa il
progetto base; per notebook e training usare sempre:

```bash
python -m pip install -e '.[training]'
```
