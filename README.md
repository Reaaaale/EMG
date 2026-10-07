
## Contenuto

- `notebook/training.ipynb`: notebook di training.
- `dataset/DATASET_FILTRATO`: dataset filtrato e labellato.
- `scripts`: strumenti per labeling, copia delle label e filtraggio.
- `src/eeg_emg`: applicazione desktop per registrazione, replay e inferenza.
- `models`: destinazione consigliata per modello e configurazione.
- `results`: output generati dal notebook.




## Installazione

macOS/Linux:

```bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[training]'
```

## Avvio dell'app

```bash
eeg-emg
```

In alternativa:

```bash
python -m eeg_emg
```


