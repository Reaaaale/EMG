
## Contenuto

- `notebook/training.ipynb`: notebook di training.
- `dataset/DATASET_FILTRATO`: dataset filtrato e labellato.
- `scripts`: strumenti per labeling, copia delle label e filtraggio.
- `src/eeg_emg`: applicazione desktop per registrazione, replay e inferenza.
- `models`: destinazione consigliata per modello e configurazione.
- `results`: output generati dal notebook.
- `docs`: istruzioni dettagliate e documentazione tecnica.

I modelli non sono inclusi. Per l'inferenza copiare nella stessa sottocartella
di `models/` il checkpoint (`model_state_dict.pt` oppure `model_best.pt`) e il
relativo `config.json` prodotto dal notebook.

## Installazione

macOS/Linux:

```bash
cd /percorso/scelto/pipeline_emg
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


