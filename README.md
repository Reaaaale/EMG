# Pipeline EMG

Pacchetto portabile per acquisizione, labeling, filtraggio, training e inferenza
di segnali EMG. Tutti i comandi riportati sotto vanno eseguiti dalla cartella
radice `pipeline_emg`.

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

È richiesto Python 3.10 per la compatibilità con Lava-DL.

macOS/Linux:

```bash
cd /percorso/scelto/pipeline_emg
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[training]'
```

Windows PowerShell:

```powershell
cd C:\percorso\scelto\pipeline_emg
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[training]"
```

## Avvio dell'app

```bash
eeg-emg
```

In alternativa:

```bash
python -m eeg_emg
```

Per registrare: collegare la scheda, premere `SEARCH`, selezionare la porta,
premere `Connect`, configurare i registri e usare `START`/`STOP`. Per fare
inferenza, usare `Load model` e selezionare il checkpoint messo in `models/`.

## Training

```bash
jupyter lab notebook/training.ipynb
```

Il notebook ricava automaticamente la radice del progetto dalla propria
posizione e usa `dataset/DATASET_FILTRATO`. I risultati vengono scritti in
`results/<data_ora>/`. La variabile `PIPELINE_EMG_ROOT`, se impostata prima di
avviare Jupyter, permette di indicare esplicitamente un'altra radice:

```bash
export PIPELINE_EMG_ROOT=/percorso/assoluto/pipeline_emg
jupyter lab notebook/training.ipynb
```

Su PowerShell:

```powershell
$env:PIPELINE_EMG_ROOT = "C:\percorso\assoluto\pipeline_emg"
jupyter lab notebook/training.ipynb
```

## Labeling

```bash
python scripts/label_csv.py /percorso/al/file.csv --fs 1000 --label 1 --background-label 0
```

Convenzione: `0=rest`, `1=sasso`, `2=forbici`, `3=carta`.

## Filtraggio

FIR su una cartella:

```bash
python scripts/batch_fir_filter.py /percorso/dati_raw --output-dir /percorso/dati_filtrati --fs 1000
```

Pacchetto comparativo completo:

```bash
python scripts/batch_filter_package.py /percorso/dati_raw --output-dir /percorso/output --fs 1000
```

Per conoscere tutte le opzioni disponibili usare `python scripts/NOME.py --help`.
Dettagli ulteriori sono in `docs/ISTRUZIONI.md`.
