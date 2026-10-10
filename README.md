# Crop Spatial Classification

Progetto per la classificazione multiclasse e la mappatura territoriale delle colture agricole e della copertura del suolo (Land Cover) nella pianura della Capitanata (provincia di Foggia, Puglia), realizzato per il corso universitario di Machine Learning e Data Mining.

---

## Panoramica del Progetto

L'obiettivo principale è identificare automaticamente le tipologie di colture presenti sul territorio combinando osservazioni satellitari multitemporali e tecniche di machine learning supervisionato.

Poiché molte colture presentano firme spettrali simili in singoli momenti dell'anno, il progetto sfrutta l'evoluzione temporale (fenologia) lungo un arco di 12 mesi per catturare le diverse fasi vegetative (semina, crescita, maturazione, senescenza e raccolto).

I componenti cardine del progetto includono:
* **Dati Satellitari**: Immagini multispettrali Sentinel-2 L2A (Bottom-Of-Atmosphere) scaricate tramite catalogo STAC, selezionando 6 bande spettrali agronomiche (B02, B03, B04, B08, B11, B12) a risoluzione 10 metri.
* **Ground Truth**: Dati ufficiali ad alta risoluzione del Copernicus Land Monitoring Service (CLMS) - Crop Types (CTY), utilizzati per etichettare le parcelle agricole.
* **Feature Engineering**: Calcolo vettoriale di 51 feature per ogni punto/pixel, comprendenti medie spettrali, serie mensili di NDVI e NDWI, statistiche fenologiche annuali, differenziali stagionali e indicatori di biomassa.
* **Modellazione**: Addestramento e ottimizzazione di un classificatore Random Forest con analisi di feature importance e strategie di feature pruning.
* **Inferenza Spaziale**: Predizione sull'intera estensione territoriale della Capitanata per la generazione di mappe tematiche a confronto con il ground truth reale.

---

## Flusso di Lavoro e Descrizione dei Notebook

Il flusso di lavoro è articolato in 5 notebook sequenziali nella cartella `notebooks/`.

### 01 - Estrazione del ground truth e campionamento delle colture
*File: `notebooks/01_ground_truth_extraction.ipynb`*

* **Obiettivo**: Elaborare i dati di riferimento al suolo (Copernicus CLMS), ricavare la legenda ufficiale e costruire un campione bilanciato di punti geografici all'interno dei confini della Capitanata.
* **Operazioni svolte**:
  1. Ricerca dei raster annuali del suolo (formato EPSG:3035) per gli anni target (2021, 2022, 2023).
  2. Riproiezione in coordinate geografiche EPSG:4326 mediante interpolazione Nearest Neighbor per preservare i codici numerici delle classi.
  3. Parsing automatico dei metadati XML (`.aux.xml`) per generare il file di corrispondenza codice-nome coltura (`legend.json`).
  4. Campionamento casuale e bilanciato (fino a 125 punti per classe per file raster), scartando le classi non agricole (aree urbane, boschi, mare) e i punti fuori dal bounding box della Capitanata.
  5. Esportazione delle coordinate e delle classi estratte in `data/interim/points.json` (22.500 punti totali).

---

### 02 - Download e riepilogo dei dati satellitari per l'area di studio
*File: `notebooks/02_download_sentinel2_data.ipynb`*

* **Obiettivo**: Scaricare in modo programmatico le immagini satellitari Sentinel-2 per tutti i 12 mesi degli anni analizzati e costruire mosaici regionali completi per la Capitanata.
* **Operazioni svolte**:
  1. Connessione alle API STAC di Element84 Earth Search per interrogare la collezione Sentinel-2 L2A.
  2. Selezione della scena mensile più limpida per ciascuna tile MGRS intersecante l'area di studio, escludendo passaggi orbitali parziali tramite soglie su nuvolosità e pixel NoData.
  3. Download e ritaglio delle 6 bande spettrali agronomiche: Blu (B02), Verde (B03), Rosso (B04), Vicino Infrarosso (B08), SWIR1 (B11) e SWIR2 (B12).
  4. Riproiezione e allineamento geometrico delle bande a 20m sulla griglia a 10m.
  5. Fusione delle tile e salvataggio dei raster mensili a 6 bande in formato GeoTIFF compresso (36 file totali per gli anni 2021-2023).

---

### 03 - Creazione e salvataggio del dataset
*File: `notebooks/03_dataset_creation.ipynb`*

* **Obiettivo**: Estrarre i valori spettrali dai GeoTIFF mensili in corrispondenza dei punti campionati, calcolare le feature fenologiche e costruire il dataset tabulare finale.
* **Operazioni svolte**:
  1. Lettura dei punti di ground truth e campionamento dei pixel corrispondenti su tutti i 36 raster mensili.
  2. Organizzazione dei valori in una struttura tridimensionale (12 mesi, 6 bande, N punti).
  3. Ricostruzione dei valori mancanti o coperti da nubi tramite interpolazione temporale bidirezionale (forward/backward fill).
  4. Calcolo delle 51 feature fenologiche tramite il modulo `src/feature_engineering.py`:
     * Medie annuali delle 6 bande.
     * Serie mensili di NDVI (vigore vegetativo) e NDWI (idratazione).
     * Statistiche annuali: massimo, minimo, escursione, mese di picco, media e variabilità di NDVI.
     * Differenziali chiave tra stagioni (es. variazione luglio-aprile, crescita primaverile, tasso di senescenza, rapporto orzo/grano).
     * Indicatori strutturali e di biomassa (area sotto la curva NDVI, mesi attivi, indici cellulosa/lignina).
  5. Esportazione del dataset tabulare (22.500 righe per 55 colonne) nei formati `dataset.parquet` e `dataset.csv`.

---

### 04 - Addestramento Random Forest e valutazione delle prestazioni
*File: `notebooks/04_fit_random_forest.ipynb`*

* **Obiettivo**: Addestrare un classificatore Random Forest, ottimizzarne gli iperparametri, valutarne l'accuratezza e analizzare l'effetto della riduzione delle feature.
* **Operazioni svolte**:
  1. Suddivisione del dataset in Train (70%) e Test (30%) con campionamento stratificato sulle classi.
  2. Ottimizzazione degli iperparametri con `GridSearchCV` su 5 fold (valutando combinazioni di `n_estimators`, `max_depth` e pesi bilanciati), massimizzando la metrica Macro F1-Score.
  3. Valutazione sul test set indipendente tramite Accuracy, Macro F1, classification report dettagliato per classe e matrice di confusione normalizzata per righe.
  4. Analisi della Feature Importance calcolata dal modello per individuare le variabili con maggior potere discriminante.
  5. Studio di Feature Pruning con `SelectFromModel` confrontando tre soglie (`0.75*mean`, `median`, `mean`) per misurare il bilanciamento tra numero di feature, tempi di training e accuratezza predittiva.

---

### 05 - Visualizzazione territoriale della Capitanata
*File: `notebooks/05_spatial_visualization.ipynb`*

* **Obiettivo**: Eseguire l'inferenza del modello su scala territoriale per generare una cartografia predittiva completa della Capitanata e valutarne la coerenza spaziale.
* **Operazioni svolte**:
  1. Caricamento dello stack completo dei 12 GeoTIFF mensili (1931 x 2000 pixel, pari a circa 3.86 milioni di pixel).
  2. Estrazione vettorializzata delle 51 feature per ogni singolo pixel del territorio.
  3. Predizione delle classi colturali su tutta la matrice territoriale tramite il modello Random Forest addestrato.
  4. Applicazione della maschera agricola del ground truth e mappatura con la legenda colori ufficiale di Copernicus.
  5. Creazione di una dashboard comparativa a 5 riquadri:
     * Immagine satellitare reale in colori naturali (composito RGB estivo).
     * Mappa reale del suolo (Ground Truth Copernicus CLMS).
     * Mappa predetta dal Random Forest.
     * Mappa dei campi corretti (aree con esito conforme).
     * Mappa degli errori con calcolo dell'accuratezza spaziale complessiva e del tasso di errore.

---

## Struttura del Repository

```plaintext
crop-spatial-classification/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Coordinate BBox, setup ambiente e percorsi universali
│   ├── feature_engineering.py    # Logica di estrazione delle 51 feature e gestione raster 4D
│   └── sentinel2_downloader.py   # Modulo per download e mosaicatura via STAC API
├── notebooks/
│   ├── 01_ground_truth_extraction.ipynb
│   ├── 02_download_sentinel2_data.ipynb
│   ├── 03_dataset_creation.ipynb
│   ├── 04_fit_random_forest.ipynb
│   └── 05_spatial_visualization.ipynb
├── data/                         # Cartella dati (locale o collegata da Google Drive)
│   ├── raw/                      # Raster originali CLMS EPSG:3035 e metadati
│   ├── interim/                  # Coordinate campionate (points.json)
│   └── processed/                # GeoTIFF Sentinel-2, dataset (parquet/csv) e modelli
├── requirements.txt              # Dipendenze Python
├── LICENSE                       # Licenza del progetto
└── README.md                     # Documentazione del repository
```

---

## Requisiti e Installazione

### Requisiti di Sistema
* Python 3.10 o versioni successive
* Connessione internet (per il download dei dati satellitari via API STAC)
* Spazio su disco sufficiente per la memorizzazione dei GeoTIFF mensili della Capitanata

### Installazione

Clona il repository e installa le dipendenze richieste:

```bash
git clone https://github.com/SimoRinaldi/crop-spatial-classification.git
cd crop-spatial-classification

# Creazione ambiente virtuale (consigliato)
python3 -m venv .venv
source .venv/bin/activate  # Su Linux/macOS
# .venv\Scripts\activate   # Su Windows

# Installazione librerie
pip install -r requirements.txt
```

---

## Esecuzione dei Notebook

### Inizializzazione dell'Ambiente e Percorsi
I notebook condividono una funzione unificata `setup_environment()` (definita in `src/config.py`) richiamata nella prima cella:

```python
from src.config import setup_environment

DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR = setup_environment()
```

Questa funzione gestisce in modo trasparente l'ambiente di esecuzione:
* **Ambiente Locale**: rileva automaticamente la cartella del progetto e definisce i percorsi standard (`DATA_DIR`, `RAW_DIR`, `INTERIM_DIR`, `PROCESSED_DIR`).
* **Google Colab**: monta automaticamente Google Drive (`/content/drive`), collega lo storage condiviso con i dati del progetto e aggiorna le dipendenze.

Per eseguire l'intero workflow, aprire ed eseguire i notebook in ordine sequenziale da `01` a `05`.

---

## Autori

* **Simone Rinaldi**
* **Matteo Legati**