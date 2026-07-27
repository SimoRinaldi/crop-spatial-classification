# Crop Spatial Classification

Progetto per la classificazione spaziale e l'analisi del suolo / coperture del suolo (Land Cover).

## 🚀 Requisiti e Installazione

Assicurati di avere Python installato nel tuo ambiente. Per installare tutte le dipendenze necessarie, esegui:

```bash
pip install -r requirements.txt
```

---

## 📓 Gestione dei Notebook e Versionamento con Jupytext

Per evitare di committare i file binari pesanti `.ipynb` su Git (con output, immagini e diff illeggibili), questo repository utilizza **[Jupytext](https://jupytext.readthedocs.io/)**.

I notebook vengono sincronizzati automaticamente in script Python trasparenti a Git nella cartella `notebooks/` in formato `percent` (`.py`).

### 🔌 Configurazione VS Code (Consigliata)

Per far sì che il file `.py` venga aggiornato automaticamente ad ogni salvataggio (`Ctrl+S`):

1. Installa l'estensione VS Code **[Jupytext](https://marketplace.visualstudio.com/items?itemName=mwouts.jupytext)** (`mwouts.jupytext`).
2. Quando salvi un file `.ipynb`, l'estensione aggiornerà automaticamente il file `.py` speculare.
3. Fai il commit su Git del solo file `.py` aggiornato.

### ⚙️ Come funziona il workflow

1. **File tracciati da Git**: Solo i file `.py` (e la configurazione `jupytext.toml`) sono tracciati da Git. I file `.ipynb` sono ignorati da `.gitignore`.
2. **Sviluppo locale**: Lavori sui notebook `.ipynb` normalmente. L'estensione VS Code / Jupyter aggiornerà il file `.py`.

### 🔄 Comandi Utili per Jupytext

* **Rigenerare tutti i notebook `.ipynb` dal codice `.py`** (ad es. dopo un `git pull`):
  ```bash
  jupytext --to notebook notebooks/*.py
  ```

* **Associare un nuovo notebook appena creato** (se non associato automaticamente):
  ```bash
  jupytext --set-formats ipynb,py:percent notebooks/mio_nuovo_notebook.ipynb
  ```

* **Sincronizzare manualmente i file se necessario**:
  ```bash
  jupytext --sync notebooks/*.py
  ```