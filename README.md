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

### ⚙️ Come funziona nel workflow quotidiano

1. **File tracciati da Git**: Solo i file `.py` (e la configurazione `jupytext.toml`) sono tracciati da Git. I file `.ipynb` sono ignorati da `.gitignore`.
2. **Sviluppo locale**: Quando salvi un notebook `.ipynb` in VS Code o JupyterLab, Jupytext aggiornerà automaticamente il corrispettivo file `.py`.

### 🔄 Comandi Utili per Jupytext

* **Rigenerare tutti i notebook `.ipynb` dal codice `.py`** (ad es. dopo un `git pull`):
  ```bash
  jupytext --to notebook notebooks/*.py
  ```

* **Associare un nuovo notebook appena creato** (se non associato automaticamente):
  ```bash
  jupytext --set-formats ipynb,py:percent notebooks/mio_nuovo_notebook.ipynb
  ```

### ⚓ Git Hook Automatico (Cross-Platform)

È stato configurato un hook `pre-commit` compatibile sia con **Linux / macOS** che con **Windows** (`Git Bash` / `CMD` / `PowerShell`).

Ad ogni `git commit`, l'hook:
1. Individua automaticamente l'ambiente Python (`.venv/bin/jupytext` su Linux/macOS o `.venv/Scripts/jupytext` su Windows).
2. Sincronizza i file `.ipynb` modificati nei rispettivi script `.py`.
3. Aggiunge automaticamente i file `.py` aggiornati al commit.

Non dovrai preoccuparti di eseguire manualmente `jupytext --sync` prima di fare il commit.