# Proof of Concept — TUS EEG Classification

This repository contains minimal code for a proof-of-concept experiment comparing **classical DSP/ML** and **graph-based GNN** pipelines for **Transcranial Ultrasound Stimulation (TUS)** EEG analysis.

## Quick Start

```bash
git clone https://github.com/<your-username>/proof_of_concept.git
cd proof_of_concept
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Data

Place the three EDF recordings under:

```
dataset/TUS/
├── EEG 2_EPOCX_218863_2025.08.28T10.34.10+12.00.edf   # post-sham
├── EEG 3_EPOCX_218863_2025.08.28T10.58.26+12.00.edf   # pre-active
└── EEG 4_EPOCX_218863_2025.08.28T11.31.55+12.00.edf   # post-active
```

Update paths if needed in `config.py`.

## Output

All metrics, plots, and model results are saved in `results/`.

## TODO (IEEE OJEMB readiness)

* [ ] Add git tag + DOI versioning
* [ ] Include checksum manifest for EDFs and outputs
* [ ] Record git commit info in results CSVs
* [ ] Add reproducibility scripts and environment lockfile

## License

MIT License
