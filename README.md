[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17401702.svg)](https://doi.org/10.5281/zenodo.17401702)

# Graph-Wavelet Patterns with Graph Attention Networks for Cognitive State Classification: A Nested Optimization and Explainable AI Framework



Minimal codebase for comparing **classical DSP/ML** and **graph-based GNN** pipelines in **Transcranial Ultrasound Stimulation (TUS)** EEG analysis.

## Setup

```bash
git clone https://github.com/<your-username>/eeg-gat.git
cd proof_of_concept
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Outputs are saved in:

```
results/   # Figures, metrics
tracking/  # MLflow, Optuna, TensorBoard logs
```

## Data

Place EDF recordings in:

```
dataset/TUS/
├── EEG 2_EPOCX_...edf   # post-sham
├── EEG 3_EPOCX_...edf   # pre-active
└── EEG 4_EPOCX_...edf   # post-active
```

Update paths in `config.py` if needed.

## License

MIT License


## References

- Virtanen, P. et al. (2020). *SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python*. Nature Methods, 17, 261–272.
- Gramfort, A. et al. (2013). *MEG and EEG data analysis with MNE-Python*. Frontiers in Neuroscience, 7:267.
- Fey, M. & Lenssen, J.E. (2019). *Fast Graph Representation Learning with PyTorch Geometric*. ICLR.
- Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD.
- Paszke, A. et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS.
