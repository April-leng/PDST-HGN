##  Project Structure

The project is modularized into the following components for better maintainability:

```text
.
├── main.py          # Entry point: Argument parsing, training loop, and evaluation
├── model.py         # Model Assembly: HSDHN_Hybrid_SOTA main architecture
├── layers.py        # Core Operators: InceptionTCN, HSDHGNNLayer, DynamicRegionGrowing
├── utils.py         # Utilities: Data loading, FFT features, MI precomputation, metrics
├── data/            # Directory for datasets (PEMS03, 04, 07, 08)
└── save/            # Directory for saved models and MI cache
