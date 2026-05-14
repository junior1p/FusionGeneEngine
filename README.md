# FusionGeneEngine

Pure Python fusion gene detection and scoring pipeline.

## Features
- Split-read + discordant pair filtering
- In-frame fusion prediction (exon structure)
- Oncogenic scoring (20 known fusions database)
- COSMIC cancer gene census lookup
- Precision/recall evaluation

## Usage
```bash
pip install numpy scipy pandas matplotlib
python fusion_gene_engine.py
```

## Results (synthetic RNA-seq, 200 candidates)
- 52 high-confidence fusions detected
- Precision=0.96, Recall=1.00, F1=0.98
- 17 in-frame fusions
- 14 match known oncogenic database
- Top: BCR-ABL1 (score=9.8, CML driver)
