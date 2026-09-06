# Data access and redistribution

This repository separates redistributable release material from third-party molecular matrices whose
redistribution is restricted. The excluded files are not needed to audit the reported numbers or
rebuild the paper figures. They are needed to retrain a model or score a new cell line.

## Included public inputs

- `data/GDSC2_IC50.csv`: aligned 536 x 169 response matrix (81,467 measured pairs);
- `data/drug_info.csv`: compound identifiers, names and structures;
- `data/drug/morgan_encoding.pkl`: 2,048-bit Morgan fingerprints;
- `data/cell/cell_line_info.csv`: public identifier mapping;
- `data/splits/primary_seed42.json`: frozen 428/54/54 primary train/validation/test split;
- `data/splits/fivefold_seed42.json`: frozen five-fold cell-disjoint partition;
- `data/ctrp/`: harmonized CTRPv2 evaluation records and annotations;

## Excluded restricted matrices

The following four feature matrices are deliberately absent from the release:

```text
data/cell/geo_expression_cosmic.csv
data/cell/geo_methylation_cosmic.csv
data/cell/geo_mutation_cosmic.csv
data/cell/pathway_cosmic.csv
```

Obtain the underlying CCLE/DepMap molecular profiles and the COSMIC Cancer Gene Census under their
current provider terms. For the final model, place the authorized, aligned expression and methylation
tables at the first two paths above. Rows must be DepMap IDs; the expression and methylation tables
must contain 714 and 603 features, respectively. Mutation and pathway matrices are not used by the
final BCGDRP or BANDRP-Gate implementations, but are listed because they were used in comparator and
modality analyses.

Run:

```bash
python scripts/data.py --strict
```

before training or prediction. The command checks dimensions, cell coverage, response counts and
fingerprint widths without copying or modifying any source data.

## Experimental measurements

The canonical public wet-lab input is `source_data/Fig5_wetlab_replicates.csv`. It contains every
replicate well used for the seven reported dose-response curves. The original laboratory workbooks
are not distributed because they contain unrelated laboratory context; no reported value requires
them. `experiments/analysis/dose.py` refits the public tidy table directly.

See the article Data Availability statement for provider URLs and citations. The MIT License applies
to original code, not to third-party datasets.
