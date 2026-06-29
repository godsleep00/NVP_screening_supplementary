# Cardio Herb DB

This directory contains the cardiovascular herb-compound database files and supporting utilities used as supplementary material for the manuscript.

The data are organized by herb. Each herb-level folder stores compound tables and, where available, structure files, molecular images, source spreadsheets, and processing scripts.

## Contents

```text
cardio_herb_db/
|-- Compounds/                 # Herb-level compound data, images, and SDF files
|-- cardio_herb_db.py          # Desktop database query and visualization program
|-- cas_to_smiles.py           # CAS-to-SMILES processing utility
|-- LICENSE
`-- README.md
```

## Data

Most folders under `Compounds/` contain one herb-specific compound table, commonly named `compounds.csv`, together with generated molecular structure images and SDF files when available.

The compound tables may include fields such as compound identifier, Chinese name, English name, CAS number, molecular formula, molecular weight, compound category, SMILES, InChI, InChIKey, and herb source. Field names are not fully identical across all source tables because the records were compiled from herb-specific files.

## Database Program

`cardio_herb_db.py` is a Windows desktop program used for compound querying, structure display, and database interaction. It uses PyQt5 and RDKit and supports database connections to MySQL or SQL Server.

The default MySQL connection uses `localhost:3306` and the database name `compounds`. Set the database password with the `CARDIO_HERB_DB_PASSWORD` environment variable before running the program. No database password is stored in this repository.

```bash
python cardio_herb_db.py
```

## CAS-to-SMILES Utility

`cas_to_smiles.py` extracts CAS numbers from Excel files and can query PubChem through `pubchempy` to obtain SMILES strings.

```bash
python cas_to_smiles.py --help
```

## Environment

Main Python dependencies include:

```bash
pip install PyQt5 rdkit-pypi pandas numpy pymysql pyodbc pubchempy pywin32 Pillow openpyxl
```

The full desktop workflow is intended for Windows because some image-extraction steps use Windows COM automation for Excel and ChemDraw.

## License

This directory is released under the MIT License. See `LICENSE` for details.
