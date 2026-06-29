# NVP Screening Supplementary Materials

This repository contains supplementary source code and data for the manuscript on machine-learning-based prediction and screening of natural vascular protectants.

The materials include model-development scripts, training and prediction datasets, exploratory R analyses, vasodilation assay data, and a cardiovascular herb-compound database used during candidate screening.

## Repository Contents

```text
.
|-- training set.xlsx                 # Training data used for machine learning
|-- prediction set.xlsx               # Candidate/prediction data
|-- 机器学习预测.py                    # Python machine-learning workflow
|-- 数据探索.Rmd                       # Exploratory data analysis in R
|-- 岭回归分种类预测.Rmd               # Ridge-regression analysis by category
|-- 血管舒张数据库/                    # Vasodilation experimental data and web table
|-- cardio_herb_db/                   # Cardiovascular herb-compound database and tools
|-- LICENSE
`-- README.md
```

## Main Analyses

`机器学习预测.py` contains the Python workflow for model training, evaluation, and candidate prediction. The script uses the training and prediction Excel files in the repository root.

`数据探索.Rmd` contains exploratory data analysis and statistical summaries.

`岭回归分种类预测.Rmd` contains ridge-regression analyses for compound-category-level prediction.

## Data Directories

`血管舒张数据库/` stores vasodilation-related source tables, EC50 values, and a lightweight web table under `血管舒张数据库/web/`.

`cardio_herb_db/` stores curated cardiovascular traditional Chinese medicine compound data, including compound tables, structure files, molecular images, and scripts for database construction and query. See `cardio_herb_db/README.md` for details.

## Environment

The Python analysis workflow was developed with Python 3 and commonly used scientific-computing packages:

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn openpyxl
```

The compound database utilities additionally use:

```bash
pip install PyQt5 rdkit-pypi pymysql pyodbc pubchempy pywin32 Pillow
```

The R Markdown analyses use packages including:

```r
install.packages(c(
  "readxl", "tidyverse", "easystats", "broom", "glmnet",
  "caret", "randomForest", "kernlab", "reshape2", "ggplot2",
  "openxlsx"
))
```

Some R code also uses `YangPac`; install it according to its package source if it is not available in the local R environment.

## Usage Notes

Run the Python machine-learning workflow from the repository root so relative paths resolve correctly:

```bash
python 机器学习预测.py
```

Render the R Markdown files in RStudio or with `rmarkdown::render()` after installing the listed packages.

The desktop compound database program is located at:

```bash
python cardio_herb_db/cardio_herb_db.py
```

If connecting the desktop database program to MySQL, set the database password through the `CARDIO_HERB_DB_PASSWORD` environment variable before running the program. No database password is stored in this repository.

## Reproducibility

These files are provided as supplementary material for manuscript review and record keeping. Some workflows depend on local database configuration, Windows desktop automation, or external chemical-data services, so exact reruns may require matching the local environment described above.

## Citation

Citation details will be updated after the manuscript information is finalized.

## License

This repository is released under the MIT License. See `LICENSE` for details.
