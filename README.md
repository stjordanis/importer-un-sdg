# UN SDG scripts

This repository contains a set of scripts for processing and storing
the datasets from the [UN Sustainable Development Goals](https://unstats.un.org/sdgs/indicators/database) database.

## `UN Sustainable Development Goals.ipynb`

This script accepts a CSV file exported from the online tool provided by the UN, and produces a set of CSVs to be later processed by a separate script.

  - `datasets.csv`: records to be inserted into the `datasets` table in `grapher`.
  - `vsriables.csv`: records to be inserted into the `variables` table in `grapher`.
  - `*_datapoints.csv*`: records to be inserted into the `data_values` table in `grapher`.
  
## `Import UN SDG.ipynb`

This script reads the CSVs referenced before, and imports their contents as records in the relevant tables in the `grapher` database.


