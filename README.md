# UN SDG scripts

This repository contains a set of scripts for processing and storing
the datasets from the [UN Sustainable Development Goals](https://unstats.un.org/sdgs/indicators/database) database.

## [`1-extract.ipynb`](./1-extract.ipynb)

This script accepts a CSV file exported from the online tool provided by the UN, and produces a set of CSVs to be later processed by a separate script.

  - `entities.csv`: needs to be standardized and entities missing in the database need to be inserted into the `entities` table in `grapher`.
  - `datasets.csv`: records to be inserted into the `datasets` table in `grapher`.
  - `variables.csv`: records to be inserted into the `variables` table in `grapher`.
  - `sources.csv`: records to be inserted into the `sources` table in `grapher`.
  - `*_datapoints.csv*`: records to be inserted into the `data_values` table in `grapher`.

## [`2-import.ipynb`](./2-import.ipynb)

This script reads the CSVs referenced before, and imports their contents as records in the relevant tables in the `grapher` database.
