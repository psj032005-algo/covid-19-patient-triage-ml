# Data (not committed - licence: non-commercial / educational use only)

Download "Routine Blood Values Datasets that can be used to predict the diagnosis, prognosis and mortality of
COVID-19" (Huyut, Mendeley Data, v1): https://data.mendeley.com/datasets/w9mdp9k472/1
and cite the four papers listed on that page.

**Put the downloaded `.sav` files in this folder with their original names - no renaming needed.**
The code finds the six it uses by keyword (case-insensitive):

| used for | file name contains |
|---|---|
| ICU admission | `biochemical` / `hematological` / `immunological` + `mildly-severely` |
| In-hospital death | `biochemical` / `hematological` / `immunological` + `Survivor` |

The three "patients-healty" (diagnosis) files may stay in the folder; they are ignored.
To keep the data elsewhere: `set COVID_DATA_DIR=C:\path\to\folder` (Windows) / `export COVID_DATA_DIR=/path` (Linux, macOS).
