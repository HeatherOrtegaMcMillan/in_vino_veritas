# In Vino Veritas
### Predicting wine ratings based on their chemical properties

## Project Overview

## Goals
The goal with this project is to create a machine learning model to predict the quality of a wine based on it's physicochemical properties. Also it will be to determine key drivers in the wine quality. This Dataset was acquired through UCI's Machine Learning Repository and can be found here

## Findings

## Conclusion

## Data Info
Title: Wine Quality

Citation: 

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
Modeling wine preferences by data mining from physicochemical properties.
In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

  Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
                [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
                [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib

The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine.
For more details, consult: http://www.vinhoverde.pt/en/ or the reference [Cortez et al., 2009].
Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output)variables 
are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).


### Data Dictionary

Two features `is_good` and `is_bad` are boolian, target categories engineered from the `quality` 
column. The models are attempting to predict whether or not the wine will be a stand out wine
(rating of 7 or higher), or of poor quality (rating of 4 or below).

| Column Name                | Use       | Type            |
|----------------------------|-----------|-----------------|
| `quality`                  | Target    | int64           |
| `quality_bins`             | Target*   | int64 (boolean) |
| `is_good`                  | Target*   | int64 (boolean) |
| `is_bad`                   | Target*   | int64 (boolean) |
| `fixed acidity`            | Variable  | Float64         |
| `volatile acidity`         | Variable  | Float64         |
| `citric acid`              | Variable  | Float64         |
| `residual sugar`           | Variable  | Float64         |
| `chlorides`                | Variable  | Float64         |
| `free sulfur dioxide`      | Variable  | Float64         |
| `total sulfur dioxide`     | Variable  | Float64         |
| `density`                  | Variable  | Float64         |
| `pH`                       | Variable  | Float64         |
| `sulphates`                | Variable  | Float64         |
|  `alcohol`                 | Variable  | Float64         |
| `is_white`                 | Variable* | int64 (boolean) |


* indicates an engineered feature



## Re Creation
If you wish to recreate this project download the csv files for red wine and white wine and save them to your repo. More information about this data can be found [here] <---- Insert link here

