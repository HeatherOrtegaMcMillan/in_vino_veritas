# In Vino Veritas
### Predicting wine ratings based on their chemical properties

![](images/grapevine.png)

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

| Column Name                | Use       | Type            | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|----------------------------|-----------|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `quality`                  | Target    | int64           | Average quality rating of the wine. 0 being worst 10 being best                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `quality_bins`             | Target*   | int64           | Bins created to represent bad (0), average (1) and good (2) wines.                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 'quality_bins_str'         | Target*   | Object          | Same bins as `quality_bins` but data is represented in a string for readability                                                                                                                                                                                                                                                                                                                                                                                                         |
| `is_good`                  | Target*   | int64 (boolean) | Wine has rating of 7 or higher                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `is_bad`                   | Target*   | int64 (boolean) | Wine has rating of 4 or lower                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `fixed acidity`            | Variable  | Float64         | Fixed acids include tartaric, malic, citric, and succinic acids which are found in grapes (except succinic). This variable is usually expressed in  g(tartaricacid)/dm3  in the dataset.                                                                                                                                                                                                                                                                                                 |
| `volatile acidity`         | Variable  | Float64         | These acids are to be distilled out from the wine before completing the production process. It is primarily constituted of acetic acid though other acids like lactic, formic and butyric acids might also be present. Excess of volatile acids are undesirable and lead to unpleasant flavor. In the US, the legal limits of volatile acidity are 1.2 g/L for red table wine and 1.1 g/L for white table wine. The volatile acidity is expressed in  g(aceticacid)/dm3  in the dataset. |
| `citric acid`              | Variable  | Float64         | This is one of the fixed acids which gives a wine its freshness. Usually most of it is consumed during the fermentation process and sometimes it is added separately to give the wine more freshness. It's usually expressed in  gdm3  in the dataset.                                                                                                                                                                                                                                  |
| `residual sugar`           | Variable  | Float64         | This typically refers to the natural sugar from grapes which remains after the fermentation process stops, or is stopped. It's usually expressed in  gdm3  in the dataset.                                                                                                                                                                                                                                                                                                              |
| `chlorides`                | Variable  | Float64         | Chloride concentration in the wine is influenced by terroir and its highest levels are found in wines coming from countries where irrigation is carried out using salty water or in areas with brackish terrains. This is usually a major contributor to saltiness in wine. It's usually expressed in  g(sodiumchloride)/dm3  in the dataset.                                                                                                                                            |
| `free sulfur dioxide`      | Variable  | Float64         | This is the part of the sulphur dioxide that when added to a wine is said to be free after the remaining part binds. Winemakers will always try to get the highest proportion of free sulphur to bind. They are also known as sulfites and too much of it is undesirable and gives a pungent odour. This variable is expressed in  mgdm3  in the dataset.                                                                                                                               |
| `total sulfur dioxide`     | Variable  | Float64         | This is the sum total of the bound and the free sulfur dioxide ( SO2 ). Here, it's expressed in  mgdm3 . This is mainly added to kill harmful bacteria and preserve quality and freshness. There are usually legal limits for sulfur levels in wines and excess of it can even kill good yeast and give out undesirable odour.                                                                                                                                                          |
| `density`                  | Variable  | Float64         | This can be represented as a comparison of the weight of a specific volume of wine to an equivalent volume of water. It is generally used as a measure of the conversion of sugar to alcohol. Here, it's expressed in  gcm3 .                                                                                                                                                                                                                                                           |
| `pH`                       | Variable  | Float64         | Also known as the potential of hydrogen, this is a numeric scale to specify the acidity or basicity the wine. Fixed acidity contributes the most towards the pH of wines. You might know, solutions with a pH less than 7 are acidic, while solutions with a pH greater than 7 are basic. With a pH of 7, pure water is neutral. Most wines have a pH between 2.9 and 3.9 and are therefore acidic.                                                                                     |
| `sulphates`                | Variable  | Float64         | These are mineral salts containing sulfur. They are connected to the fermentation process and affects the wine aroma and flavor. Here, it's expressed in  g(potassiumsulphate)/dm3  in the dataset.                                                                                                                                                 |
|  `alcohol`                 | Variable  | Float64         | Measured in % vol or ABV (Alcohol by Volume)                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `is_white`                 | Variable* | int64 (boolean) | Boolean value indicating if the wine is white. If 0 wine is red. This is from joining the red and white wine data sets together.                                                                                                                                                                                                                                                                                                                                                        |



* indicates an engineered feature



## Re Creation
If you wish to recreate this project download the csv files for red wine and white wine and save them to your repo. More information about this data can be found [here] <---- Insert link here

