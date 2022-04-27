# Diamonds Prediction

This is the classic Diamonds dataset.

## Data Description 

The description is provided from the `R` Documentation. 

```r
?diamonds
```
A dataset containing the prices and other attributes of almost 54,000 diamonds. Included is `volume` and `price_per_carat`, accordingly. The variables are as follows:

**Format**
A data frame with 53940 rows and 12 variables:

`carat`: Weight of the diamond (0.2–5.01). 
`cut`: Quality of the cut (`Fair`, `Good`, `Very Good`, `Premium`, `Ideal`).
`color`: Diamond color, from D (best) to J (worst).
`clarity`: A measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best)).
`depth`:	Width of top of diamond relative to widest point (43–95).
`table`: Total depth percentage = z / mean(`x`, `y`) = 2 * `z` / (`x` + `y`) (43–79).	
`price`: Price in US dollars (\$326–\$18,823).	
`x`: Length in mm (0–10.74).
`y`: Width in mm (0–58.9).
`z`: Depth in mm (0–31.8).
`volume`: The amount of space the diamond occupies, `x` times `y` times `z`.
`price_per_carat`: The price per carat of each diamond, `price` divided by `carat`.


## Original Diamonds Dataset

### Found [here](https://github.com/kyle-w-brown/diamonds-prediction/blob/main/data/diamonds.csv)

The original Diamonds dataset was built for `R`, but has since found a home in `Python`. Seaborn now stores the dataset as a built in to its package. 

```python
import seaborn as sns
sns.get_dataset_names()
sns.load_dataset('diamonds')
```

|   |  carat |    cut	  | color	| clarity	| depth	| table	| price	|  x   |   y	|   z  |
|:-:|:------:|:--------:|:-----:|:-------:|:-----:|:-----:|:-----:|:----:|:----:|:----:|
| 0	|  0.23	 |   Ideal  |   E	  |   SI2 	|  61.5	|  55.0	|  326 	| 3.95 | 3.98 |	2.43 |
| 1	|  0.21	 |  Premium	|   E   |   SI1	  |  59.8	|  61.0	|  326  | 3.89 | 3.84 |	2.31 |
| 2	|  0.23	 |   Good	  |   E	  |   VS1   |  56.9	|  65.0	|  327  | 4.05 | 4.07 |	2.31 |
| 3	|  0.29	 |  Premium	|   I   |   VS2 	|  62.4	|  58.0	|  334  | 4.20 | 4.23 |	2.63 |
| 4	|  0.31	 |   Good	  |   J   |   SI2 	|  63.3	|  58.0	|  335	| 4.34 | 4.35 |	2.75 |


## Cleaned Data

### Found [here](https://github.com/kyle-w-brown/diamonds-prediction/blob/main/data/diamonds_cleaned.csv)

The Diamonds dataset was cleaned by removing outliers from scatterplots and imputing the column means to `nan`. Columns `volume` and `price_per_carat` were created for further analysis and model features. 

|   |  carat |    cut	  | color	| clarity	| depth	| table	| price	|  x   |   y	|   z  | volume | price_per_carat |
|:-:|:------:|:--------:|:-----:|:-------:|:-----:|:-----:|:-----:|:----:|:----:|:----:|:------:|:----------------:|
| 0	|  0.23	 |   Ideal  |   E	  |   SI2 	|  61.5	|  55.0	|  326 	| 3.95 | 3.98 |	2.43 | 38.202 |      1417.39     |
| 1	|  0.21	 |  Premium	|   E   |   SI1	  |  59.8	|  61.0	|  326  | 3.89 | 3.84 |	2.31 | 34.506 |      1552.38     |
| 2	|  0.23	 |   Good	  |   E	  |   VS1   |  56.9	|  65.0	|  327  | 4.05 | 4.07 |	2.31 | 38.077 |      1421.74     |
| 3	|  0.29	 |  Premium	|   I   |   VS2 	|  62.4	|  58.0	|  334  | 4.20 | 4.23 |	2.63 | 46.725 |      1151.72     |
| 4	|  0.31	 |   Good	  |   J   |   SI2 	|  63.3	|  58.0	|  335	| 4.34 | 4.35 |	2.75 | 51.917 |      1080.65     |

