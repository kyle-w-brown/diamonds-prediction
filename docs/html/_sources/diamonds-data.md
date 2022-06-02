# Diamonds Dataset

This is the classic Diamonds dataset.

## Data Description 

This  dataset contains the prices and other attributes of almost 54,000 diamonds. The variables are as follows:

**Format**

A data frame with 53940 rows and 12 variables:

- `carat`: Weight of the diamond (0.2–5.01). 
- `cut`: Quality of the cut (`Fair`, `Good`, `Very Good`, `Premium`, `Ideal`).
- `color`: Diamond color, from `D` (best) to `J` (worst).
- `clarity`: A measurement of how clear the diamond is (`I1` (worst), `SI2`, `SI1`, `VS2`, `VS1`, `VVS2`, `VVS1`, `IF` (best)).
- `depth`:	Width of top of diamond relative to widest point (43–95).
- `table`: Total depth percentage = `z` / mean(`x`, `y`) = 2 * `z` / (`x` + `y`) (43–79).	
- `price`: Price in US dollars (\$326–\$18,823).	
- `x`: Length in mm (0–10.74).
- `y`: Width in mm (0–58.9).
- `z`: Depth in mm (0–31.8).


## Original Diamonds Dataset

The original Diamonds dataset was built for `R`, but has since found a home in `Python`. Seaborn now stores the dataset as a built in to its package. 


## Loading the Data

```python
import seaborn as sns
sns.get_dataset_names()

sns.load_dataset('diamonds')
```

or

```python
diamonds = "https://raw.githubusercontent.com/kyle-w-brown/diamonds-prediction/main/data/diamonds.csv"
df_diamonds = pd.read_csv(diamonds)
df_diamonds.head()
```

|   |  carat |    cut	  | color	| clarity	| depth	| table	| price	|  x   |   y	|   z  |
|:-:|:------:|:--------:|:-----:|:-------:|:-----:|:-----:|:-----:|:----:|:----:|:----:|
| 0	|  0.23	 |   Ideal  |   E	  |   SI2 	|  61.5	|  55.0	|  326 	| 3.95 | 3.98 |	2.43 |
| 1	|  0.21	 |  Premium	|   E   |   SI1	  |  59.8	|  61.0	|  326  | 3.89 | 3.84 |	2.31 |
| 2	|  0.23	 |   Good	  |   E	  |   VS1   |  56.9	|  65.0	|  327  | 4.05 | 4.07 |	2.31 |
| 3	|  0.29	 |  Premium	|   I   |   VS2 	|  62.4	|  58.0	|  334  | 4.20 | 4.23 |	2.63 |
| 4	|  0.31	 |   Good	  |   J   |   SI2 	|  63.3	|  58.0	|  335	| 4.34 | 4.35 |	2.75 |


## Updated Data

Column `volume`, `cut_rk`, `color_rk`, and `clarity_rk` were created for ordinal encoding and model features. 

- `volume`: The amount of space the diamond occupies (`x` x `y` x `z`).
- `cut_rk`: The cut rank was ordinal encoded of `cut` 1-5 from best (Ideal) to worst (Fair) diamonds. 
- `color_rk`: The color rank was ordinal encoded of `color` 1-7 from best (D) to worst (J) diamonds
- `clarity_rk`: The clarity rank was ordinal encoded of `clarity` 1-8 from best (I1) to worst (IF) diamonds.  

## Loading Updated Data

```python
diamonds = "https://raw.githubusercontent.com/kyle-w-brown/diamonds-prediction/main/data/diamonds-new.csv"
df_diamonds = pd.read_csv(diamonds)
df_diamonds.head()
```


|   |  carat |    cut	  | color	| clarity	| depth	| table	| price	|  x   |   y	|   z  | volume | cut_rk |	color_rk |	clarity_rk |
|:-:|:------:|:--------:|:-----:|:-------:|:-----:|:-----:|:-----:|:----:|:----:|:----:|:------:|:------:|:--------:|:-----------:|
| 0	|  0.23	 |   Ideal  |   E	  |   SI2 	|  61.5	|  55.0	|  326 	| 3.95 | 3.98 |	2.43 | 38.202 |   1   |	    2    |	    7      |
| 1	|  0.21	 |  Premium	|   E   |   SI1	  |  59.8	|  61.0	|  326  | 3.89 | 3.84 |	2.31 | 34.506 |   2   |   	2	   |      6      |
| 2	|  0.23	 |   Good	  |   E	  |   VS1   |  56.9	|  65.0	|  327  | 4.05 | 4.07 |	2.31 | 38.077 |   4	  |     2    |	    4      |
| 3	|  0.29	 |  Premium	|   I   |   VS2 	|  62.4	|  58.0	|  334  | 4.20 | 4.23 |	2.63 | 46.725 |   2   |    	6	   |      5      |
| 4	|  0.31	 |   Good	  |   J   |   SI2 	|  63.3	|  58.0	|  335	| 4.34 | 4.35 |	2.75 | 51.917 |   4   |    	7    |    	7      |