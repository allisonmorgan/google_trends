# Predicting Google Search Trends

Final Project for CSCI 5466 - Chaotic Dynamics

## Introduction

The goals of this project were to see whether of not some Google search trends are predictable. Class presentation can be found in [`slides.pdf`](https://github.com/allisonmorgan/google_trends/blob/master/slides.pdf).

### Setup

This package requires `pytrends`, `numpy`, `sklearn`, `matplotlib` and `pandas`. This [`notebook`](https://github.com/allisonmorgan/google_trends/blob/master/parse_google_trends.ipynb) can be used to download a Google search trend. To learn more about the data, read Samantha Molnar's [blog post](http://samanthamolnar.me/personal/2017/05/02/hacking-google-trends.html).

To play with one of the trends already downloaded ("baseball", "influenza", and "full moon"), you may run:

```{bash}
python main.py "full moon"
``` 

This will generate a plot of the [time series](https://github.com/allisonmorgan/google_trends/blob/master/data/fullmoon_hourly.csv.trend.png). It will also create plots of [mutual information](https://github.com/allisonmorgan/google_trends/blob/master/data/fullmoon_hourly.csv.mi.png) and percentage of [false nearest neighbors](https://github.com/allisonmorgan/google_trends/blob/master/data/fullmoon_hourly.csv.fnn.png) - the steps required to [delay-coordinate embed](https://github.com/allisonmorgan/google_trends/blob/master/data/fullmoon_hourly.csv.embed.png) the time series. Finally, it will produce a [prediction](https://github.com/allisonmorgan/google_trends/blob/master/data/fullmoon_hourly.csv_prediction.png) for the last 20% of the time series.

<img src="https://github.com/allisonmorgan/google_trends/blob/master/data/fullmoon_hourly.csv_prediction.png?raw=true"/>

In blue is the real time series, and in red are is our single-step forecast using Lorenz Method of Analogues with k=5.

### Works Cited

I would highly recommend looking over Joshua Garland and Liz Bradley's paper ["Prediction in Projection"](https://arxiv.org/abs/1503.01678) for information about delay-coordinate embedding and prediction using Lorenz Method of Analogues (LMA).

```
J. Garland and E. Bradley, "Prediction in projection," Chaos 25:123108 (2015)
```

The file [`embedding.py`](https://github.com/allisonmorgan/google_trends/blob/master/embedding.py) contains python wrappers for some functions from the time series analysis package, [TISEAN](https://www.mpipks-dresden.mpg.de/~tisean/Tisean_3.0.1/index.html). Binaries relevant to this project have been reproduced in the [`tisean`](https://github.com/allisonmorgan/google_trends/tree/master/tisean) folder.

```
R. Hegger, H. Kantz, and T. Schreiber, Practical implementation of nonlinear time series methods: The TISEAN package, CHAOS 9, 413 (1999)
```
