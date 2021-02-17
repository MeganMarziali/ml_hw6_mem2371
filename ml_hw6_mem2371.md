Assignment 6
================
Megan Marziali
Feb 17, 2021

## Assignment Set-Up

``` r
library(tidyverse)
library(NHANES)
library(Amelia)
library(caret)

data(NHANES)
```

## Problem 1: Import and Restrict Data

``` r
nhanes = NHANES %>% 
  janitor::clean_names() %>% 
  select(
    age, race1, education, hh_income, weight, height, 
    pulse, diabetes, bmi, phys_active, smoke100
  )
```

The NHANES data has 10,000 observations. To investigate missingness, I
used the mapping function.

``` r
missmap(nhanes)
```

![](ml_hw6_mem2371_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

It seems that education, smoking and pulse have a large amount of
missing observations. However, I opted to keep all variables and exclude
missing observations.

``` r
nhanes_restr = nhanes %>% na.omit()
```

With missing observations remove, the total number of observations in
this dataset is 6356. I next checked the balance of the outcome
observations within the dataset:

``` r
summary(nhanes_restr$diabetes)
```

    ##   No  Yes 
    ## 5697  659

There are 5697 “no” responses, and 659 “yes” responses, for a prevalence
of diabetes within this sample of 11.6%.

``` r
train.indices = createDataPartition(y = nhanes_restr$diabetes,p = 0.7,list = FALSE)

training = nhanes_restr[train.indices,]
testing = nhanes_restr[-train.indices,]
```
