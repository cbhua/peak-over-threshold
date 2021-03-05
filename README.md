# Peak over Threshold

Implement of algorithms about Peaks-over-Threshold, including POT, Stream POT, and Streaming POT with drift.

This repository mainly refers to paper: *Siffer, Alban, et al. "Anomaly detection in streams with extreme value theory." Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2017.*, including introduction and algoritm fake codes. 

## Introduction

The Peak Over Threshold-method (POT-method) is one way to model extreme values. This method relies on the Pickands-Balkema-de Haan theorem. The main concept of the method is to use a threshold to seclude values considered extreme to the rest of the data and create a model for the extreme values by modeling the tail of all the values the exceeds this threshold. 

But POT method has the problem that when the data is static, or is a stream coming from an extremely controlled environment, such assumptions can safely be made. But in the general case of streaming data from an open environment, these assumptions are no longer true. They may fail in unexpected cases. 

So we should consider some method can deal with streaming dataset. The Streaming POT (SPOT-method) and the Streaming POT with drift (DSPOT-method) are rised by Alban et al. Then I will introduce these three methods one by one. 

### Peak over Threshold Method

The Peak over Threshold (POT) mehtod will process on the whole dataset to get one threshold by such process: 

![POT Algorithm](/figs/pot_algorithm.png)

In the first step, set initial threshold, we choose a initial rate for data. After descending sroting data, we choose the `(total datapoints number * initial rate)`th datapoint as the initial threshold.

For the second step, Grimshaw's Process, the paper provides a trick to get the result. For more details please refer to the paper. We will get a list of threshold candidate, and then we go through such candidates to find the best threshold as our final result. 

POT is a basic tool, to process more types of data, we shoudl have more powerful methods. 

### Streaming Peak over Threshold Method

The Sreaming Peak over Threshold (SPOT) Method can process such datasets changing with time, mostly time-series data. Compare with original POT, SPOT can handle long time-series data processing. 

The algorithm is shown below: 

![SPOT Algorithm](/figs/spot_algorithm.png)

SPOT firstly uses POT to get an initial threshold (t) and base threshold (z), based on beginning several datapoints (here we set 1000). And then sequentially traversal all rest datapoints, during this process it will keep updating the threshold. 

In this method, datapoints can be classified to three types: 

1. Normal: datapoints below the initial threshold;
2. Peaks: datapoints between initial threshold and base threshold, such datapoints will not be detected as anomaly, but will affect the threshold;
3. Anomaly: datapoints over the base threshold, such datapoints will be detected as anomaly datapoints directly, will not affect the threshold. 

With the processing, if the datapoints go up, with the help of peaks, the threshold will also adapt to the trend. 

But SPOT still has problem that it can only be used on such datasets without drift. To deal with this problem, there is anothre method DSPOT. 

### Streaming Peak over Threshold with drift Method

The Streaming Peak over Threshold with drift (DSPOT) Method has similar structure with SPOT, except the model will update the mean value every several steps. The algorithm is shown as following: 

![DSPOT Algorithm](/figs/dspot_algorithm.png)

Compared with SPOT algorithm, DSPOT has "mean updating part" during the data process. Which can process the dataset with drift. 

## Requirements

The implement of such methods is done by Python, with packages `numpy`, `scipy` and `matplotlib`. Tests are created by Jupyter Notebook.  

```Python 
Python Version: >= 3.6. 
Packages: numpy, scipy, matplotlib, jupyter notebook / jupyter lab. 
```

## Repository Structure

- data: example data is saved here;
- figs: some result figures and readme images are saved here;
- src: all codes are here, each method has one single file;
    - utils: tools like grimshaw calculator are saved here;
- test: some tests for three methods, we can refer to how to use such methods and check results. 

## How to Use

**Step 1.** Clone or Download this repository. 

**Step 2.** Run files within `test` folder by jupyter notebook or jupyter lab to check how to use functions and the running results. 

**Step 3.** Check source function files to modify what you want. Comments are provided and functions are written following the algorithm shown before.  

## Example Results 

**POT**

![POT Result](/figs/pot.png)

**SPOT**

![SPOT Result](/figs/spot.png)

**DSPOT**

![DSPOT Result](/figs/dspot.png)

## Reference

Siffer, Alban, et al. "Anomaly detection in streams with extreme value theory." Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2017.

## Licence

[MIT License](https://opensource.org/licenses/mit-license.php)