---
toc_depth: 3
---

# Main API: `dartsort()` and the `DARTsortUserConfig`

This page shows the main functions and objects that you'd run into when spike sorting with *dartsort*.
If you're just getting started, the [usage section on the front page](/#usage) might be good to read first. 

For details on parameters you should think about before running the sorter, see the [important configuration details note](/#important-configuration-details).

## Main function: `dartsort()`

::: dartsort.dartsort
    options:
      handler: python
      show_signature: true
      separate_signature: true

The return value from the `dartsort()` function is a DARTsortReturn object, which is a dictionary containing spike trains and motion information:

::: dartsort.DARTsortReturn
    options:
      show_bases: false

The spike trains and motion info are stored in some internal objects, described in the [last section on this page](#data-objects). Up next, we'll discuss how to adjust *dartsort*'s parameters.


## Configuration

For details on parameters you should know about before running the sorter, see the [important configuration details note](/#important-configuration-details).
Here, we'll show a reference for all of the configuration options.
Some of the important ones not mentioned in that note include: the [voltage threshold](#dartsort.DARTsortUserConfig.voltage_threshold) for initial spike detection and the "energy" thresholds for spike detection in the [initial](#dartsort.DARTsortUserConfig.initial_threshold) and [template matching](#dartsort.DARTsortUserConfig.matching_threshold) passes.
You may also want to take a look at the paramters for 

::: dartsort.DARTsortUserConfig
    options:
      show_if_no_docstring: true



## Data objects

*dartsort* uses some internal classes to represent spike trains and other data. Spike trains and motion are represented in the following objects.

::: dartsort.DARTsortSorting

::: dartsort.MotionInfo
