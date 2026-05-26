---
toc_depth: 2
---

# Toolkit documentation

*dartsort* is also a toolkit for common spike sorting analyses, and the main spike sorter is built out of those tools.
The most important tools are the ["Peelers"](#detecting-cleaning-and-featurizing-spikes-peeling), which detect spikes and featurize them.
These rely on [featurization pipelines](#featurization-pipelines).
Other tools include [clustering](#clustering) and [template waveform estimation](#template-waveform-estimation) from spike trains.

## Detecting, cleaning, and featurizing spikes ("peeling")

In spike sorting workflows like template matching or thresholding-based spike detection, there is a common sequence of steps: spikes are detected, there is possibly some iterative subtraction of estimated clean waveforms, and finally waveforms are extracted and featurized.
In *dartsort* those kind of workflows are called "peelers" (inspired by the iterative subtraction of events) and are implemented as subclasses of a `BasePeeler` class which handles the shared logic (processing chunks of the recording in parallel, fitting featurization models, ...).

All of the peelers accept a featurization pipeline or a [`FeaturizationConfig`](#dartsort.FeaturizationConfig) object which handles the spike featurization; these are discussed [in the next section](#featurization-pipelines).

*dartsort* includes high-level functions for running various kinds of peelers and corresponding configuration objects.
These are in sections below.

### Template matching

[`match()`](#dartsort.match) runs template matching from known templates if its `template_data` parameter is set, or else estimating templates from `sorting` using [`estimate_template_library()`](#dartsort.estimate_template_library).

It is configured by the [matching_cfg](#dartsort.MatchingConfig) argument.

::: dartsort.match

::: dartsort.MatchingConfig
    options:
      show_if_no_docstring: true

### Neural-net based collision-cleaned spike detection

::: dartsort.subtract

::: dartsort.SubtractionConfig
    options:
      show_if_no_docstring: true

### Thresholding spike detection

::: dartsort.threshold

::: dartsort.ThresholdingConfig
    options:
      show_if_no_docstring: true


### Spike extraction and featurization at known event times

::: dartsort.grab


## Featurization pipelines

Featurization is configured by building a `FeaturizationConfig`.
Inside *dartsort*, the FeaturizationPipeline will be turned into a [`WaveformPipeline`](#dartsort.WaveformPipeline) with its [`.from_config()`](#dartsort.WaveformPipeline.from_config) constructor.

::: dartsort.FeaturizationConfig
    options:
      show_if_no_docstring: true

::: dartsort.WaveformPipeline

This pipeline is a sequence of denoising and featurization objects from the `dartsort.transform` module:

::: dartsort.transform
    options:
       filters:
         - "!^WaveformConfig"


## Clustering

*dartsort* includes configuration options and a main function for running several clustering strategies.

::: dartsort.cluster

::: dartsort.ClusteringFeaturesConfig
    options:
      show_if_no_docstring: true

::: dartsort.ClusteringConfig
    options:
      show_if_no_docstring: true

::: dartsort.RefinementConfig
    options:
      show_if_no_docstring: true


## Template waveform estimation

::: dartsort.estimate_template_library

::: dartsort.TemplateData
    options:
      show_if_no_docstring: true
      filters:
       - "!^_"

::: dartsort.TemplateConfig
    options:
      show_if_no_docstring: true
