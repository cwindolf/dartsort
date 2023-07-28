Reference position-free, stable region-free, drift-invariant templates

## Introduction

In a drifting recording, a naive average of templates does not make sense.
Rather, one should shift the template's component waveforms to account for
the drift before averaging.

To do this, we will compute a template on virtual channels. Let the probe
consist of $P$ pitches, each a local channel neighborhood with consistent
structure (up to missing channels etc). Define a virtual probe on $3P-2$
pitches: this is the maximum possible extent that a unit could have, if it
drifted a full probe length. This virtual probe has a "central probe", and
it is padded by two full probes, each missing one pitch, on either side.

E.g., for NP1, the virtual probe looks like:
```
       .
       .
       .
    o   o
      o   o
    o   o          virtual
      o   o        padding
    o   o
___   o   o ____
    o   o
      o   o
       .
       .
       .
    o   o          central
      o   o        probe
       .
       .
       .
    o   o
___   o   o ____
    o   o
      o   o
    o   o          virtual
      o   o        padding
    o   o
      o   o
       .
       .
       .
```

Each unit's spikes will be shifted to land inside the central
probe before averaging. Then, during deconvolution, a probe-
length portion (i.e., $P$ pitches) channels of the template
computed on the virtual probe should be extracted at each time
point according to the drift at that time point and at the
unit's depth (see the definition of $v(t)$ below).

If the drift is rigid, picking channels like this works; if
nonrigid, then we are assuming that the nonrigidity is
negligible at the scale of individual units (<200um diameter),
which is fair enough and not worth trying to avoid, and anyway
is it even true that individual templates would get "bent"
when the brain bends?

## Algorithm

Concretely, here is the algorithm for computing the free
template for a single unit. This is a thorough and conceptual
description with justifications, with formal notation describing
the exact computations.

Probe notation (idealized probe without missing channels,
there is a note below about extending to the realistic case):
 - Number of channels $C$, consisting of $P$ pitch groups,
   so that each pitch has $C/P$ channels.

Parameters:
 - $B$, the number of superresolution bins in the central
   pitch; equivalently can be parameterized by the bin
   spacing $h=\Delta/B$.
 - Binning mode: either by displacement estimate (mode
   "p"), by the z position (mode "z"), or a practical
   mixture of both (mode "hybrid"). More on this below.

Inputs:
 - A drift estimate $p$, which here we treat as a function
   $p(t,z)$ giving the drift at time $t$ and depth $z$.
 - Spike times and (unregistered) depths $t_1,...,t_m$ and
   $z_1,...,z_m$ for a particular unit.
 - Waveforms $w_1,...,w_m$ for this unit, extracted on all
   channels, so that $w_i$ is $T\times C$.

Define:
 - Let $p_i=p(t_i,z_i)$ and $\bar{p}$ be the average of these.
 - Registered depths $r_i = z_i - p(t_i,z_i)$ for this unit.
 - The unit's registered location $\bar{r}$, which is the mean
   or better yet median of $r_1,...,r_m$.
 - The unit's implied "virtual position" trace
   $v(t) = \bar{r} + p(t, \bar{r})$.
 - The floor division $\lfloor a/b \rfloor$, which is the largest
   integer such that $b \lfloor a/b \rfloor \leq a$.


Now, for each spike, we must do two things:
 1. Figure out how many pitches to shift the waveform before
    averaging it with other waveforms
     - This signed integer quantity is called $k_i$ and must
       be determined for each spike.
 2. Assign the spike to a superres bin, which determines the
    groups of spikes that will be averaged together
     - This integer bin identity is called $b_i$ and must
       be determined for each spike.
(1) handles the large scale (bigger than a pitch) component of the
drift, and (2) handles the sub-pitch part.

Depending on the binning mode, we will change how this is done.

### In detail: the binning modes, what are they?

First, what are they?
 - In mode "z", each spike's localization alone determines where we think it
   is relative to the unit: the number of pitches $k_i$ is the integer part
   of $z_i - \bar{r}$ mod the pitch, and the bin identities are determined
   from the remainder.
 - In mode "p", the drift estimate alone determines the above: the number
   of pitches $k_i$ is the integer part of $p_i - \bar{p}$ mod the pitch,
   and the bin identity is chosen based on the remainder.
 - Mode "hybrid" seeks a middle ground, generalizing the previous "stable
   region" approach. First, the recording is divided into several "stable
   regions", which are just determined by binning the drift estimate mod
   the pitch. Then, $k_i$ is exactly this drift bin identity, which is the
   same as setting $k_i$ as in mode "p". The superres bin ids are then
   determined by binning $z_i - k_i\Delta$ -- which is the residual of $z$
   modulo which drift-bin we are in.

Formally, how are $k_i$ and $b_i$ chosen in each case?

 - Modes "z" and "p":
    - If mode is "z", let $o_i=z_i-\bar{r}$. If mode is "p", let
      $o_i=p_i-\bar{p}$.
    - Let $k_i = \lfloor (o_i + \Delta/2) / \Delta\rfloor$
    - Let $s_i = o_i - k_i\Delta$ be the remaining sub-pitch
      drift, and note $s_i\geq0$ by construction.
    - $b_i$ is obtained as $\lfloor (s_i + h/2) / h\rfloor$.
 - Mode "hybrid":
    - Let $k_i = \lfloor (p_i - \bar{p} + \Delta/2) / \Delta\rfloor$
    - Let $s_i = z_i - \bar{r} - k_i\Delta$ and $b_i=\lfloor (s_i + h/2) / h\rfloor$.
    - Note that here, the bin ids are signed and may include values
      outside $\{0,...,B - 1\}$.

### Why include all of these modes, and how do they differ?

Mode "p" is useful when the depth estimate is noisy and the
drift estimate is good; mode "z" is useful when the depth estimate
is good and the drift estimate is noisy.

The downside to using mode "z" is that we may unnecessarily shift
and misassign superres bins for units with noisy locations, but this
is done at random so is in some sense "unbiased". The downside to mode
"p" is that we unnecessarily shift and misassign superres bins when
the drift estimate is bad, and this could introduce a systemic error
since drift failures tend to be correlated through time -- so, this
can lead to bias.

Mode "hybrid" goes for a middle ground, where the drift estimate is
fairly good with <1 pitch length $\Delta$ in error or unmodeled fast
drift. Spikes are grouped together (shifted together by $k_i$)
according to the drift estimate, which we assume catches the pitch-
scale drift. But, spikes are separated into superres groups after
this according to $z_i$ within each pitch group, allowing for error
in the drift estimate. Note also that these bins can span more than
a pitch, and the choice of bin spacing $h$ is arbitrary and need not
divide the pitch length $\Delta$.

"hybrid" mode is practical, since it uses information from the drift
estimate but hedges this by allowing for some unmodeled drift, without
suffering from the noise issue that mode "z" suffers from when deciding
$k_i$. So, it is conservative about small drifts, but exploits knowledge
of large drifts.

### Finishing the algorithm

Assume now that we have chosen $k_i$ and $b_i$ somehow according to
the previous section. How do we compute a template?

For each occupied bin $b$, let $I_b = \lbrace i : b_i = b \rbrace$ be the indices
of waveforms in this bin.

For each waveform, we want to pad to the full $3P-2$ pitch virtual
probe according to $k_i$ so that it is centered on the central probe.
If $k_i$ is positive, we want to shift down by $k_i$ pitches.
This means that we pad above by $P + k_i - 1$ pitches, and we pad below
by $P - k_i - 1$ pitches, for a total of $3P-2$. Define the padded waveform

$$
\begin{align}
u_i = \text{pad}(w_i, (C/P)(P - k_i - 1), (C/P)(P + k_i - 1)),
\end{align}
$$

where here the pad operation $pad(w,c_1,c_2)$ is adding extra channels filled with
missing value markers (NaNs) that can be accounted for during averaging. $c_1$
channels are added below and $c_2$ above.

Then the template is simply computed as the missing-value aware average
of the padded waveforms $u_i$.

## Important implementation details

 - In practice, we have missing channels and the probe is not a perfect
   set of repeating units; the narrative above ignored this for clarity.
   We can handle this, but I haven't written it down yet.
 - Usually we want denoised templates -- so we need to keep track of
   the number of spikes that make it into each virtual channel to 
   make sure we compute our per-channel SNR right. Other than that
   things are the same.
