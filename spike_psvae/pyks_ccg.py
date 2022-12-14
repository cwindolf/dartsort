"""
This code is taken verbatim from PyKilosort.
"""
import numpy as np
import numba
from scipy.special import erf


def ccg_metrics(st1, st2, nbins, tbin):
    """
    For two arrays of spike times, use the cross-correlogram to estimate the contamination rate
    and the statistical significance that there are fewer spikes in the refractory period
    :param st1: Array of spike times (seconds), numpy or cupy array
    :param st2: Array of spike times (seconds), numpy or cupy array
    :param nbins: Number of time bins either side, int
    :param tbin: Length of each time bin, float
    :return: contam_ratio: Proportion of refractory period violations, float
             p_value: Statistical significance of fewer spikes in the refractory period, float
    """
    K = ccg(st1, st2, nbins, tbin)
    contam_ratio, p_value = _ccg_metrics(K, nbins)
    return contam_ratio, p_value


def ccg(st1, st2, nbins, tbin):
    """
    Computes the cross-correlogram for two arrays of spike times
    :param st1: Array of spike times (seconds), numpy or cupy array
    :param st2: Array of spike times (seconds), numpy or cupy array
    :param nbins: Number of time bins either side, int
    :param tbin: Length of each time bin, float
    :return: Cross-correlogram, numpy array
    """
    if (len(st1) == 0) or (len(st2) == 0):
        return np.zeros(2 * nbins + 1)

    return _ccg(st1, st2, nbins, tbin)


def _ccg_metrics(K, nbins):
    # Indices for the tails of the ccg
    irange1 = np.concatenate(
        (np.arange(1, nbins // 2), np.arange(3 * nbins // 2, 2 * nbins))
    )

    # Indices for left shoulder of the ccg
    irange2 = np.arange(nbins - 50, nbins - 10)

    # Indices for right shoulder of the ccg
    irange3 = np.arange(nbins + 11, nbins + 50)

    # Estimate the average non-refractory ccg rate by the maximum rate across these ranges
    ccg_rate = max(
        np.mean(K[irange1]),
        np.mean(K[irange2]),
        np.mean(K[irange3]),
    )

    # Set centre of CCG to 0 to avoid double-counted spikes
    K[nbins] = 0

    # test the probability that a central area in the autocorrelogram might be refractory
    # test increasingly larger areas of the central CCG

    contam_rates = np.zeros(10)
    p_values = np.zeros(10)

    for i in range(1, 11):
        irange = range(nbins - i, nbins + i + 1)

        # for this central range of the CCG compute the mean CCG rate
        # as central value is set to 0, divide by 2*i
        contam_rates[i - 1] = np.sum(K[irange]) / 2 * i

        n = np.sum(K[irange]) / 2
        lam = ccg_rate * i
        if lam == 0:
            p = 1
        else:
            # NOTE: make sure lam is not zero to avoid divide by zero error
            # lam = max(1e-10, R00 * i)

            # log(p) = log(lam) * n - lam - gammaln(n+1)

            # this is tricky: we approximate the Poisson likelihood with a gaussian of equal mean
            # and variance that allows us to integrate the probability that we would see <N spikes
            # in the center of the cross-correlogram from a distribution with mean R00*i spikes

            p = 1 / 2 * (1 + erf((n - lam) / np.sqrt(2 * lam)))

        p_values[i - 1] = p  # keep track of p for each bin size i

    # Use the central region that has lowest refractory violation rate
    p_value = np.min(p_values)
    if ccg_rate == 0:
        if np.min(contam_rates) == 0:
            contam_ratio = 0  # CCG is empty so contamination rate set to 0
        else:
            contam_ratio = 1  # Contamination rate is infinite so set to full contamination
    else:
        contam_ratio = np.min(contam_rates) / ccg_rate

    return contam_ratio, p_value


@numba.jit(nopython=True)
def _ccg(st1, st2, nbins, tbin):
    """JIT compiled ccg function for speed"""
    st1 = np.sort(
        st1
    )  # makes sure spike trains are sorted in increasing order
    st2 = np.sort(st2)

    dt = nbins * tbin

    # Avoid divide by zero error.
    # T = max(
    #     1e-10,
    #     np.max(np.concatenate((st1, st2)))
    #     - np.min(np.concatenate((st1, st2))),
    # )
    N1 = max(1, len(st1))
    N2 = max(1, len(st2))

    # we traverse both spike trains together, keeping track of the spikes in the first
    # spike train that are within dt of spikes in the second spike train

    ilow = 0  # lower bound index
    ihigh = 0  # higher bound index
    j = 0  # index of the considered spike

    K = np.zeros(2 * nbins + 1)

    while j <= N2 - 1:  # traverse all spikes in the second spike train

        while (ihigh <= N1 - 1) and (st1[ihigh] < st2[j] + dt):
            ihigh += 1  # keep increasing higher bound until it's OUTSIDE of dt range

        while (ilow <= N1 - 1) and (st1[ilow] <= st2[j] - dt):
            ilow += (
                1  # keep increasing lower bound until it's INSIDE of dt range
            )

        if ilow > N1 - 1:
            break  # break if we exhausted the spikes from the first spike train

        if st1[ilow] > st2[j] + dt:
            # if the lower bound is actually outside of dt range, means we overshot (there were no
            # spikes in range)
            # simply move on to next spike from second spike train
            j += 1
            continue

        for k in range(ilow, ihigh):
            # for all spikes within plus/minus dt range
            ibin = np.rint((st2[j] - st1[k]) / tbin)  # convert ISI to integer
            ibin2 = np.asarray(ibin, dtype=np.int64)

            K[ibin2 + nbins] += 1

        j += 1

    return K
