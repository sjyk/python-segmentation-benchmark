import logging
import numpy as np
import scipy.linalg
import scipy.cluster
import scipy.spatial
import time

"""
This is contrib code from
https://github.com/yoojioh/gamelanpy/tree/master/gamelanpy
"""


def get_coreset(data, num_clusters, coreset_size, delta=0.1):
    '''
    Parameters
    ----------
    data: array-like, (num_frames, num_vars)
    num_clusters: int, number of clusters
    coreset_size: int, number of coreset samples
    delta: float, default=0.1

    Return
    ------
    samples: coreset samples
    weights: coreset weights
    '''

    logger = logging.getLogger()

    if len(data.shape) == 1:
        logger.debug('Input data is 1-D, converting it to 2-D')
        data = data[:, np.newaxis]

    num_frames, num_vars = data.shape

    if coreset_size < num_clusters:
        raise ValueError("coreset_size %d is less than num_mixtures %d" % (coreset_size, num_clusters))

    if num_frames < coreset_size:
        raise ValueError("num_frames %d is less than coreset_size %d" % (num_frames, num_clusters))

    data_remain = data.copy()

    samples = np.zeros((0, num_vars))
    # first, do the subsampling : pick core samples, and remove closest point to
    # it

    logger.debug('Before Coreset random sampling')

    num_iters = 0
    num_single_samples = int(1.0 * num_vars * num_clusters * np.log(1.0 / delta))
    logger.debug('num_single_samples: %d', num_single_samples)

    while data_remain.shape[0] > num_single_samples:
        cur_time = time.time()
        logger.debug('Starting iteration %d', num_iters)

        num_frames_remain = data_remain.shape[0]
        idx = np.random.permutation(num_frames_remain)[:num_single_samples]
        single_samples = data_remain[idx, :]

        prev_time = cur_time
        cur_time = time.time()
        logger.debug('After random sampling (took %.3f sec)', cur_time - prev_time)

        # Here we define similarity matrix, based on some measure of
        # similarity or kernel. Feel free to change

        dists = scipy.spatial.distance.cdist(data_remain, single_samples)

        prev_time = cur_time
        cur_time = time.time()
        logger.debug('After evaluating cdist (took %.3f sec)', cur_time - prev_time)

        # minimum distance from random samples
        min_dists = np.min(dists, axis=1)
        # median distance
        v = np.median(min_dists)

        # remove rows with distance <= median distance
        remove_idx = np.where(min_dists <= v)[0]

        # remove rows of remove_idx
        data_remain = np.delete(data_remain, remove_idx, 0)
        samples = np.vstack((samples, single_samples))
        logger.debug('Shape of the coreset samples so far (%d, %d)', samples.shape)
        logger.debug('Shape of the remaining samples (%d, %d)', data_remain.shape)

        prev_time = cur_time
        cur_time = time.time()
        logger.debug('End of iteration %d (took %.3f sec)', (num_iters, cur_time - prev_time))

        num_iters += 1
    # end of while loop

    logger.debug('Shape of the final remaining samples (%d, %d)', data_remain.shape)

    samples = np.vstack((samples, data_remain))

    logger.debug('Shape of the final coreset samples (%d, %d)', samples.shape)

    # now compute the weights of all the points, according to how close they
    # are to the closest core-sample.
    db_size = np.zeros(samples.shape[0])
    min_dists = np.zeros(num_frames)
    closest_sample_idx = np.zeros(num_frames)
    for i in xrange(num_frames):
        dists = scipy.spatial.distance.cdist(data[i:i+1, :], samples)
        min_dist = np.min(dists)
        min_idx = np.argmin(dists)
        min_dists[i] = min_dist
        closest_sample_idx[i] = min_idx

    for i in xrange(num_frames):
        # for each datapoint, Ix[i] is the index of the coreset point
        # it is assigned to.
        db_size[closest_sample_idx[i]] += 1

    sq_sum_min_dists = (min_dists ** 2).sum()
    m = np.zeros(num_frames)
    for i in xrange(num_frames):
        m[i] = np.ceil(5.0 / db_size[closest_sample_idx[i]] + (min_dists[i] ** 2) / sq_sum_min_dists)

    m_sum = m.sum()
    cdf = (1.0 * m / m_sum).cumsum()
    samples = np.zeros((coreset_size, num_vars))
    weights = np.zeros(coreset_size)

    # Now, sample from the weighted points, to generate final corset
    # and the corresponding weights
    for i in xrange(coreset_size):
        r = np.random.rand()
        idx = (cdf <= r).sum()
        samples[i, :] = data[idx, :]
        weights[i] = m_sum / (coreset_size * m[idx])

    return samples, weights



