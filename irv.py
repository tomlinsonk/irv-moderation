import argparse
import os
import pickle
from multiprocessing import Pool

import numpy as np
from numpy.random import default_rng
from scipy import stats
from tqdm import tqdm


def fast_continuous_plurality_voter_dsn(cand_pos, voter_dsn):
    cand_idxs = np.argsort(cand_pos)
    ordered_cands = cand_pos[cand_idxs]

    regions = np.concatenate(((0,), (ordered_cands[1:] + ordered_cands[:-1]) / 2, (1,)))

    cdfs = voter_dsn.cdf(regions)
    votes = np.diff(cdfs)

    return cand_idxs[np.argmax(votes)]


def fast_continuous_irv_voter_dsn(cand_pos, voter_dsn):
    k = len(cand_pos)

    cand_idxs = np.argsort(cand_pos)
    ordered_cands = cand_pos[cand_idxs]

    while k > 1:
        regions = np.concatenate(((0,), (ordered_cands[1:] + ordered_cands[:-1]) / 2, (1,)))
        cdfs = voter_dsn.cdf(regions)
        votes = np.diff(cdfs)
        elim = np.argmin(votes)

        k = k - 1
        cand_idxs = np.delete(cand_idxs, elim)
        ordered_cands = np.delete(ordered_cands, elim)

    return cand_idxs[0]


def simulate_1d_custom_dsn_winner_distribution_helper(args):
    k, dsn, trial_set = args

    dsn.random_state = default_rng(seed=k*trial_set[0])

    results = dict()

    for trial in trial_set:
        cand_pos = dsn.rvs(k)

        plurality_winner = fast_continuous_plurality_voter_dsn(cand_pos, dsn)
        irv_winner = fast_continuous_irv_voter_dsn(cand_pos, dsn)

        extremities = np.abs(0.5 - cand_pos)

        most_moderate = np.argmin(extremities)
        most_extreme = np.argmax(extremities)
        median = np.median(cand_pos)

        results[k, trial] = cand_pos[plurality_winner], cand_pos[irv_winner], cand_pos[most_moderate], cand_pos[most_extreme], median

    return results


def simulate_1d_custom_dsn_winner_distribution(ks, dsn, trials, threads, name):
    print('Running simulate_1d_custom_dsn_winner_distribution', ks, dsn, trials, threads, name)
    params = ((k, dsn, trial_set) for k in ks for trial_set in np.array_split(np.arange(trials), threads))

    all_results = dict()
    with Pool(threads) as pool:
        for sub_results in tqdm(pool.imap_unordered(simulate_1d_custom_dsn_winner_distribution_helper, params), total=len(ks)*threads):
            all_results.update(sub_results)

    with open(f'results/{name}-dsn-winner-positions-{trials}-trials.pickle', 'wb') as f:
        pickle.dump((ks, dsn, trials, all_results), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int)
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)

    trials = 1_000_000
    simulate_1d_custom_dsn_winner_distribution([3, 4, 5], stats.uniform(), trials, args.threads, f'uniform-small-k-{trials}-trials')

    trials = 100_000
    sweep_k = 30
    for alpha in np.linspace(0.05, 3.95, 79):
        simulate_1d_custom_dsn_winner_distribution([sweep_k], stats.beta(alpha, alpha), trials, args.threads, f'beta-{alpha}-{alpha}-k-{sweep_k}-{trials}-trials')

    max_k = 20
    simulate_1d_custom_dsn_winner_distribution(range(2, max_k+1), stats.uniform(), trials, args.threads, 'uniform-k-2-20')

    high_ks = [50, 100, 150, 200, 250, 500, 1000]
    simulate_1d_custom_dsn_winner_distribution(high_ks, stats.uniform(), trials, args.threads, 'uniform-k-high')







