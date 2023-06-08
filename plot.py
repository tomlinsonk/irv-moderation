import glob
import math
import os
from collections import defaultdict
from functools import partial

import numpy as np
import pickle
import matplotlib.pyplot as plt
import pylab as pl
import yaml
from matplotlib import patches, cm
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.stats import gaussian_kde, wilcoxon, bootstrap

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

RESULTS_DIR = config['resultsdir']


def plot_truncation_heatmaps():

    names = ['1d', '1d-partial', 'general', 'general-partial']
    nice_names = ['1-Euclidean', '1-Euclidean partial', 'General', 'General partial']

    for name, nice_name in zip(names, nice_names):

        cp = '-cand-pos' if '1d' in name else ''

        with open(f'results/{name}-truncation-results-40-max-10000-trials{cp}.pickle', 'rb') as f:
            results = pickle.load(f)

        agree_frac = results[-3]

        agree_frac = np.hstack((np.zeros((agree_frac.shape[0], 1)), agree_frac))
        agree_frac = np.vstack((np.zeros((1, agree_frac.shape[1])), agree_frac))

        fig, ax = plt.subplots(figsize=(4, 3))

        cmap = cm.get_cmap('inferno').copy()
        cmap.set_bad('w')

        agree_frac = np.ma.masked_array(agree_frac, mask=agree_frac==0)


        plt.imshow(agree_frac, cmap='inferno', origin='lower')
        plt.xlim(0.5, 40.5)
        plt.ylim(2.5, 40.5)
        plt.colorbar(label='Pr(IRV winner wins)')

        w1 = patches.Wedge((0, 0), 100, 0, 45, fc='white', hatch=r'++')
        w2 = patches.Wedge((0, 0), 100, 0, 45, fc='white', alpha=0.95)

        ax.add_patch(w1)
        ax.add_patch(w2)


        # for i, collection in enumerate(cs.collections):
        #     collection.set_edgecolor('red')
            # collection.set_linewidth(0.)


        # y, x = np.where(agree_frac==1)
        # plt.scatter(x, y, marker='*', color='black')

        plt.ylabel('# candidates ($k$)')
        plt.xlabel('ballot length ($h$)')
        plt.title(nice_name)
        plt.savefig(f'plots/{name}-ballot-length-heatmap.pdf', bbox_inches='tight')
        plt.show()
        plt.close()


def plot_combined_small_heatmaps():
    names = ['general', '1d']
    nice_names = ['General', '1-Euclidean']

    fig, axes = plt.subplots(1, 2, sharey='row', figsize=(6, 2.5))

    for col, (name, nice_name) in enumerate(zip(names, nice_names)):
        cp = '-cand-pos' if '1d' in name else ''

        with open(f'results/{name}-truncation-results-40-max-10000-trials{cp}.pickle', 'rb') as f:
            results = pickle.load(f)

        agree_frac = results[-3]

        agree_frac = np.hstack((np.zeros((agree_frac.shape[0], 1)), agree_frac))
        agree_frac = np.vstack((np.zeros((1, agree_frac.shape[1])), agree_frac))

        cmap = cm.get_cmap('inferno').copy()
        cmap.set_bad('w')

        agree_frac = np.ma.masked_array(agree_frac, mask=agree_frac == 0)

        im = axes[col].imshow(agree_frac, cmap='inferno', origin='lower')
        axes[col].set_xlim(0.5, 40.5)
        axes[col].set_ylim(2.5, 40.5)

        w1 = patches.Wedge((0, 0), 100, 0, 45, fc='white', hatch=r'++')
        w2 = patches.Wedge((0, 0), 100, 0, 45, fc='white', alpha=0.95)

        axes[col].add_patch(w1)
        axes[col].add_patch(w2)

        # for i, collection in enumerate(cs.collections):
        #     collection.set_edgecolor('red')
        # collection.set_linewidth(0.)

        # y, x = np.where(agree_frac==1)
        # plt.scatter(x, y, marker='*', color='black')

        axes[col].set_xlabel('ballot length ($h$)')
        axes[col].set_title(nice_name)

    axes[0].set_ylabel('# candidates ($k$)')

    fig.subplots_adjust(wspace=0.05, right=0.82)
    cbar_ax = fig.add_axes([0.83, 0.11, 0.02, 0.775])
    fig.colorbar(im, cax=cbar_ax, label='Pr(IRV winner wins)')

    plt.savefig(f'plots/combined-ballot-length-heatmap.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


def plot_truncation_num_winners():

    with open(f'results/general-truncation-results-40-max-10000-trials.pickle', 'rb') as f:
        winners_general, _, max_k, _ = pickle.load(f)

    with open(f'results/general-partial-truncation-results-40-max-10000-trials.pickle', 'rb') as f:
        winners_general_partial, _, _, _ = pickle.load(f)

    with open(f'results/1d-truncation-results-40-max-10000-trials-cand-pos.pickle', 'rb') as f:
        winners_1d, _, _, _, _ = pickle.load(f)

    with open(f'results/1d-partial-truncation-results-40-max-10000-trials-cand-pos.pickle', 'rb') as f:
        winners_1d_partial, _, _, _, _ = pickle.load(f)

    ks = np.arange(3, max_k + 1)
    plt.figure(figsize=(4, 2.5))

    for winners, name in zip((winners_general, winners_general_partial, winners_1d, winners_1d_partial),
                             ('general full', 'general partial', '1-Euclidean full',  '1-Euclidean partial')):

        color = 'green' if 'general' in name else 'blue'

        winner_counts = [[len(np.unique(x)) for x in winners[k]] for k in ks]
        winner_count_means = np.mean(winner_counts, axis=1)
        winner_count_stds = np.std(winner_counts, axis=1)

        print(name, winner_count_means)

        plt.plot(ks, winner_count_means, label=name, ls='dashed' if 'partial' in name else 'solid', c=color)
        plt.fill_between(ks, winner_count_means - winner_count_stds,
                         winner_count_means + winner_count_stds, alpha=0.2, color=color)

    plt.xticks([1, 10, 20, 30, 40], [])
    # plt.xlabel('# candidates ($k$)')
    plt.ylabel('mean # truncation winners')
    plt.legend(loc='upper left', fontsize=8)
    plt.savefig('plots/mean-winner-counts.pdf', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(4, 2.5))

    for winners, name in zip((winners_general, winners_general_partial, winners_1d, winners_1d_partial),
                             ('general full', 'general partial', '1-Euclidean full', '1-Euclidean partial')):
        color = 'green' if 'general' in name else 'blue'

        winner_counts = [[len(np.unique(x)) for x in winners[k]] for k in ks]
        winner_count_maxs = np.max(winner_counts, axis=1)
        # winner_count_stds = np.std(winner_counts, axis=1)

        plt.plot(ks, winner_count_maxs, label=name, ls='dashed' if 'partial' in name else 'solid', c=color)
    plt.xticks([1, 10, 20, 30, 40], [])
    plt.plot(ks, np.maximum(1, ks-1), color='black', ls='dotted', label='theoretical max')
    plt.xlabel('# candidates ($k$)')


    plt.xticks([1, 10, 20, 30, 40], [1, 10, 20, 30, 40])

    plt.ylabel('max # truncation winners')
    plt.legend(loc='upper left', fontsize=8)
    plt.savefig('plots/max-winner-counts.pdf', bbox_inches='tight')
    plt.close()


def plot_truncation_winner_patterns(name):
    with open(f'results/{name}-truncation-results.pickle', 'rb') as f:
        all_winners, agree_frac, max_k, n_trials = pickle.load(f)

    os.makedirs(f'plots/{name}-truncation-agree-patterns', exist_ok=True)
    os.makedirs(f'plots/{name}-truncation-winner-patterns', exist_ok=True)

    subset_counts = []
    for k in range(1, max_k+1):

        all_winners[k] = all_winners[k].astype(int)

        distinct_winners = np.count_nonzero(np.diff(np.sort(all_winners[k])), axis=1)+1

        print(k, max(distinct_winners))

        agree_patterns = (all_winners[k] == all_winners[k][:, -1, None]).astype(int)

        unique_agree_patterns = np.unique(agree_patterns, axis=0)

        plt.imshow(unique_agree_patterns, cmap='gray', interpolation='nearest')
        plt.title(f'Unique agreement patterns ($k={k}$)')
        # plt.xlabel('$h$')
        plt.savefig(f'plots/{name}-truncation-agree-patterns/{name}-agree-{k}.pdf', bbox_inches='tight', dpi=1000)
        plt.close()

        subset_counts.append(len(unique_agree_patterns))

        winner_patterns = []

        for i in range(len(all_winners[k])):
            row = all_winners[k][i]

            unique, idx = np.unique(np.flip(row), return_index=True)

            relabel_map = {x: j for j, x in enumerate(unique[np.argsort(idx)])}

            relabeled_row = np.vectorize(relabel_map.get)(row)

            winner_patterns.append(relabeled_row)

        unique_winners = np.unique(winner_patterns, axis=0)
        #
        plt.figure(figsize=(k/2, min(20, len(unique_winners))))
        plt.imshow(unique_winners, cmap='inferno', interpolation='nearest')
        plt.title(f'Unique winner patterns ($k={k}$)')
        # plt.xlabel('$h$')
        plt.savefig(f'plots/{name}-truncation-winner-patterns/{name}-winner-{k}.pdf', bbox_inches='tight', dpi=1000)
        plt.close()

    print(subset_counts)
    print([max(1, 2**(h - 2)) for h in range(1, max_k+1)])
    plt.plot(range(1, max_k+1), subset_counts, label='unique agreement patterns')
    plt.plot(range(1, max_k+1), [max(1, 2**(h - 2)) for h in range(1, max_k+1)], ls='dashed', label='# subsets')
    plt.hlines(10000, 1, max_k, label='# trials', ls='dotted')

    plt.yscale('log')
    plt.legend()
    plt.xlabel('$h$')

    # plt.show()
    plt.savefig(f'plots/{name}-truncation-agree-patterns/pattern-count.pdf', bbox_inches='tight')
    plt.close()


def summarize_results_file(f_name):
    with open(f_name, 'rb') as f:
        all_winners, agree_frac, max_k, n_trials = pickle.load(f)

    for k in np.arange(1, max_k + 1):
        # per-row unique count (https://stackoverflow.com/a/48473125)
        sorted_rows = np.sort(all_winners[k], axis=1)
        unique_winners = (sorted_rows[:, 1:] != sorted_rows[:, :-1]).sum(axis=1) + 1

        print('unique winners', k, np.max(unique_winners))

        # print('winner bincount', k, np.bincount(all_winners[k].flatten().astype(int)))
        #
        # print()
        # print(np.unique(all_winners[k], axis=0))
        # print()
        #


def plot_k_winner_distributions(f_name, outdir):
    with open(f_name, 'rb') as f:
        all_winners, all_cands, agree_frac, max_k, n_trials = pickle.load(f)

    os.makedirs(f'plots/{outdir}', exist_ok=True)

    for k in range(2, max_k + 1):
        print('k', k)
        winner_positions = all_cands[k][np.arange(n_trials)[:, None], all_winners[k].astype(int)]

        for col in range(winner_positions.shape[1]):
            print(f'h = {col+1}, {np.mean(winner_positions[:, col])}{np.std(winner_positions[:, col])}')

        plt.figure(figsize=(4, 4))
        plt.yticks(range(k-1), range(1, k))

        dsns = [winner_positions[:, col] for col in range(winner_positions.shape[1]-1)]
        ridgeline(dsns)
        plt.yticks(fontsize=8)
        plt.xlim(0, 1)

        plt.ylabel('Ballot length')
        plt.xlabel('Winner position')

        plt.savefig(f'plots/{outdir}/{outdir}-{k}-distributions.pdf', bbox_inches='tight')
        # plt.show()
        plt.close()

        variances = np.std(dsns, axis=1)**2

        plt.figure(figsize=(4, 3))
        plt.axes().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.scatter(range(1, len(variances)+1), variances)

        plt.xlabel('Ballot length')
        plt.ylabel('Winner distribution variance')
        plt.savefig(f'plots/{outdir}/{outdir}-{k}-variances.pdf', bbox_inches='tight')
        plt.close()

    # plt.hist(all_cands[2][np.arange(n_trials), all_winners[2][:, 0].astype(int)], bins=1000)
    # plt.show()
    #
    # print(all_cands[2][0], all_winners[2][0])


def make_small_stacked_bars():

    with open(f'results/preflib-resampling/all-resampling-results.pickle', 'rb') as f:
        elections, resampled_results, true_results = pickle.load(f)

    candidate_counts = []
    ballot_lengths = []
    voter_counts = []
    min_unique_winners = []
    max_unique_winners = []
    true_unique_winners = []
    expected_unique_winners = []

    out_dir = 'plots/preflib-data'
    os.makedirs(out_dir, exist_ok=True)

    resample_win_prob_dir = 'plots/preflib-data/resampling-win-probabilities'
    os.makedirs(resample_win_prob_dir, exist_ok=True)

    replace_names = {'ED-00007-00000005': 'ERS Election 5', 'ED-00005-00000002': '2009 Burlington Mayor'}

    burlington_results = None
    ers_results = None


    for collection, election_name, ballots, ballot_counts, cand_names, skipped_votes in elections:
        stripped_election_name = election_name.replace('.soi', '').replace('.toi', '')
        if stripped_election_name == 'ED-00007-00000005':
            ers_results = collection, election_name, ballots, ballot_counts, cand_names, skipped_votes

        elif stripped_election_name == 'ED-00005-00000002':
            burlington_results = collection, election_name, ballots, ballot_counts, cand_names, skipped_votes

    fig, axes = plt.subplots(1, 2, figsize=(6, 2), sharey=True)


    for col, results in enumerate((burlington_results, ers_results)):
        collection, election_name, ballots, ballot_counts, cand_names, skipped_votes = results
        stripped_election_name = election_name.replace('.soi', '').replace('.toi', '')

        winners_seqs = np.array([winners for winners, majority_winner in resampled_results[collection, election_name]])

        true_winner_seq = np.array(true_results[collection, election_name][0])

        candidate_counts.append(len(cand_names))
        # print(len(cand_names))
        ballot_lengths.append(max(len(b) for b in ballots))
        voter_counts.append(sum(ballot_counts))

        unique_winners = np.unique(winners_seqs)
        unique_winners = np.array(sorted(unique_winners, key=lambda x: cand_names[x]))

        true_unique_winners.append(len(np.unique(true_winner_seq)))
        unique_winner_counts = [len(np.unique(row)) for row in winners_seqs]
        min_unique_winners.append(min(unique_winner_counts))
        max_unique_winners.append(max(unique_winner_counts))
        expected_unique_winners.append(np.mean(unique_winner_counts))

        proportions = np.zeros((len(unique_winners), winners_seqs.shape[1]))
        for i, winner in enumerate(unique_winners):
            proportions[i] = np.sum(winners_seqs == winner, axis=0) / winners_seqs.shape[0]

        for winner in reversed(range(unique_winners.shape[0])):
            bottom = np.sum(proportions[:winner], axis=0)
            axes[col].bar(np.arange(1, winners_seqs.shape[1] + 1), proportions[winner], bottom=bottom,
                    label=cand_names[unique_winners[winner]], width=1)

            for h, true_winner in enumerate(true_winner_seq):
                if true_winner == unique_winners[winner]:
                    axes[col].scatter([1 + h], [bottom[h] + proportions[winner][h] / 2], marker='*', color='black')

        if collection != 'ers':
            axes[col].legend(fontsize=8, framealpha=0.8)

        axes[col].set_title(f'{replace_names[stripped_election_name]}')

        # plt.xticks(range(1, winners_seqs.shape[1]+1), fontsize=8)
        axes[col].set_xlim(0.5, winners_seqs.shape[1] + 0.5)
        axes[col].set_ylim(0, 1)

        axes[col].set_xlabel('Ballot length $h$')

        if col == 1:
            axes[col].set_xticks([1, 5, 10, 15, 20, 25])

    axes[0].set_ylabel('Resampling win prob.')
    plt.subplots_adjust(wspace=0.1)

    # plt.show()

    plt.savefig(f'plots/combined-stacked-bars.pdf', bbox_inches='tight')
    plt.close()



def plot_preflib_resampling():
    # Skip duplicate elections with tons of write-ins
    to_skip = ['ED-00018-00000001.soi', 'ED-00018-00000003.soi']

    with open(f'results/preflib-resampling/all-resampling-results.pickle', 'rb') as f:
        elections, resampled_results, true_results = pickle.load(f)

    candidate_counts = []
    ballot_lengths = []
    voter_counts = []
    min_unique_winners = []
    max_unique_winners = []
    true_unique_winners = []
    expected_unique_winners = []

    out_dir = 'plots/preflib-data'
    os.makedirs(out_dir, exist_ok=True)

    resample_win_prob_dir = 'plots/preflib-data/resampling-win-probabilities'
    os.makedirs(resample_win_prob_dir, exist_ok=True)

    replace_names = {'ED-00007-00000005': 'ERS Election 5', 'ED-00005-00000002': '2009 Burlington Mayor'}

    for collection, election_name, ballots, ballot_counts, cand_names, skipped_votes in elections:
        if election_name in to_skip:
            continue

        stripped_election_name = election_name.replace('.soi', '').replace('.toi', '')

        print(collection, election_name)

        winners_seqs = np.array([winners for winners, majority_winner in resampled_results[collection, election_name]])

        true_winner_seq = np.array(true_results[collection, election_name][0])

        candidate_counts.append(len(cand_names))
        # print(len(cand_names))
        ballot_lengths.append(max(len(b) for b in ballots))
        voter_counts.append(sum(ballot_counts))

        unique_winners = np.unique(winners_seqs)
        unique_winners = np.array(sorted(unique_winners, key=lambda x: cand_names[x]))

        true_unique_winners.append(len(np.unique(true_winner_seq)))
        unique_winner_counts = [len(np.unique(row)) for row in winners_seqs]
        min_unique_winners.append(min(unique_winner_counts))
        max_unique_winners.append(max(unique_winner_counts))
        expected_unique_winners.append(np.mean(unique_winner_counts))

        if len(unique_winners) > 1:
            proportions = np.zeros((len(unique_winners), winners_seqs.shape[1]))
            for i, winner in enumerate(unique_winners):
                proportions[i] = np.sum(winners_seqs == winner, axis=0) / winners_seqs.shape[0]

            plt.figure(figsize=(3, 2))
            for winner in reversed(range(unique_winners.shape[0])):
                bottom = np.sum(proportions[:winner], axis=0)
                plt.bar(np.arange(1, winners_seqs.shape[1]+1), proportions[winner], bottom=bottom, label=cand_names[unique_winners[winner]], width=1)

                for h, true_winner in enumerate(true_winner_seq):
                    if true_winner == unique_winners[winner]:
                        plt.scatter([1+h], [bottom[h] + proportions[winner][h] / 2], marker='*', color='black')

            if collection != 'ers':
                plt.legend(fontsize=8, framealpha=0.8)

            if stripped_election_name in replace_names:
                plt.title(f'{replace_names[stripped_election_name]}')
            else:
                plt.title(f'{collection}, {stripped_election_name}, $k={len(cand_names)}$, $n={sum(ballot_counts)}$')
            # plt.xticks(range(1, winners_seqs.shape[1]+1), fontsize=8)
            plt.xlim(0.5, winners_seqs.shape[1]+0.5)
            plt.ylim(0, 1)
            plt.xlabel('Ballot length $h$')
            plt.ylabel('Resampling win prob.')
            plt.savefig(f'{resample_win_prob_dir}/{stripped_election_name}-stacked-bars.pdf', bbox_inches='tight')
            plt.close()

    # fig, axes = plt.subplots(1, 3, figsize=(10, 2.5), sharey=True)
    #
    # labels, counts = np.unique(true_unique_winners, return_counts=True)
    # print(counts)
    # axes[0].bar(labels, counts, align='center')
    # axes[0].set_ylabel('Count')
    # axes[0].set_xlabel('Truncation winners')
    # axes[0].set_title('Actual')
    # # for label, count in zip(labels, counts):
    # #     axes[0].text(label, count, str(count), ha='center')
    #
    # axes[1].hist(expected_unique_winners, bins=30)
    # axes[1].set_xlabel('Truncation winners')
    # axes[1].set_title('Expected')
    #
    # labels, counts = np.unique(max_unique_winners, return_counts=True)
    # axes[2].bar(labels, counts, align='center')
    # axes[2].set_xticks([1, 2, 3, 4, 5])
    # axes[2].set_xlabel('Truncation winners')
    # axes[2].set_title('Maximum')
    #
    # # plt.xscale('log')
    # plt.savefig(f'{out_dir}/truncation-winners.pdf', bbox_inches='tight')
    # plt.close()
    #
    # fig, axes = plt.subplots(1, 3, figsize=(10, 2.5))
    #
    # axes[0].bar(np.arange(max(candidate_counts)+1), np.bincount(candidate_counts), width=1)
    # axes[0].set_xlabel('number of candidates $k$')
    # axes[0].set_ylabel('count')
    # axes[0].set_xlim(left=0)
    #
    # axes[1].bar(np.arange(max(ballot_lengths)+1), np.bincount(ballot_lengths), width=1)
    # axes[1].set_xlabel('ballot length $h$')
    # axes[1].sharey(axes[0])
    # axes[1].set_xlim(left=0)
    #
    # # axes[1].set_ylabel('count')
    #
    # print(voter_counts, candidate_counts)
    #
    # hist, bins = np.histogram(voter_counts, bins=30)
    # logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    # axes[2].hist(voter_counts, bins=logbins)
    # axes[2].set_xscale('log')
    # axes[2].set_xlabel('number of voters $n$')
    # axes[2].set_xticks([10**1, 10**2, 10**3, 10**4, 10**5])
    #
    # plt.savefig(f'{out_dir}/k-and-h.pdf', bbox_inches='tight')
    #
    # plt.close()

    with open(f'results/general-truncation-results-40-max-10000-trials.pickle', 'rb') as f:
        winners_general, _, max_k, _ = pickle.load(f)

    with open(f'results/general-partial-truncation-results-40-max-10000-trials.pickle', 'rb') as f:
        winners_general_partial, _, _, _ = pickle.load(f)

    with open(f'results/1d-truncation-results-40-max-10000-trials-cand-pos.pickle', 'rb') as f:
        winners_1d, _, _, _, _ = pickle.load(f)

    with open(f'results/1d-partial-truncation-results-40-max-10000-trials-cand-pos.pickle', 'rb') as f:
        winners_1d_partial, _, _, _, _ = pickle.load(f)

    ks = np.arange(3, max_k + 1)
    plt.figure(figsize=(4, 2.5))

    for winners, name in zip((winners_general, winners_general_partial, winners_1d, winners_1d_partial),
                             ('general full', 'general partial', '1-Euclidean full', '1-Euclidean partial')):
        color = 'green' if 'general' in name else 'blue'

        winner_counts = [[len(np.unique(x)) for x in winners[k]] for k in ks]
        winner_count_means = np.mean(winner_counts, axis=1)
        winner_count_stds = np.std(winner_counts, axis=1)

        print(name, winner_count_means)

        plt.plot(ks, winner_count_means, label=name, ls='dashed' if 'partial' in name else 'solid', c=color)
        plt.fill_between(ks, winner_count_means - winner_count_stds,
                         winner_count_means + winner_count_stds, alpha=0.2, color=color)

    plt.scatter(np.array(candidate_counts) + (np.random.random(len(candidate_counts)) * 0.5) - 0.25,
                np.array(expected_unique_winners),
                label='PrefLib', marker='.', color='red', s=10, zorder=10)

    # plt.xlabel('# candidates ($k$)')
    plt.ylabel('mean # truncation winners')
    plt.xticks([1, 10, 20, 30, 40], [])

    plt.legend(loc='upper left', fontsize=8)
    plt.savefig('plots/mean-winner-counts.pdf', bbox_inches='tight')
    # plt.show()
    plt.close()

    plt.figure(figsize=(4, 2.5))

    for winners, name in zip((winners_general, winners_general_partial, winners_1d, winners_1d_partial),
                             ('general full', 'general partial', '1-Euclidean full', '1-Euclidean partial')):
        color = 'green' if 'general' in name else 'blue'

        winner_counts = [[len(np.unique(x)) for x in winners[k]] for k in ks]
        winner_count_maxs = np.max(winner_counts, axis=1)
        # winner_count_stds = np.std(winner_counts, axis=1)

        plt.plot(ks, winner_count_maxs, label=name, ls='dashed' if 'partial' in name else 'solid', c=color)

    plt.scatter(np.array(candidate_counts) + (np.random.random(len(candidate_counts)) * 0.5) - 0.25,
                np.array(max_unique_winners), label='PrefLib', marker='.', color='red', s=10, zorder=10)

    plt.plot(ks, np.maximum(1, ks - 1), color='black', ls='dotted', label='theoretical max')
    plt.xlabel('# candidates ($k$)')
    plt.ylabel('max # truncation winners')
    plt.ylim(0, 15)
    plt.xticks([1, 10, 20, 30, 40], [1, 10, 20, 30, 40])
    plt.yticks([1, 5, 10, 15])

    # plt.legend(loc='upper left', fontsize=8)

    plt.text(5, 6, 'theoretical max', rotation=59)
    plt.savefig('plots/max-winner-counts.pdf', bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_preflib_deviances():
    with open('results/preflib-winner-deviances.pickle', 'rb') as f:
        elections, winner_deviances, all_means, all_stds, winner_sequences, all_cand_orders = pickle.load(f)

    print(winner_deviances)

    statistic, p = wilcoxon([d[0] for d in winner_deviances], [d[-1] for d in winner_deviances])

    print(np.mean([d[0] for d in winner_deviances]), np.mean([d[-1] for d in winner_deviances]))
    print(p)


def plot_k_3_winner_distributions():
    f_name = 'results/1d-truncation-results-20-max-1000000-trials-cand-pos.pickle'

    with open(f_name, 'rb') as f:
        all_winners, all_cands, agree_frac, max_k, n_trials = pickle.load(f)

    outdir = 'plots'
    os.makedirs(outdir, exist_ok=True)


    winner_positions = {k: all_cands[k][np.arange(n_trials)[:, None], all_winners[k].astype(int)] for k in range(3, max_k+1)}


    fig, axes = plt.subplots(2, 3, figsize=(15, 5), sharey='row', sharex='col')
    p1 = lambda x: 4 * x + x**2 / 2
    p2 = lambda x: - 3 / 2 + 13 * x - 13*x**2

    p1_x = np.linspace(0, 1/3, 100)
    p2_x = np.linspace(1/3, 1/2, 100)

    # axes[0, 0].plot(p1_x, p1(p1_x), lw=2, zorder=0)
    # axes[0, 0].plot(p2_x, p2(p2_x), lw=2, zorder=0)
    # axes[0, 0].scatter([1/3], [p1(1/3)], marker='o', color='black', zorder=1, s=5)


    i1 = lambda x: 12 * x**2
    i2 = lambda x: 1-12*x + 48*x**2
    i3 = lambda x: -5 + 36*x - 48*x**2
    i4 = lambda x: -1 + 12*x - 12*x**2

    i1_x = np.linspace(0, 1/6, 100)
    i2_x = np.linspace(1/6, 1/4, 100)
    i3_x = np.linspace(1/4, 1/3, 100)
    i4_x = np.linspace(1/3, 1/2, 100)

    # axes[1, 0].plot(i1_x, i1(i1_x), lw=2, zorder=0)
    # axes[1, 0].plot(i2_x, i2(i2_x), lw=2, zorder=0)
    # axes[1, 0].plot(i3_x, i3(i3_x), lw=2, zorder=0)
    # axes[1, 0].plot(i4_x, i4(i4_x), lw=2, zorder=0)
    #
    # axes[1, 0].scatter([1/6], [i1(1/6)], marker='o', color='black', zorder=1, s=5)
    # axes[1, 0].scatter([1/4], [i2(1/4)], marker='o', color='black', zorder=1, s=5)
    # axes[1, 0].scatter([1/3], [i3(1/3)], marker='o', color='black', zorder=1, s=5)


    axes[1, 0].set_ylabel('Density')
    axes[0, 0].set_ylabel('Density')

    for k in (3, 4, 5):
        col = k - 3
        axes[0, col].hist(winner_positions[k][:, 0], bins=100, density=True, alpha=0.4, color='blue', zorder=-2)
        axes[0, col].set_title(f'Plurality, $k={k}$')

        axes[1, col].hist(winner_positions[k][:, -1], bins=100, density=True, alpha=0.4, color='green', zorder=-2)
        axes[1, col].set_title(f'IRV, $k={k}$')
        axes[1, col].axvline(1 / 6, ls='dashed', color='black', lw=1, zorder=0)
        axes[1, col].axvline(5 / 6, ls='dashed', color='black', lw=1, zorder=0)

        print((1/3)**k, np.count_nonzero(winner_positions[k][:, -1] < 1/6) + np.count_nonzero(winner_positions[k][:, -1] > 5/6))

    plt.savefig('plots/moderation/k-3-4-5-winner-distributions-bars.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


def plot_iterative_positioning(h):

    outdir = 'plots/iterative-cand-position'
    os.makedirs(outdir, exist_ok=True)

    for k in range(2, 11):
        with open(f'results/iterative-cand-position/h-{h}-k-{k}-50000-trials-20-iterations-cand-pos.pickle', 'rb') as f:
            winner_positions = pickle.load(f)

        rows = 3
        cols = 7
        fig, axes = plt.subplots(rows, cols, figsize=(30, 10))

        for i, x in enumerate(winner_positions):
            binwidth = 0.01

            row, col = i // cols, i % cols
            axes[row, col].hist(x, bins=np.arange(0, 1 + binwidth, binwidth))
            axes[row, col].set_title(f'Iteration {i}')
            axes[row, col].set_xlim(0, 1)

        plt.savefig(f'{outdir}/h-{h}-k-{k}-plurality-iterative-positions.pdf', bbox_inches='tight')
        plt.close()


def summarize_iterative_positioning(h, run):
    outdir = f'{RESULTS_DIR}/iterative-cand-position/summaries'
    os.makedirs(outdir, exist_ok=True)

    binwidth = 0.01
    bins = np.arange(0, 1 + binwidth, binwidth)

    for k in range(2, 11):
        winner_positions = np.load(f'{RESULTS_DIR}/iterative-cand-position/h-{h}-k-{k}-50000-trials-1000-iterations-epsilon-0-grid-start-cand-pos-run-{run}.npy')

        histograms = []
        for iteration in range(len(winner_positions)):
            hist, edges = np.histogram(winner_positions[iteration], bins=bins)
            histograms.append(hist)

        outfile = f'{outdir}/h-{h}-k-{k}-50000-trials-1000-iterations-epsilon-0-grid-start-cand-pos-run-{run}-histogram.pickle'
        print('Saving', outfile)
        with open(outfile, 'wb') as f:
            pickle.dump((histograms, edges), f)


def plot_iterative_positioning_from_summaries(h, run):

    indir = f'{RESULTS_DIR}/iterative-cand-position/summaries'
    outdir = 'plots/iterative-cand-position'
    os.makedirs(outdir, exist_ok=True)

    for k in range(2, 11):
        with open(f'{indir}/h-{h}-k-{k}-100000-trials-40-iterations-epsilon-0.01-cand-pos-run-{run}-histogram.pickle', 'rb') as f:
            winner_positions = pickle.load(f)

        rows = 5
        cols = 9
        fig, axes = plt.subplots(rows, cols, figsize=(20, 10))

        for i, (hist, edges) in enumerate(winner_positions):

            row, col = i // cols, i % cols
            axes[row, col].stairs(hist, edges)
            # axes[row, col].set_title(f'Iteration {i}')
            axes[row, col].set_xlim(0, 1)
            axes[row, col].set_xticklabels([])
            axes[row, col].set_yticklabels([])

        plt.suptitle(f'k = {k}, run {run}, {"plurality" if h == 1 else "IRV"}')
        # plt.show()
        # plt.close()
        # exit()
        #
        plt.savefig(f'{outdir}/h-{h}-k-{k}-iterative-positions-epsilon-0.01-run-{run}.pdf', bbox_inches='tight')
        plt.close()
        #


def plot_iterative_positioning_density(h):

    indir = f'{RESULTS_DIR}/iterative-cand-position/summaries'
    outdir = 'plots/iterative-cand-position/heatmaps'
    os.makedirs(outdir, exist_ok=True)

    for k in range(2, 11):
        fig, axes = plt.subplots(10, 1, figsize=(20, 20))

        for run in range(10):
            with open(f'{indir}/h-{h}-k-{k}-50000-trials-1000-iterations-epsilon-0.01-cand-pos-run-{run}-histogram.pickle', 'rb') as f:
                hists, edges = pickle.load(f)

            hists = (1 + np.array(hists).T) / 50000
            axes[run].set_xlabel('Generation')
            axes[run].set_ylabel('Position')

            im = axes[run].imshow(np.log(hists), cmap='inferno')

        # fig.colorbar(im, label='log prob.')
        # plt.show()
        plt.savefig(f'{outdir}/h-{h}-k-{k}-50000-trials-1000-iterations-epsilon-0.01-cand-pos-heatmap.pdf', bbox_inches='tight')
        plt.close()
        # exit()


def plot_iterative_positioning_density_grid_start(h):

    indir = f'{RESULTS_DIR}/iterative-cand-position/summaries'
    outdir = 'plots/iterative-cand-position/heatmaps'
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(9, 1, figsize=(20, 20))

    for k in range(2, 11):
        row = k-2

        with open(f'{indir}/h-{h}-k-{k}-50000-trials-1000-iterations-epsilon-0-grid-start-cand-pos-run-0-histogram.pickle', 'rb') as f:
            hists, edges = pickle.load(f)

        hists = (1 + np.array(hists).T) / 50000

        # axes[row].set_xlabel('Generation')
        # axes[row].set_ylabel('Position')

        im = axes[row].imshow(np.log(hists), cmap='inferno')
        axes[row].set_title(f'k = ${k}$')

    # fig.colorbar(im, label='log prob.')
    # plt.show()
    plt.savefig(f'{outdir}/h-{h}-all-k-50000-trials-1000-iterations-epsilon-0-grid-start-cand-pos-heatmap.pdf', bbox_inches='tight')
    plt.close()
    # exit()


def make_preflib_table():
    # Skip duplicate elections with tons of write-ins
    to_skip = ['ED-00018-00000001.soi', 'ED-00018-00000003.soi']

    descs = {'sl': 'San Leandro, CA', 'pierce': 'Pierce County, WA', 'irish': 'Dublin, Ireland', 'sf': 'San Francisco, CA',
             'takomapark': 'Takoma Park, WA', 'uklabor': 'UK Labour Party', 'aspen': 'Aspen, CO', 'berkley': 'Berkeley, CA',
             'burlington': 'Burlington, VT', 'debian': 'Debian Project', 'ers': 'Anonymous organizations',
             'minneapolis': 'Minneapolis, MN', 'glasgow': 'Glasgow, Scotland', 'apa': 'American Psychological Association',
             'oakland': 'Oakland, CA'}

    with open(f'results/preflib-resampling/all-resampling-results.pickle', 'rb') as f:
        elections, resampled_results, true_results = pickle.load(f)

    collec_counts = defaultdict(int)
    collec_hs = {c: [] for c in descs}
    collec_ns = {c: [] for c in descs}
    collec_ks = {c: [] for c in descs}

    skipped_vote_frac = []
    skipped_counts = []

    for collection, election_name, ballots, ballot_counts, cand_names, skipped_votes in elections:
        if election_name in to_skip:
            continue

        collec_counts[collection] += 1

        collec_hs[collection].append(max(len(b) for b in ballots))

        collec_ns[collection].append(sum(ballot_counts))
        collec_ks[collection].append(len(cand_names))

        skipped_vote_frac.append(skipped_votes / (skipped_votes + sum(ballot_counts)))
        skipped_counts.append(skipped_votes)

        if collec_hs[collection][-1] > collec_ks[collection][-1]:
            print(election_name)
            print(cand_names)
            print(max(ballots, key=lambda x: len(x)))
            print()

        stripped_election_name = election_name.replace('.soi', '').replace('.toi', '')

    for collec in sorted(collec_counts):
        min_h, max_h = min(collec_hs[collec]), max(collec_hs[collec])
        min_n, max_n = min(collec_ns[collec]), max(collec_ns[collec])
        min_k, max_k = min(collec_ks[collec]), max(collec_ks[collec])

        h_string = min_h if min_h == max_h else f'{min_h}--{max_h}'
        n_string = min_n if min_n == max_n else f'{min_n}--{max_n}'
        k_string = min_k if min_k == max_k else f'{min_k}--{max_k}'

        print(f'\\texttt{{{collec}}} & {descs[collec]} & {collec_counts[collec]} & '
              f'{k_string} & {h_string} & {n_string} \\\\')

    total_votes = sum(x for c in collec_ns for x in collec_ns[c])
    total_skipped = sum(skipped_counts)
    print(total_skipped / (total_votes + total_skipped))


def print_lp_constructions():
    for file in sorted(glob.glob('results/lp-constructions/*.pickle')):
        with open(file, 'rb') as f:
            res, x_rounded, ballots, counts = pickle.load(f)

        print(file, sum(x_rounded), len(ballots))


def plot_winner_variances():
    f_name = 'results/1d-truncation-results-20-max-1000000-trials-cand-pos.pickle'

    # f_name = 'results/1d-truncation-results-40-max-10000-trials-cand-pos.pickle'

    with open(f_name, 'rb') as f:
        all_winners, all_cands, agree_frac, max_k, n_trials = pickle.load(f)

    outdir = 'plots'
    os.makedirs(outdir, exist_ok=True)

    ks = np.arange(3, max_k+1)

    plurality_vars = []
    irv_vars = []

    for k in ks:
        winner_positions = all_cands[k][np.arange(n_trials)[:, None], all_winners[k].astype(int)]

        irv_winners = winner_positions[:, -1]
        plurality_winners = winner_positions[:, 0]

        irv_vars.append(np.var(irv_winners, ddof=1))
        plurality_vars.append(np.var(plurality_winners, ddof=1))

    plurality_vars = np.array(plurality_vars)
    irv_vars = np.array(irv_vars)

    # plurality_var_std_errs = plurality_vars * np.sqrt(2 / (n_trials - 1))
    # irv_var_std_errs = irv_vars * np.sqrt(2 / (n_trials - 1))


    # plt.fill_between(ks, plurality_vars - plurality_var_std_errs, plurality_vars + plurality_var_std_errs)
    plt.plot(ks, plurality_vars, label='plurality', marker='.')

    # plt.fill_between(ks, irv_vars - irv_var_std_errs, irv_vars + irv_var_std_errs)
    plt.plot(ks, irv_vars, label='IRV', marker='.')

    odd_ks = np.arange(3, max_k+1, 2)
    kappas = (odd_ks - 1) / 2
    plt.plot(odd_ks, 1 / (8 * kappas + 12), label='median (odd k)', ls='dashed')

    plt.plot(ks, 1 / (2*ks**2 + 6*ks + 4), label='most moderate', ls='dashed')

    plt.plot(ks, 1 / (4 + 8 / ks), label='most extreme', ls='dashed')

    plt.xticks([5, 10, 15, 20])
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Candidates ($k$)')
    plt.ylabel('Winner position variance')
    plt.show()


def summarize_high_dim_variances(f_name):
    print('opening file')
    with open(f_name, 'rb') as f:
        k_range, d, trials, irv_winner_pos, plurality_winner_pos, most_moderate_pos, most_extreme_pos = pickle.load(f)

    print('opened')


    irv_vars = []
    plurality_vars = []
    most_moderate_vars = []
    most_extreme_vars = []

    for k in k_range:
        print('IRV Cov', np.cov(irv_winner_pos[k], rowvar=False))

        irv_vars.append(np.linalg.det(np.cov(irv_winner_pos[k], rowvar=False)))
        plurality_vars.append(np.linalg.det(np.cov(plurality_winner_pos[k], rowvar=False)))
        most_moderate_vars.append(np.linalg.det(np.cov(most_moderate_pos[k], rowvar=False)))
        most_extreme_vars.append(np.linalg.det(np.cov(most_extreme_pos[k], rowvar=False)))

        print(k)
        print(most_moderate_pos[k][:10], most_extreme_pos[k][:10])
        print(most_moderate_vars[-1], most_extreme_vars[-1])

        print('\n\n\n\n')
    with open(f_name.replace('positions', 'variances'), 'wb') as f:
        pickle.dump((k_range, d, trials, irv_vars, plurality_vars, most_moderate_vars, most_extreme_vars), f)


def plot_high_dim_winner_variances(f_name):

    # f_name = 'results/1d-truncation-results-40-max-10000-trials-cand-pos.pickle'

    print('opening file')
    with open(f_name, 'rb') as f:
        k_range, d, trials, irv_vars, plurality_vars, most_moderate_vars, most_extreme_vars = pickle.load(f)

    print('opened')
    print(most_moderate_vars)
    print(most_extreme_vars)

    outdir = 'plots'
    os.makedirs(outdir, exist_ok=True)

    plt.plot(k_range, plurality_vars, label='plurality', marker='.')
    plt.plot(k_range, irv_vars, label='IRV', marker='.')
    plt.plot(k_range, most_moderate_vars, label='most moderate', ls='dashed', marker='.')
    plt.plot(k_range, most_extreme_vars, label='most extreme', ls='dashed', marker='.')

    plt.yscale('log')
    plt.ylabel('Winner position generalized variance')
    plt.xlabel('Candidates ($k$)')
    plt.legend()
    plt.show()


def plot_multiple_dim_variances():
    f_name = 'results/1d-truncation-results-20-max-1000000-trials-cand-pos.pickle'
    # f_name = 'results/1d-truncation-results-40-max-10000-trials-cand-pos.pickle'

    colors = ["#f8cf6e", "#3c9cff", "#ee504e", "#01b06a", "#540105"]

    with open(f_name, 'rb') as f:
        all_winners, all_cands, agree_frac, max_k, n_trials = pickle.load(f)

    outdir = 'plots'
    os.makedirs(outdir, exist_ok=True)

    ks = np.arange(3, max_k+1)

    plurality_vars = []
    irv_vars = []

    for k in ks:
        winner_positions = all_cands[k][np.arange(n_trials)[:, None], all_winners[k].astype(int)]

        irv_winners = winner_positions[:, -1]
        plurality_winners = winner_positions[:, 0]

        irv_vars.append(np.var(irv_winners, ddof=1))
        plurality_vars.append(np.var(plurality_winners, ddof=1))

        print(k, irv_vars[-1], plurality_vars[-1])


    plurality_vars = np.array(plurality_vars)
    irv_vars = np.array(irv_vars)

    # plurality_var_std_errs = plurality_vars * np.sqrt(2 / (n_trials - 1))
    # irv_var_std_errs = irv_vars * np.sqrt(2 / (n_trials - 1))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # plt.fill_between(ks, plurality_vars - plurality_var_std_errs, plurality_vars + plurality_var_std_errs)
    axes[0].plot(ks, plurality_vars, label='plurality', marker='.', c=colors[0])
    axes[0].plot([2, 3], [1 / (2*2**2 + 6*2 + 4), plurality_vars[0]], c=colors[0])

    # plt.fill_between(ks, irv_vars - irv_var_std_errs, irv_vars + irv_var_std_errs)
    axes[0].plot(ks, irv_vars, label='IRV', marker='.', c=colors[1])
    axes[0].plot([2, 3], [1 / (2*2**2 + 6*2 + 4), irv_vars[0]], c=colors[1])

    ks = np.arange(2, max_k + 1)
    odd_ks = np.arange(3, max_k+1, 2)
    kappas = (odd_ks - 1) / 2
    axes[0].plot(odd_ks, 1 / (8 * kappas + 12), label='median (odd k)', ls='dashed', c=colors[2])

    axes[0].plot(ks, 1 / (2*ks**2 + 6*ks + 4), label='most moderate', ls='dashed', c=colors[3])

    axes[0].plot(ks, 1 / (4 + 8 / ks), label='most extreme', ls='dashed', c=colors[4])

    axes[0].set_title('1 dimension')
    axes[0].set_xticks([5, 10, 15, 20])
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Candidates ($k$)')

    for i, dim in enumerate((2, 3)):
        with open(f'results/{dim}-dim-uniform-winner-variances-1000000-trials-max-k-20.pickle', 'rb') as f:
            k_range, d, trials, irv_vars, plurality_vars, most_moderate_vars, most_extreme_vars = pickle.load(f)

        col = i+1
        axes[col].set_title(f'{dim} dimensions')

        axes[col].plot(k_range, plurality_vars, label='plurality', marker='.', c=colors[0])
        axes[col].plot(k_range, irv_vars, label='IRV', marker='.', c=colors[1])
        axes[col].plot(k_range, most_moderate_vars, label='most moderate', ls='dashed', marker='.', c=colors[3])
        axes[col].plot(k_range, most_extreme_vars, label='most extreme', ls='dashed', marker='.', c=colors[4])

        axes[col].set_yscale('log')
        axes[col].set_xlabel('Candidates ($k$)')

    axes[0].set_ylabel('Winner position (generalized) variance')
    axes[0].legend()

    plt.show()


def plot_multiple_dsn_winner_variances(k_string='k-2-20'):

    names = f'beta-0.2-0.2-{k_string}', f'beta-0.5-0.5-{k_string}',  f'uniform-{k_string}', f'beta-5-5-{k_string}'

    colors = ["#f8cf6e", "#3c9cff", "#ee504e", "#01b06a", "#540105"]
    systems = ['plurality', 'irv', 'most moderate', 'most extreme', 'median']
    plurality_idx, irv_idx, moderate_idx, extreme_idx, median_idx = range(5)

    all_winners = {name: None for name in names}

    fig, axes = plt.subplots(1, 4, figsize=(15, 3), sharey='row')
    for col, name in enumerate(names):
        with open(f'results/{name}-dsn-winner-positions-100000-trials.pickle', 'rb') as f:
            ks, dsn, trials, all_results = pickle.load(f)

        plurality_winners = {k: [all_results[k, trial][plurality_idx] for trial in range(trials)] for k in ks}
        irv_winners = {k: [all_results[k, trial][irv_idx] for trial in range(trials)] for k in ks}
        moderate_winners = {k: [all_results[k, trial][moderate_idx] for trial in range(trials)] for k in ks}
        extreme_winners = {k: [all_results[k, trial][extreme_idx] for trial in range(trials)] for k in ks}
        median_winners = {k: [all_results[k, trial][median_idx] for trial in range(trials)] for k in ks}

        all_winners[name] = [np.array([winners[k] for k in ks]) for winners in (plurality_winners, irv_winners, moderate_winners, extreme_winners, median_winners)]

        var_fn = partial(np.var, ddof=1)
        rng = np.random.default_rng(seed=0)

        for i, system in enumerate(systems):
            system_vars = np.array([var_fn(all_winners[name][i][k_idx]) for k_idx in range(len(ks))])

            if system == 'median':
                if k_string != 'k-high':
                    odd_k = [k for k in ks if k % 2 == 1]
                    odd_vars = np.array([var for idx, var in enumerate(system_vars) if ks[idx] % 2 == 1])

                    axes[col].plot(odd_k, odd_vars, label=system, marker='.', color=colors[i])

                    cis = [bootstrap((all_winners[name][i][idx],), var_fn, confidence_level=0.99, random_state=rng).confidence_interval for idx in range(len(ks)) if ks[idx] % 2 == 1]

                    axes[col].fill_between(odd_k, [ci.low for ci in cis], [ci.high for ci in cis], color=colors[i], alpha=0.5)

            else:
                axes[col].plot(ks, system_vars, label=system, marker='.', color=colors[i])

                cis = [bootstrap((all_winners[name][i][idx],), var_fn, confidence_level=0.99,
                                 random_state=rng).confidence_interval for idx in range(len(ks)) if ks[idx] % 2 == 1]

                axes[col].fill_between(ks, [ci.low for ci in cis], [ci.high for ci in cis], color=colors[i], alpha=0.5)

        axes[col].set_title(name)
        axes[col].set_yscale('log')
        axes[col].set_xlabel('Candidates ($k$)')

    axes[0].set_ylabel('Winner position variance')
    axes[0].legend()

    plt.savefig(f'plots/moderation/different-dsn-winner-variances-{k_string}.pdf', bbox_inches='tight')
    plt.show()

    plt.close()

    plt.figure(figsize=(4, 3))
    for name in names:
        plt.plot(ks, variances[name][0] / variances[name][1], label=name, marker='.')

    plt.axhline(1, color='black', ls='dashed')
    plt.legend()
    plt.ylabel('Plurality-IRV variance ratio')
    plt.xlabel('Candidates ($k$)')
    plt.savefig(f'plots/moderation/different-dsn-winner-variance-ratio-{k_string}.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


def bootstrap_vars_and_ratios(k_string='k-2-20'):
    names = f'beta-0.2-0.2-{k_string}', f'beta-0.5-0.5-{k_string}',  f'uniform-{k_string}', f'beta-5-5-{k_string}'
    systems = ['plurality', 'irv', 'most moderate', 'most extreme', 'median']

    plurality_idx, irv_idx, moderate_idx, extreme_idx, median_idx = range(5)

    all_winners = {name: None for name in names}

    rng = np.random.default_rng(seed=0)

    var_bootstraps = {(name, system): [] for name in names for system in systems}
    var_ratio_bootstraps = {name: [] for name in names}

    for col, name in enumerate(names):
        with open(f'results/{name}-dsn-winner-positions-100000-trials.pickle', 'rb') as f:
            ks, dsn, trials, all_results = pickle.load(f)

        plurality_winners = {k: [all_results[k, trial][plurality_idx] for trial in range(trials)] for k in ks}
        irv_winners = {k: [all_results[k, trial][irv_idx] for trial in range(trials)] for k in ks}
        moderate_winners = {k: [all_results[k, trial][moderate_idx] for trial in range(trials)] for k in ks}
        extreme_winners = {k: [all_results[k, trial][extreme_idx] for trial in range(trials)] for k in ks}
        median_winners = {k: [all_results[k, trial][median_idx] for trial in range(trials)] for k in ks}

        all_winners[name] = np.array([[winners[k] for k in ks] for winners in
                                      (plurality_winners, irv_winners, moderate_winners, extreme_winners, median_winners)])

        var_fn = partial(np.var, ddof=1)
        def variance_ratio(sample1, sample2, axis):
            return np.var(sample1, ddof=1, axis=axis) / np.var(sample2, ddof=1, axis=axis)

        for j, k in enumerate(ks):
            bs = bootstrap([all_winners[name][plurality_idx, j], all_winners[name][irv_idx, j]], variance_ratio,
                           confidence_level=0.95, random_state=rng, batch=256, method='basic', paired=True)
            var_ratio_bootstraps[name].append((bs.confidence_interval.low, bs.confidence_interval.high, bs.standard_error))
            print(name, k, var_ratio_bootstraps[name][-1])

        for i, system in enumerate(systems):
            for j, k in enumerate(ks):

                bs = bootstrap([all_winners[name][i, j]], var_fn, confidence_level=0.95, random_state=rng, batch=256)
                var_bootstraps[name, system].append((bs.confidence_interval.low, bs.confidence_interval.high, bs.standard_error))
                print(name, system, k, var_bootstraps[name, system][-1])

    with open(f'results/bootstraps-{k_string}.pickle', 'wb') as f:
        pickle.dump((var_bootstraps, var_ratio_bootstraps), f)


def plot_irv_plurality_scatter():

    k_strings = ('k-2-20', 'k-high')

    ks = [3, 4, 5, 10, 20, 100, 1000]
    trials = 100000

    systems = ['plurality', 'irv', 'most moderate', 'most extreme', 'median']
    plurality_idx, irv_idx, moderate_idx, extreme_idx, median_idx = range(5)

    all_results = dict()
    for k_string in k_strings:

        with open(f'results/uniform-{k_string}-dsn-winner-positions-{trials}-trials.pickle', 'rb') as f:
            _, _, _, results = pickle.load(f)
            all_results.update(results)

    plurality_winners = {k: np.array([all_results[k, trial][plurality_idx] for trial in range(trials)]) for k in ks}
    irv_winners = {k: np.array([all_results[k, trial][irv_idx] for trial in range(trials)]) for k in ks}


    fig, axes = plt.subplots(1, len(ks), figsize=(18, 2.5), sharey='row', sharex='col')
    for col, k in enumerate(ks):

        plurality_extremity = np.abs(0.5 - plurality_winners[k])
        irv_extremity = np.abs(0.5 - irv_winners[k])

        print(len(np.unique(plurality_winners[k])), len(np.unique(irv_winners[k])))

        equal_idx = plurality_winners[k] == irv_winners[k]
        left_idx = (plurality_winners[k] < 0.5) & (plurality_extremity > irv_extremity)
        right_idx = (plurality_winners[k] > 0.5) & (plurality_extremity > irv_extremity)
        bottom_idx = (irv_winners[k] < 0.5) & (plurality_extremity < irv_extremity)
        top_idx = (irv_winners[k] > 0.5) & (plurality_extremity < irv_extremity)

        colors = ['green', 'blue', 'blue', 'red', 'red']
        text_pos = [(0.9, 0.9), (0.1, 0.5), (0.9, 0.5), (0.5, 0.1), (0.5, 0.9)]
        for i, idx in enumerate([equal_idx, left_idx, right_idx, bottom_idx, top_idx]):
            print(len(plurality_winners[k][idx]))
            axes[col].scatter(plurality_winners[k][idx], irv_winners[k][idx], s=1, color=colors[i], alpha=0.1)
            axes[col].text(text_pos[i][0], text_pos[i][1], f'{np.count_nonzero(idx) / trials:.2f}', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.5))

        axes[col].set_xlim(0, 1)
        axes[col].set_ylim(0, 1)
        axes[col].set_title(f'$k$ = {k}')
        axes[col].set_xlabel('plurality winner')
        axes[col].set_xticks([0, 0.25, 0.5, 0.75, 1])
        axes[col].set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])

    axes[0].set_ylabel('IRV winner')


    plt.subplots_adjust(wspace=0.1)

    # axes[-1, 3].axis('off')
    plt.savefig(f'plots/moderation/uniform-irv-vs-plurality-scatter.png', dpi=400, bbox_inches='tight')
    plt.show()
    plt.close()



def plot_winner_interval(k):

    all_irv_winners = dict()
    all_plurality_winners = dict()

    for file in glob.glob(f'results/beta-sweep-k-{k}-beta-0-4-100000-trials/*.pickle'):
        with open(file, 'rb') as f:
            ks, dsn, trials, all_results = pickle.load(f)

        # cand_pos[plurality_winner], cand_pos[irv_winner], cand_pos[most_moderate], cand_pos[most_extreme], median
        irv_winners = np.array([all_results[ks[0], trial][1] for trial in range(trials)])
        plurality_winners = np.array([all_results[ks[0], trial][0] for trial in range(trials)])

        all_irv_winners[dsn.args[0]] = irv_winners
        all_plurality_winners[dsn.args[0]] = plurality_winners

    alphas = list(all_irv_winners.keys())
    subset_alphas = [a for a in alphas if a <= 4]

    # centrist_alphas = sorted([alpha for alpha in subset_alphas if alpha >= 1])
    # polarized_alphas = sorted([alpha for alpha in subset_alphas if alpha <= 1])

    centrist_alphas = sorted([alpha for alpha in alphas if alpha >= 1])
    polarized_alphas = sorted([alpha for alpha in alphas if alpha <= 1])

    for winners, name in zip((all_irv_winners, all_plurality_winners), ('IRV', 'plurality')):
        ls = 'solid' if name == 'irv' else 'dashed'

        # plt.figure(figsize=(3, 5))
        # parts = plt.violinplot([winners[alpha] for alpha in centrist_alphas], centrist_alphas, widths=0.04, bw_method=0.15, vert=False)
        #
        # together_polarized_alphas = [alpha for alpha in polarized_alphas if alpha > 1/2]
        # parts = plt.violinplot([winners[alpha] for alpha in together_polarized_alphas], together_polarized_alphas, widths=0.04, bw_method=0.15, vert=False)
        #
        # split_polarized_alphas = [alpha for alpha in polarized_alphas if alpha <= 1/2]
        # parts = plt.violinplot([winners[alpha] for alpha in split_polarized_alphas], split_polarized_alphas, widths=0.04, bw_method=0.15, vert=False)

        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        axes[1].plot(centrist_alphas, [stats.beta(alpha, alpha).ppf(1/6) for alpha in centrist_alphas], ls=ls, color='green')
        axes[1].plot(centrist_alphas, [1 - stats.beta(alpha, alpha).ppf(1/6) for alpha in centrist_alphas], ls=ls, color='green')

        parts = axes[1].violinplot([winners[alpha] for alpha in centrist_alphas], centrist_alphas, widths=0.4, bw_method=0.2)
        for pc in parts['bodies']:
            pc.set_facecolor('red')
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        for part_name in ('cbars', 'cmins', 'cmaxes'):
            parts[part_name].set_edgecolor('black')


        # for alpha in centrist_alphas:
        #     axes[1].scatter(np.array([alpha] * trials) + np.random.randn(trials) / 10, winners[alpha], marker='.', color='red', s=1, alpha=0.005)

        c_polarized = np.array([2*(stats.beta(alpha, alpha).ppf(1/3) - 1/4) for alpha in polarized_alphas])

        c2_polarized = np.array([2*(stats.beta(alpha, alpha).ppf(1/3)) for alpha in polarized_alphas if alpha <= 0.5])

        axes[0].plot(polarized_alphas, c_polarized, ls=ls,  color='blue')
        axes[0].plot([alpha for alpha in polarized_alphas if alpha <= 0.5], c2_polarized, ls=ls, color='orange')
        axes[0].plot(polarized_alphas, 1-c_polarized, ls=ls, color='blue')
        axes[0].plot([alpha for alpha in polarized_alphas if alpha <= 0.5], 1-c2_polarized, ls=ls, color='orange')

        # for alpha in polarized_alphas:
        #     axes[0].scatter(np.array([alpha] * trials) + np.random.randn(trials) / 100, winners[alpha], marker='.', color='red', s=1, alpha=0.005)

        together_polarized_alphas = [alpha for alpha in polarized_alphas if alpha > 1/2]
        parts = axes[0].violinplot([winners[alpha] for alpha in together_polarized_alphas], together_polarized_alphas, widths=0.04, bw_method=0.2)
        for pc in parts['bodies']:
            pc.set_facecolor('red')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        for part_name in ('cbars', 'cmins', 'cmaxes'):
            parts[part_name].set_edgecolor('black')

        split_polarized_alphas = [alpha for alpha in polarized_alphas if  alpha <= 1/2]

        parts = axes[0].violinplot([[x for x in winners[alpha] if x >= 1/2] for alpha in split_polarized_alphas], split_polarized_alphas, widths=0.04, bw_method=0.2)
        for pc in parts['bodies']:
            pc.set_facecolor('red')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        for part_name in ('cbars', 'cmins', 'cmaxes'):
            parts[part_name].set_edgecolor('black')

        for alpha in split_polarized_alphas:
            print()
            print(alpha, len([x for x in winners[alpha] if x <= 1/2]), len([x for x in winners[alpha] if x > 1/2]), winners[alpha])

        parts = axes[0].violinplot([[x for x in winners[alpha] if x <= 1/2] for alpha in split_polarized_alphas], split_polarized_alphas, widths=0.04, bw_method=0.2)
        for pc in parts['bodies']:
            pc.set_facecolor('red')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        for part_name in ('cbars', 'cmins', 'cmaxes'):
            parts[part_name].set_edgecolor('black')

        axes[0].set_ylabel('Winner position')
        axes[0].set_ylim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[0].set_xlim(0, 1)
        axes[1].set_xlim(1, 10)

        axes[1].set_yticks([])
        axes[1].set_xticks([2, 4, 6, 8, 10])


        axes[0].set_xlabel(r'$\alpha = \beta$')
        axes[1].set_xlabel(r'$\alpha = \beta$')

        axes[0].set_title(r'$\alpha \leq 1$, ' + name)
        axes[1].set_title(r'$\alpha \geq 1$, ' + name)


        # fig, axes = plt.subplots(1, 2, sharey='row', figsize=(12, 2.5))
        # axes[1].plot(centrist_alphas, [stats.beta(alpha, alpha).ppf(1/6) for alpha in centrist_alphas], ls=ls, color='green')
        #
        # parts = axes[1].violinplot([winners[alpha] for alpha in centrist_alphas], centrist_alphas, widths=0.4, bw_method=0.2)
        # for pc in parts['bodies']:
        #     pc.set_facecolor('red')
        #     pc.set_edgecolor('black')
        #     pc.set_alpha(1)
        #
        # for part_name in ('cbars', 'cmins', 'cmaxes'):
        #     parts[part_name].set_edgecolor('black')
        #
        #
        # # for alpha in centrist_alphas:
        # #     axes[1].scatter(np.array([alpha] * trials) + np.random.randn(trials) / 10, winners[alpha], marker='.', color='red', s=1, alpha=0.005)
        #
        # c_polarized = [2*(stats.beta(alpha, alpha).ppf(1/3) - 1/4) for alpha in polarized_alphas]
        #
        # c2_polarized = [2*(stats.beta(alpha, alpha).ppf(1/3)) for alpha in polarized_alphas if alpha <= 0.5]
        #
        # axes[0].plot(polarized_alphas, c_polarized, ls=ls,  color='blue')
        # axes[0].plot([alpha for alpha in polarized_alphas if alpha <= 0.5], c2_polarized, ls=ls, color='orange')
        #
        # # for alpha in polarized_alphas:
        # #     axes[0].scatter(np.array([alpha] * trials) + np.random.randn(trials) / 100, winners[alpha], marker='.', color='red', s=1, alpha=0.005)
        #
        # together_polarized_alphas = [alpha for alpha in polarized_alphas if alpha > 1/2]
        # parts = axes[0].violinplot([winners[alpha] for alpha in together_polarized_alphas], together_polarized_alphas, widths=0.04, bw_method=0.2)
        # for pc in parts['bodies']:
        #     pc.set_facecolor('red')
        #     pc.set_edgecolor('black')
        #     pc.set_alpha(1)
        # for part_name in ('cbars', 'cmins', 'cmaxes'):
        #     parts[part_name].set_edgecolor('black')
        #
        # split_polarized_alphas = [alpha for alpha in polarized_alphas if  alpha <= 1/2]
        #
        # parts = axes[0].violinplot([[x for x in winners[alpha] if x >= 1/2] for alpha in split_polarized_alphas], split_polarized_alphas, widths=0.04, bw_method=0.2)
        # for pc in parts['bodies']:
        #     pc.set_facecolor('red')
        #     pc.set_edgecolor('black')
        #     pc.set_alpha(1)
        # for part_name in ('cbars', 'cmins', 'cmaxes'):
        #     parts[part_name].set_edgecolor('black')
        #
        # for alpha in split_polarized_alphas:
        #     print()
        #     print(alpha, len([x for x in winners[alpha] if x <= 1/2]), len([x for x in winners[alpha] if x > 1/2]), winners[alpha])
        #
        # parts = axes[0].violinplot([[x for x in winners[alpha] if x <= 1/2] for alpha in split_polarized_alphas], split_polarized_alphas, widths=0.04, bw_method=0.2)
        # for pc in parts['bodies']:
        #     pc.set_facecolor('red')
        #     pc.set_edgecolor('black')
        #     pc.set_alpha(1)
        # for part_name in ('cbars', 'cmins', 'cmaxes'):
        #     parts[part_name].set_edgecolor('black')
        #
        # axes[0].set_ylabel('Winner position')
        # axes[0].set_ylim(0, 1)
        # axes[1].set_ylim(0, 1)
        #
        # axes[0].set_xlabel(r'$\alpha = \beta$')
        # axes[1].set_xlabel(r'$\alpha = \beta$')
        #
        # axes[0].set_title(r'$\alpha \leq 1$, ' + name)
        # axes[1].set_title(r'$\alpha \geq 1$, ' + name)
        plt.subplots_adjust(wspace=0)

        plt.savefig(f'plots/moderation/beta-c-bound-k-{k}-{name.lower()}.pdf', bbox_inches='tight')


        plt.show()
        plt.close()

    # for alpha in np.linspace(1, 11, 10):
    #     dsn = stats.beta(alpha, alpha)
    #     print(alpha, dsn.ppf(1/6))



def plot_winner_interval_single_pane():

    k = 20
    all_irv_winners = dict()
    all_plurality_winners = dict()

    for file in glob.glob(f'results/beta-sweep-k-30-beta-0-4-100000-trials/*.pickle'):
        with open(file, 'rb') as f:
            ks, dsn, trials, all_results = pickle.load(f)

        # cand_pos[plurality_winner], cand_pos[irv_winner], cand_pos[most_moderate], cand_pos[most_extreme], median
        irv_winners = np.array([all_results[ks[0], trial][1] for trial in range(trials)])
        plurality_winners = np.array([all_results[ks[0], trial][0] for trial in range(trials)])

        all_irv_winners[dsn.args[0]] = irv_winners
        all_plurality_winners[dsn.args[0]] = plurality_winners

    alphas = list(all_irv_winners.keys())


    # centrist_alphas = sorted([alpha for alpha in subset_alphas if alpha >= 1])
    # polarized_alphas = sorted([alpha for alpha in subset_alphas if alpha <= 1])

    centrist_alphas = sorted([alpha for alpha in alphas if 1 < alpha <= 2])
    polarized_alphas = sorted([alpha for alpha in alphas if alpha <= 1])

    for winners, name, color, line_alpha in zip((all_irv_winners, all_plurality_winners), ('IRV', 'Plurality'), ('green', 'blue'), (1, 0.5)):

        # plt.figure(figsize=(3, 5))
        # parts = plt.violinplot([winners[alpha] for alpha in centrist_alphas], centrist_alphas, widths=0.04, bw_method=0.15, vert=False)
        #
        # together_polarized_alphas = [alpha for alpha in polarized_alphas if alpha > 1/2]
        # parts = plt.violinplot([winners[alpha] for alpha in together_polarized_alphas], together_polarized_alphas, widths=0.04, bw_method=0.15, vert=False)
        #
        # split_polarized_alphas = [alpha for alpha in polarized_alphas if alpha <= 1/2]
        # parts = plt.violinplot([winners[alpha] for alpha in split_polarized_alphas], split_polarized_alphas, widths=0.04, bw_method=0.15, vert=False)

        plt.figure(figsize=(10, 3))
        plt.plot([1] + centrist_alphas, [stats.beta(alpha, alpha).ppf(1/6) for alpha in [1] + centrist_alphas], ls='dashed', color='black', alpha=line_alpha)
        plt.plot([1] + centrist_alphas, [1 - stats.beta(alpha, alpha).ppf(1/6) for alpha in [1] + centrist_alphas], ls='dashed', color='black', alpha=line_alpha)

        parts = plt.violinplot([winners[alpha] for alpha in centrist_alphas], centrist_alphas, widths=0.04, bw_method=0.2)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)

            pc.set_alpha(0.5)



        for part_name in ('cbars', 'cmins', 'cmaxes'):
            parts[part_name].set_edgecolor(color)


        # for alpha in centrist_alphas:
        #     axes[1].scatter(np.array([alpha] * trials) + np.random.randn(trials) / 10, winners[alpha], marker='.', color='red', s=1, alpha=0.005)

        c_polarized = np.array([2*(stats.beta(alpha, alpha).ppf(1/3) - 1/4) for alpha in polarized_alphas])

        c2_polarized = np.array([2*(stats.beta(alpha, alpha).ppf(1/3)) for alpha in polarized_alphas if alpha <= 0.5])

        plt.plot(polarized_alphas, c_polarized, ls='dashed', color='black', alpha=line_alpha)
        plt.plot([alpha for alpha in polarized_alphas if alpha <= 0.5], c2_polarized, ls='dashed', color='black', alpha=line_alpha)
        plt.plot(polarized_alphas, 1-c_polarized, ls='dashed', color='black', alpha=line_alpha)
        plt.plot([alpha for alpha in polarized_alphas if alpha <= 0.5], 1-c2_polarized, ls='dashed', color='black', alpha=line_alpha)

        # for alpha in polarized_alphas:
        #     axes[0].scatter(np.array([alpha] * trials) + np.random.randn(trials) / 100, winners[alpha], marker='.', color='red', s=1, alpha=0.005)

        together_polarized_alphas = [alpha for alpha in polarized_alphas if alpha > 1/2]
        parts = plt.violinplot([winners[alpha] for alpha in together_polarized_alphas], together_polarized_alphas, widths=0.04, bw_method=0.2)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)

            pc.set_alpha(0.5)
        for part_name in ('cbars', 'cmins', 'cmaxes'):
            parts[part_name].set_edgecolor(color)

        split_polarized_alphas = [alpha for alpha in polarized_alphas if alpha <= 1/2]

        parts = plt.violinplot([[x for x in winners[alpha] if x >= 1/2] for alpha in split_polarized_alphas], split_polarized_alphas, widths=0.04, bw_method=0.2)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)

            pc.set_alpha(0.5)
        for part_name in ('cbars', 'cmins', 'cmaxes'):
            parts[part_name].set_edgecolor(color)

        for alpha in split_polarized_alphas:
            print()
            print(alpha, len([x for x in winners[alpha] if x <= 1/2]), len([x for x in winners[alpha] if x > 1/2]), winners[alpha])

        parts = plt.violinplot([[x for x in winners[alpha] if x <= 1/2] for alpha in split_polarized_alphas], split_polarized_alphas, widths=0.04, bw_method=0.2)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)

            pc.set_alpha(0.5)
        for part_name in ('cbars', 'cmins', 'cmaxes'):
            parts[part_name].set_edgecolor(color)

        plt.ylabel('Winner position')
        plt.ylim(0, 1)
        plt.xlim(0, 2)

        # plt.xticks([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])


        plt.xlabel(r'$\alpha = \beta$')

        plt.title(name)


        plt.savefig(f'plots/moderation/beta-c-bound-single-pane-{name.lower()}.pdf', bbox_inches='tight')


        plt.show()
        plt.close()

    # for alpha in np.linspace(1, 11, 10):
    #     dsn = stats.beta(alpha, alpha)
    #     print(alpha, dsn.ppf(1/6))



def plot_plurality_dsns(k_string='k-2-20'):
    name = f'uniform-{k_string}'

    colors = ["#f8cf6e", "#3c9cff", "#ee504e", "#01b06a", "#540105"]
    systems = ['plurality', 'irv', 'most moderate', 'most extreme', 'median']
    plurality_idx, irv_idx, moderate_idx, extreme_idx, median_idx = range(5)

    with open(f'results/{name}-dsn-winner-positions-100000-trials.pickle', 'rb') as f:
        ks, dsn, trials, all_results = pickle.load(f)

    plurality_winners = {k: [all_results[k, trial][plurality_idx] for trial in range(trials)] for k in ks}
    irv_winners = {k: [all_results[k, trial][irv_idx] for trial in range(trials)] for k in ks}
    moderate_winners = {k: [all_results[k, trial][moderate_idx] for trial in range(trials)] for k in ks}
    extreme_winners = {k: [all_results[k, trial][extreme_idx] for trial in range(trials)] for k in ks}
    median_winners = {k: [all_results[k, trial][median_idx] for trial in range(trials)] for k in ks}

    out_dir = f'plots/moderation/plurality-dsns/'
    os.makedirs(out_dir, exist_ok=True)
    for k in ks:
        plt.title(f'$k={k}$')
        plt.hist(plurality_winners[k], bins=100)
        plt.savefig(f'{out_dir}/plurality-dsn-k-{k}.pdf', bbox_inches='tight')

        plt.close()

if __name__ == '__main__':
    # allow keyboard interrupt to close pyplot
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    plot_winner_interval_single_pane()
    plot_irv_plurality_scatter()
    plot_k_3_winner_distributions()
