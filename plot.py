import glob
import os

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats

RESULTS_DIR = 'results/'


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

        equal_idx = plurality_winners[k] == irv_winners[k]
        left_idx = (plurality_winners[k] < 0.5) & (plurality_extremity > irv_extremity)
        right_idx = (plurality_winners[k] > 0.5) & (plurality_extremity > irv_extremity)
        bottom_idx = (irv_winners[k] < 0.5) & (plurality_extremity < irv_extremity)
        top_idx = (irv_winners[k] > 0.5) & (plurality_extremity < irv_extremity)

        colors = ['green', 'blue', 'blue', 'red', 'red']
        text_pos = [(0.9, 0.9), (0.1, 0.5), (0.9, 0.5), (0.5, 0.1), (0.5, 0.9)]
        for i, idx in enumerate([equal_idx, left_idx, right_idx, bottom_idx, top_idx]):
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

    plt.savefig(f'plots/uniform-irv-vs-plurality-scatter.png', dpi=400, bbox_inches='tight')
    plt.show()
    plt.close()



def plot_k_3_4_5_100_winner_distributions():
    f_name = 'results/uniform-small-k-1000000-trials-dsn-winner-positions-1000000-trials.pickle'

    with open(f_name, 'rb') as f:
        ks, dsn, trials, all_results = pickle.load(f)

    outdir = 'plots'
    os.makedirs(outdir, exist_ok=True)

    plurality_winner_positions = {k: np.array([all_results[k, trial][0] for trial in range(trials)]) for k in [3, 4, 5] }
    irv_winner_positions = {k: np.array([all_results[k, trial][1] for trial in range(trials)]) for k in [3, 4, 5] }

    fig, axes = plt.subplots(2, 4, figsize=(15, 5), sharey='row', sharex='col')
    p1 = lambda x: 4 * x + x**2 / 2
    p2 = lambda x: - 3 / 2 + 13 * x - 13*x**2

    p1_x = np.linspace(0, 1/3, 100)
    p2_x = np.linspace(1/3, 1/2, 100)

    axes[0, 0].plot(p1_x, p1(p1_x), lw=2, zorder=0)
    axes[0, 0].plot(p2_x, p2(p2_x), lw=2, zorder=0)
    axes[0, 0].scatter([1/3], [p1(1/3)], marker='o', color='black', zorder=1, s=5)


    i1 = lambda x: 12 * x**2
    i2 = lambda x: 1-12*x + 48*x**2
    i3 = lambda x: -5 + 36*x - 48*x**2
    i4 = lambda x: -1 + 12*x - 12*x**2

    i1_x = np.linspace(0, 1/6, 100)
    i2_x = np.linspace(1/6, 1/4, 100)
    i3_x = np.linspace(1/4, 1/3, 100)
    i4_x = np.linspace(1/3, 1/2, 100)

    axes[1, 0].plot(i1_x, i1(i1_x), lw=2, zorder=0)
    axes[1, 0].plot(i2_x, i2(i2_x), lw=2, zorder=0)
    axes[1, 0].plot(i3_x, i3(i3_x), lw=2, zorder=0)
    axes[1, 0].plot(i4_x, i4(i4_x), lw=2, zorder=0)
    
    axes[1, 0].scatter([1/6], [i1(1/6)], marker='o', color='black', zorder=1, s=5)
    axes[1, 0].scatter([1/4], [i2(1/4)], marker='o', color='black', zorder=1, s=5)
    axes[1, 0].scatter([1/3], [i3(1/3)], marker='o', color='black', zorder=1, s=5)

    axes[1, 0].set_ylabel('Density')
    axes[0, 0].set_ylabel('Density')

    for k in (3, 4, 5):
        col = k - 3
        axes[0, col].hist(plurality_winner_positions[k], bins=100, density=True, alpha=0.4, color='blue', zorder=-2)
        axes[0, col].set_title(f'Plurality, $k={k}$')

        axes[1, col].hist(irv_winner_positions[k], bins=100, density=True, alpha=0.4, color='green', zorder=-2)
        axes[1, col].set_title(f'IRV, $k={k}$')
        axes[1, col].axvline(1 / 6, ls='dashed', color='black', lw=1, zorder=0)
        axes[1, col].axvline(5 / 6, ls='dashed', color='black', lw=1, zorder=0)


    trials = 100000
    with open(f'results/uniform-k-high-dsn-winner-positions-100000-trials.pickle', 'rb') as f:
        high_ks, _, _, k_high_results = pickle.load(f)

    k_high_plurality = {k: np.array([k_high_results[k, trial][0] for trial in range(trials)]) for k in high_ks}
    k_high_irv = {k: np.array([k_high_results[k, trial][1] for trial in range(trials)]) for k in high_ks}

    k = 100

    axes[0, 3].hist(k_high_plurality[k], bins=100, density=True, alpha=0.4, color='blue', zorder=-2)
    axes[0, 3].set_title(f'Plurality, $k={k}$')

    axes[1, 3].hist(k_high_irv[k], bins=100, density=True, alpha=0.4, color='green', zorder=-2)
    axes[1, 3].set_title(f'IRV, $k={k}$')
    axes[1, 3].axvline(1 / 6, ls='dashed', color='black', lw=1, zorder=0)
    axes[1, 3].axvline(5 / 6, ls='dashed', color='black', lw=1, zorder=0)

    plt.savefig('plots/k-3-4-5-100-winner-distributions-bars.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


def plot_winner_intervals():
    all_irv_winners = dict()
    all_plurality_winners = dict()

    for file in glob.glob(f'results/beta*k-30-100000-trials-dsn-winner-positions-100000-trials.pickle'):
        with open(file, 'rb') as f:
            ks, dsn, trials, all_results = pickle.load(f)

        irv_winners = np.array([all_results[ks[0], trial][1] for trial in range(trials)])
        plurality_winners = np.array([all_results[ks[0], trial][0] for trial in range(trials)])

        all_irv_winners[dsn.args[0]] = irv_winners
        all_plurality_winners[dsn.args[0]] = plurality_winners

    alphas = list(all_irv_winners.keys())

    centrist_alphas = sorted([alpha for alpha in alphas if 1 < alpha <= 2])
    polarized_alphas = sorted([alpha for alpha in alphas if alpha <= 1])

    for winners, name, color, line_alpha in zip((all_irv_winners, all_plurality_winners), ('IRV', 'Plurality'), ('green', 'blue'), (1, 0.5)):
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

        c_polarized = np.array([2*(stats.beta(alpha, alpha).ppf(1/3) - 1/4) for alpha in polarized_alphas])

        c2_polarized = np.array([2*(stats.beta(alpha, alpha).ppf(1/3)) for alpha in polarized_alphas if alpha <= 0.5])

        plt.plot(polarized_alphas, c_polarized, ls='dashed', color='black', alpha=line_alpha)
        plt.plot([alpha for alpha in polarized_alphas if alpha <= 0.5], c2_polarized, ls='dashed', color='black', alpha=line_alpha)
        plt.plot(polarized_alphas, 1-c_polarized, ls='dashed', color='black', alpha=line_alpha)
        plt.plot([alpha for alpha in polarized_alphas if alpha <= 0.5], 1-c2_polarized, ls='dashed', color='black', alpha=line_alpha)

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
        plt.xlabel(r'$\alpha = \beta$')
        plt.title(name)

        plt.savefig(f'plots/beta-c-bound-single-pane-{name.lower()}.pdf', bbox_inches='tight')

        plt.show()
        plt.close()


if __name__ == '__main__':
    # allow keyboard interrupt to close pyplot
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    os.makedirs('plots/', exist_ok=True)

    plot_winner_intervals()
    plot_irv_plurality_scatter()
    plot_k_3_4_5_100_winner_distributions()
