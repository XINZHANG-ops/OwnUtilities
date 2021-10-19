import numpy as np
import matplotlib.pyplot as plt
from xin_util.Rank_top_n import find_top_n_faster
import nltk
from tqdm import tqdm


def plot_stacked_bar(
    data,
    series_labels,
    category_labels=None,
    show_values=False,
    value_format="{}",
    y_label=None,
    colors=None,
    grid=True,
    reverse=False
):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    colors          -- List of color labels
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        axes.append(
            plt.bar(ind, row_data, bottom=cum_size, label=series_labels[i], color=colors[i])
        )
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels)

    if y_label:
        plt.ylabel(y_label)

    plt.legend()

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(
                    bar.get_x() + w / 2,
                    bar.get_y() + h / 2,
                    value_format.format(h),
                    ha="center",
                    va="center"
                )


def distribution_bar(values, bars=10, width_shrink=0.99, progress=True, **kwargs):
    max_val = max(values)
    min_val = min(values)
    space = np.linspace(min_val, max_val, bars)
    bar_divide = list(nltk.bigrams(space))
    x_loc = np.array([np.mean([i, j]) for i, j in bar_divide])
    width = 2 * (bar_divide[0][1] - bar_divide[0][0]) * width_shrink

    if progress:
        f = tqdm
    else:
        f = list
    bar_heights = [0] * len(x_loc)
    for x_value in f(values):
        distances = np.abs(x_loc - x_value)
        _, sort_idx = find_top_n_faster(distances, 1, 'min', False)
        sort_idx = sort_idx[0]
        bar_heights[sort_idx] += 1
    plt.figure(figsize=kwargs.get('figsize', (12, 5)))
    plt.bar(x=x_loc, height=bar_heights, width=width)
    plt.xticks(fontsize=kwargs.get('fontsize', 20), rotation='vertical')
    plt.margins(0.01)
    plt.title("distribution plot", fontsize=kwargs.get('fontsize', 20))
    plt.ylabel('count', fontsize=kwargs.get('fontsize', 20))
    plt.show()


def demo_dist_plot():
    a = list(np.random.uniform(low=-1.0, high=1.0, size=200))
    distribution_bar(a, bars=10, width_shrink=0.8, progress=True, fontsize=13)


# Example
'''
plt.figure(figsize=(6, 4))

series_labels = ['Series 1', 'Series 2']

data = [
    [0.2, 0.3, 0.35, 0.3],
    [0.8, 0.7, 0.6, 0.5]
]

category_labels = ['Cat A', 'Cat B', 'Cat C', 'Cat D']

plot_stacked_bar(
    data, 
    series_labels, 
    category_labels=category_labels, 
    show_values=True, 
    value_format="{:.1f}",
    colors=['tab:orange', 'tab:green'],
    y_label="Quantity (units)"
)

plt.savefig('bar.png')
plt.show()
'''
