import numpy as np
import copy
from values import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from matplotlib.lines import Line2D

mpl.rcParams.update({
    # 'text.usetex': True,           # Use LaTeX for all text handling
    # 'font.family': 'serif',        # Use serif font instead of sans-serif
    'font.serif': 'Times',         # Specific serif font (e.g., Times)
    'axes.labelsize': 20,          # Size of axis labels
    'axes.titlesize': 16,          # Size of title
    'font.size': 16,               # Size of general text
    'legend.fontsize': 12,         # Size of legend text
    'xtick.labelsize': 14,         # Size of x-tick labels
    'ytick.labelsize': 14,         # Size of y-tick labels
    'figure.figsize': [6.4, 4.8],  # Default figure size
    'lines.linewidth': 4,        # Width of lines
    'lines.markersize': 6,         # Size of markers
    'axes.grid': True,             # Enable grid by default
    'grid.alpha': 0.5,             # Transparency of grid
    'grid.linestyle': '--',        # Style of grid lines
})

# Function to truncate colormap
def truncate_colormap(cmap, minval=0.3, maxval=0.95, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

TEXT_TOKENS=50


def power(x, power):
    if x==0:
        return 0
    return x**power


def func_error_fit(params, inputs, normalizer=1):
    '''
    y = A/N^b * 1/T^c + d

    '''
    a, b, c, d = params
    N, T = inputs
    y = a * 1/N**b * 1/(T**c) + d

    return y

def grid_search():
    # we will create a grid of values for each parameter
    # and then we will evaluate the error for each combination
    #and then find the one which gives the least error
    a = np.linspace(0.2, 0.7, 50)
    b = np.linspace(0.0, 0.2, 100)
    c = np.linspace(0.0, 0.05, 100)
    d = np.linspace(0.0, 0.3, 10)

    grid = np.array(np.meshgrid(a, b, c, d)).T.reshape(-1, 4)
    #randomly shuffle the grid
    np.random.shuffle(grid)
    print(f"Length of grid: {len(grid)}")

    pbar = tqdm(total=len(grid))
    best_loss = 1000000
    for param in grid:
        loss = 0
        for n in all_n_values:
            for token in all_token_values:
                func_pred = func_error_fit(param, [n, token])
                downstream_err = 1 - scores[str(n)][all_token_values.index(token)]
                curr_loss = (func_pred - downstream_err)**2
                loss += curr_loss
                # print(f"n: {n}, token: {token}, func_pred: {func_pred}, downstream_err: {downstream_err}, curr_loss: {curr_loss}")
        
        if loss < best_loss:
            best_loss = loss
            best_params = param
        
        pbar.update(1)
        pbar.set_description("best loss: {} : best params : {}".format(best_loss, best_params))
        #stop the loop if 20% of the grid has been searched
        if pbar.n > 0.2 * len(grid):
            break

    return best_params, best_loss    

import sys
do_search = sys.argv[1]
if do_search == "search":
    best_params, best_loss = grid_search()
else:
    #get the best params from the last line of csv
    with open("best_params.csv") as f:
        lines = f.readlines()
        best_params = [float(val) for val in lines[-2].split(",")]
        best_loss = float(lines[-1])
    print(f"Extracted Best params: {best_params}, Best loss: {best_loss}")


def get_loss(param):
    loss = 0
    for n in all_n_values:
        for token in all_token_values:
            func_pred = func_error_fit(param, [n, token])
            avg_downstream_error = 1 - scores[str(n)][all_token_values.index(token)]
            curr_loss = (func_pred - avg_downstream_error)**2
            loss += curr_loss
    return loss

check_loss = get_loss(best_params)
print(f"Check loss: {check_loss}")

if do_search == "search":
    with open("best_params.csv", "a") as f:
        f.write(",".join([str(val) for val in best_params]) + "\n")
        f.write(str(best_loss) + "\n")

norm = Normalize(vmin=min(all_n_values), vmax=max(all_n_values))
cmap = truncate_colormap(plt.get_cmap('YlOrRd'))
sm = ScalarMappable(norm=norm, cmap=cmap)

#let us plot the estimated equation and scatter plot of the true values
#create the plot
fig = plt.figure()
n_values = np.array(all_n_values)
token_values = np.array(all_token_values)
plot_token_values = np.array(np.linspace(1, 576, 50))
for n_idx, n in enumerate(all_n_values):
    flops = [2*n*(TEXT_TOKENS+token) for token in token_values]
    avg_downstream_error = [1 - scores[str(n)][all_token_values.index(token)] for token in token_values]
    #marker size
    # plt.scatter(flops, avg_downstream_error, color=colors[n_idx], marker=markers[n_idx], s=marker_sizes[n_idx])
    color = cmap(norm(n))
    for plot_idx, flop in enumerate(flops):
        plt.scatter(flop, avg_downstream_error[plot_idx], 
                    color=color, 
                    marker=markers[n_idx], 
                    s=marker_sizes[plot_idx])        
    #estimated downstream errors 
    y = [func_error_fit(best_params, [n, token]) for token in plot_token_values]
    x = [2*n*(TEXT_TOKENS+token) for token in plot_token_values]
    plt.plot(x, y, color=color, alpha=0.9)

# Add colorbar
ax = plt.gca()
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label(r'#LLM Params ($N$)')
cbar.set_ticks([0.5, 1.8, 4, 7])
cbar.set_ticklabels(["0.5B", "1.8B", "4B", "7B"])

#add a pareto optimal line, which connects the best points
if TEXT_TOKENS == 0:
    best_points = [(0.3, 1), (0.5,1), (1.8, 1), (4, 1), (7, 1), (9, 1), (12, 1), (15, 1), (18, 1), (20, 1)]
if TEXT_TOKENS == 50:
    best_points = [(0.3, 14), (0.5,14), (1.8, 14), (4, 14), (7, 14), (9, 14), (12, 14), (15, 14), (18, 14), (20, 14)]
#connect the best points with a line, label split into 2 lines
pareto_line, = plt.plot([2*n*(TEXT_TOKENS+token) for n, token in best_points],[func_error_fit(best_params, [n, token]) for n, token in best_points], color='black', linestyle='dotted', label="Pareto\nOptimal", alpha=0.9)
# Create custom entries for Alpha and Beta
custom_lines = [
    Line2D([0], [0], color='none', linestyle='None', label=r'$\alpha = 0.077$'),
    Line2D([0], [0], color='none', linestyle='None', label=r'$\beta = 0.015$'),
    # Line2D([0], [0], color='black', linestyle='dotted', label='Pareto\nOptimal'),
]
handles = custom_lines + [pareto_line]
#reduce spacing in the legend
legend1 = plt.legend(handles = handles, title="Scaling Params", fontsize=10, title_fontsize=10, loc='upper right', bbox_to_anchor=(1.0, 1.0), handlelength=2, handletextpad=0.5)
plt.gca().add_artist(legend1)

# Create custom scatter size legend (for token numbers 1, 288, 576)
handles = [plt.scatter([], [], s=size, color='gray', label=label) for size, label in zip(marker_sizes, all_token_values)]
#add a separate legend for the scatter sizes
legend2 = plt.legend(handles=handles, title="#Tokens(V)", loc='lower left', fontsize=10, title_fontsize=10, bbox_to_anchor=(0.0, 0.0))

if TEXT_TOKENS == 0:
    plt.xlabel(r"Inference FLOP ($\mathcal{O}(NV))$)")
else:
    plt.xlabel(r"Inference FLOP ($\mathcal{O}(N(50+V)))$)")
plt.xscale("log")
#y axis range
# plt.ylim([0.42, 0.68])
plt.ylabel("Avg. Downstream Error")


#make legend compact
plt.title(f"Scaling #Params vs #Visual Tokens")
#tight layout
plt.tight_layout()
#save the plot
plt.savefig(f"plots/scaling_plot_text{TEXT_TOKENS}.png")
plt.savefig(f"plots/scaling_plot_text{TEXT_TOKENS}.pdf")