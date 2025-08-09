"""
This file contains plotting utilities for visualizing model predictions and performance.

The file defines constants for figure sizes, layouts, colors, and ATLAS experiment info text.
It also implements a Plotter class with methods for creating various diagnostic plots.

Constants:
    figs_hist1d (tuple): Figure size for 1D histograms (width, height)
    figs_hist2d (tuple): Figure size for 2D histograms
    figs_hist_twice (tuple): Figure size for double-width plots
    figs_ratio (tuple): Figure size for ratio plots
    rect_hist1d (tuple): Plot margins for 1D histograms (left, bottom, right, top)
    rect_hist2d (tuple): Plot margins for 2D histograms
    rect_ratio (tuple): Plot margins for ratio plots
    subs (tuple): Minor tick locations for log scales
    num_bins (int): Default number of histogram bins
    dup_last (function): Helper to duplicate last array element
    
    colors (dict): Color definitions for plots
        'bk': Black
        'rd': Red
        'gn': Green
        'bl': Blue
        'yl': Yellow
        'cy': Cyan
        'db': Dark blue
        'br': Brown

    atlas_info1, atlas_info2 (str): ATLAS experiment info text templates
    
    keys_current (list): Names of input features
    log_scales (list): Whether each feature should use log scale
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import os
from matplotlib.backends.backend_pdf import PdfPages
import logging
from scipy.stats import binned_statistic

import torch

from Source.plot_utils import *
from Source.dataset import SampleEvaluation
plt.style.use('/remote/gpu03/schiller/JetCalibration/Source/plotting.mplstyle')


# Figure sizes and layout parameters
figs_hist1d = (1.40*2.953, 1.20*2.568)  # Width, height for 1D histograms
figs_hist2d = (1.40*2.953, 1.20*2.568)  # Width, height for 2D histograms  
figs_hist_twice = (1.40*2.953*2, 1.20*2.568)  # Double width for side-by-side plots
figs_ratio  = (1.40*2.953, 1.40*2.568)  # Width, height for ratio plots

# Plot margins (left, bottom, right, top)
rect_hist1d = (0.11, 0.11, 0.98, 0.97)  # For 1D histograms
rect_hist2d = (0.11, 0.11, 0.87, 0.97)  # For 2D histograms
rect_ratio  = (0.11, 0.09, 0.98, 0.97)  # For ratio plots

# Other plotting parameters
subs = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # Minor tick locations for log scales
num_bins = 100  # Default number of histogram bins
dup_last = lambda a: np.append(a, a[-1])  # Helper to duplicate last array element

# Color definitions
c_bk = '#0a0a0a'  # black      RGB(10,  10,  10)
c_rd = '#ff0000'  # red        RGB(255, 0, 0)
c_gn = '#00CC00'  # green      RGB(0, 204, 0)
#c_gn = '#3BC14A'  # green      RGB(0, 204, 0)
c_bl = '#0000ff'  # blue       RGB(0, 0, 255)
c_yl = '#ffff00'  # yellow     RGB(255, 255, 0)
c_cy = '#00ffff'  # cyan       RGB(0, 255, 255)
c_db = '#006699'  # dark blue  RGB(0, 102, 153)
c_br = '#663300'  # brown      RGB(102, 51, 0)

# Color lookup dictionary
colors = {'bk': c_bk, 'rd': c_rd, 'gn': c_gn, 'bl': c_bl, 
          'yl': c_yl, 'cy': c_cy, 'db': c_db, 'br': c_br}

# ATLAS experiment info text templates
atlas_info1 = ('\\textbf{\\textit{ATLAS}} Simulation Internal\n'
               '{\\footnotesize $\sqrt{s}=13\,\\text{TeV}$ anti-$k_{\\text{T}}$ $R=0.4$ EMTopo jets}\n'
               '{\\footnotesize $p_{\\text{T,jet}}^{\\text{JES}}>20\,\\text{GeV}$, }'
               '{\\footnotesize $\\vert y_{\\text{jet}}^{\\text{JES}}\\vert<2$, }'
               '{\\footnotesize $E_{\\text{clus}}^{\\text{dep}}>300\,\\text{MeV}$}')

atlas_info2 = ('\\textbf{\\textit{ATLAS}} Simulation Internal\n'
               '{\\footnotesize $\sqrt{s}=13\,\\text{TeV}$ anti-$k_{\\text{T}}$ $R=0.4$ EMTopo jets}\n'
               '{\\footnotesize $p_{\\text{T,jet}}^{\\text{JES}}>20\,\\text{GeV}$, }'
               '{\\footnotesize $\\vert y_{\\text{jet}}^{\\text{JES}}\\vert<2$,}\n'
               '{\\footnotesize $E_{\\text{clus}}^{\\text{dep}}>300\,\\text{MeV}$}')

# Input feature names and whether they should use log scale
# keys_current = ['E', 'mass', 'rap', 'groomMRatio', 'Width', 'Split12', 'Split23', 'C2', 'D2', 'Tau32', 'Tau21', 'Qw', 'EMFracCaloBased', 'EM3FracCaloBased', 'Tile0FracCaloBased', 'EffNClustsCaloBased', 'NeutralEFrac', 'ChargePTFrac', 'ChargeMFrac', 'averageMu', 'NPV']
keys_current = [
    'E_{\\text{reco}}','m_{\\text{reco}}', '\\eta_{\\text{reco}}', '\\text{groomMRatio}_{\\text{reco}}', '\\text{Width}_{\\text{reco}}', 
    '\\text{Split12}_{\\text{reco}}', '\\text{Split23}_{\\text{reco}}', '\\text{C2}_{\\text{reco}}', '\\text{D2}_{\\text{reco}}', '\\text{Tau32}_{\\text{reco}}',
    '\\text{Tau21}_{\\text{reco}}', '\\text{Qw}_{\\text{reco}}', '\\text{EMFracCaloBased}_{\\text{reco}}', '\\text{EM3FracCaloBased}_{\\text{reco}}',
    '\\text{Tile0FracCaloBased}_{\\text{reco}}', '\\text{EffNClustsCaloBased}_{\\text{reco}}', '\\text{NeutralEFrac}_{\\text{reco}}',
    '\\text{ChargePTFrac}_{\\text{reco}}', '\\text{ChargeMFrac}_{\\text{reco}}', '\\text{averageMu}_{\\text{reco}}', '\\text{NPV}_{\\text{reco}}'
]
log_scales = [True, True, False, False, False, True, True, False, True, False, False, True, False, False, False, False, False, False, False, False, False]


"""
Plotting class and utility functions for visualizing model predictions and performance.

The Plotter class handles creating various plots for analyzing model predictions:
- Loss history plots
- Input feature histograms
- Response (R) prediction histograms and 2D comparisons
- Energy/Mass prediction histograms and 2D comparisons 
- Prediction uncertainty/standard deviation plots
- GMM component weight distributions

Key plotting utility functions:
- make_hist_1dim_ratio: Creates 1D histogram with ratio panel
- make_hist_2dim: Creates 2D histogram/heatmap

The plots help visualize:
- Model prediction accuracy vs ground truth
- Prediction uncertainties and distributions
- Input feature correlations
- Training convergence
"""

# def get_fillbetween_values(values):
#     return np.asanyarray([val for val in values[1:-1] for _ in range(2)])
    
# def get_fillbetween_edges(edges):
#     return np.asarray([edges[0]] + [edg for edg in edges[1:-1] for _ in range(2)] + [edges[-1]])

class Plotter():
    """
    Class for creating various analysis plots of model predictions.
    
    Args:
        plot_dir (str): Directory to save plots
        params (dict): Model parameters
        test_predictions (dict): Model predictions on test set
        test_inputs (tensor): Input features for test set
        test_targets (tensor): Target values for test set
        samples (tensor): Samples drawn from predicted distributions
        log_likelihoods (tensor): Log likelihoods of samples
        variable (str): Target variable being predicted (E or M)
    """
    def __init__(
            self,
            plot_dir,
            params,
            test_predictions,
            test_inputs,
            test_targets,
            samples: SampleEvaluation,
            variable,
            target_dim,
            train_inputs=None,
            train_targets=None,
            additional_test_inputs=None,
            skip_dims=[]
        ):
        self.plot_dir = plot_dir
        self.params = params
        self.test_predictions = test_predictions
        self.test_inputs = test_inputs.numpy()
        self.test_targets = test_targets.numpy()

        self.samples = samples

        self.n_samples = self.samples.log_response_samples.shape[-1]

        self.variable = variable
        self.target_dim = target_dim

        self.train_inputs = train_inputs
        self.train_targets = train_targets
        self.plot_train = train_inputs != None and train_targets != None

        self.additional_test_inputs = additional_test_inputs

        self.keys_current = [keys_current[i] for i in range(len(keys_current)) if i not in skip_dims]
        self.log_scales = [log_scales[i] for i in range(len(log_scales)) if i not in skip_dims]

        logging.info(f"Plotter has been created for {self.variable}")
        logging.info(f"Test input: {self.test_inputs.shape[1]}, additional test input: {self.additional_test_inputs.shape[1]}")

        assert len(self.keys_current) == self.test_inputs.shape[1]
        assert len(self.log_scales) == self.test_inputs.shape[1]

    @torch.inference_mode()
    def plot_loss_history(self, name, train_loss, val_loss):
        """Plot training and validation loss curves."""
        plt.plot(train_loss, label="Training loss")
        plt.plot(val_loss, label="Validation loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss history")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, name), format='pdf', bbox_inches="tight")
        plt.close()

    @torch.inference_mode()
    def plot_inputs_histogram(self, name):
        """Create histograms of input features."""
        with PdfPages(os.path.join(self.plot_dir, name)) as pdf:
            for feature in range(self.test_inputs.shape[1]):
                input_data = self.test_inputs[:, feature]
                xmin = np.quantile(input_data, 0.0001)
                xmax = np.quantile(input_data, 0.9999)
                xbin = 100

                if log_scales[feature] and False:
                    xedges = np.logspace(np.log10(xmin), np.log10(xmax), xbin+1)
                else: 
                    xedges = np.linspace(xmin, xmax, xbin+1)

                fig, ax = plt.subplots()
                ax.hist(input_data, bins=xedges, density=True, histtype='step')
                ax.set_xlabel(keys_current[feature])
                ax.set_ylabel('Density')
                ax.set_title(keys_current[feature])
                if log_scales[feature] and False:
                    ax.set_xscale('log')
                plt.tight_layout()
                pdf.savefig()
                plt.close()
            
    @torch.inference_mode()
    def r_predictions_histogram(self, name):
        """Create histograms comparing true vs predicted response values."""

        logging.info("Plotting r predictions")

        logR_truth = self.test_targets

        logR_pred_MC = self.samples.log_response_samples[:, self.target_dim]
        logR_pred_mean = self.samples.log_response_mean[:, self.target_dim]
        logR_pred_max_likelihood = self.samples.log_response_mode[:, self.target_dim]

        logr_range = compute_range([logR_truth, logR_pred_MC], quantile=0.00001)

        data = [logR_truth, logR_pred_MC, logR_pred_mean, logR_pred_max_likelihood]
        # labels = [
        #     '$R_{\\text{true}}$',
        #     '$R_{\\text{pred}}$ MC',
        #     '$R_{\\text{pred}}$ Mean',
        #     '$R_{\\text{pred}}$ Mode',
        # ]
        labels = [
            'True',
            'Sampling',
            'Mean',
            'Mode',
        ]
        colors_fig = [colors['bk'], colors['bl'], colors['rd'], colors['gn']]
        showratios = [False, True, True, True]
        ratio_line_styles = ["-", "-", "-", "-"]
        if self.plot_train:
            logR_train = self.train_targets.squeeze()
            data.insert(1, logR_train)
            # labels.append('$R_{\\text{train}}$')
            labels.insert(1, 'Train')
            colors_fig.insert(1, colors['cy'])
            showratios.insert(1, True)
            ratio_line_styles.insert(1, "--")

        fig1, axs1 =  make_hist_1dim_ratio(
            data       = data,
            labels     = labels,
            colors     = colors_fig,
            showratios = showratios,
            ratioref   = 0,
            # xlabel     = 'Log Response $\\log_{10}(r_{\\text{%s}})$' % self.variable,
            xlabel     = '$\\log_{10} r_{\\text{%s}}$' % self.variable,
            rlabel     = '$Pred/True$',
            xrange     = logr_range,
            ticks      = [[], [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]],
            logscales  = [False, True],
            # nbins      = 100,
            nbins      = 50,
            # legend     = ['lower center', 0.65, 0.05, None] if self.variable == "E" else ['upper right', 0.95, 0.95, None],
            legend     = ['lower center', 0.55, 0.05, None] if self.variable == "E" else ['lower center', 0.55, 0.05, None],
            atlas_info = [0.04, 0.95, 'four-lines', 'left', 'top', 'none'],
            ratio_line_styles=ratio_line_styles
        )

        r_truth = 10**self.test_targets.squeeze()
        r_pred_MC = self.samples.response_samples[:, self.target_dim]
        r_pred_mean = self.samples.response_mean[:, self.target_dim]
        r_pred_max_likelihood = self.samples.response_mode[:, self.target_dim]

        r_range = compute_range([r_truth, r_pred_MC[:, 0]], quantile=0.00001)

        data = [r_truth, r_pred_MC, r_pred_mean, r_pred_max_likelihood]
        if self.plot_train:
            r_train = 10.**self.train_targets.squeeze()
            data.insert(1, r_train)

        fig2, axs2 =  make_hist_1dim_ratio(
            data       = data,
            labels     = labels,
            colors     = colors_fig,
            showratios = showratios,
            ratioref   = 0,
            # xlabel     = 'Response $r_{\\text{%s}}$' % self.variable,
            xlabel     = '$r_{\\text{%s}}$' % self.variable,
            rlabel     = '$Pred/True$',
            xrange     = r_range,
            ticks      = [[], [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]],
            logscales  = [False, True],
            # nbins      = 100,
            nbins      = 50,
            # legend     = ['lower center', 0.65, 0.05, None] if self.variable == "E" else ['upper right', 0.95, 0.95, None],
            legend     = ['upper right', 0.95, 0.95, None],
            atlas_info = [0.04, 0.95, 'four-lines', 'left', 'top', 'none'],
            ratio_line_styles = ratio_line_styles
        )

        with PdfPages(os.path.join(self.plot_dir, name)) as pdf:
            pdf.savefig(fig1)
            pdf.savefig(fig2)
        plt.close(fig1)
        plt.close(fig2)

    @torch.inference_mode()
    def E_M_predictions_histogram(self, name):
        """Create histograms comparing true vs predicted energy/mass values."""

        logging.info("Plotting E_M predictions")

        r_truth = 10**self.test_targets.squeeze()
        r_pred_MC = self.samples.response_samples[:, self.target_dim]
        r_pred_mean = self.samples.response_mean[:, self.target_dim]
        r_pred_mode = self.samples.response_mode[:, self.target_dim]
        r_pred_jetmean = self.samples.response_jet_mean[:, self.target_dim]
        r_pred_jetmode = self.samples.response_jet_mode[:, self.target_dim] 

        if self.plot_train:
            r_train = 10.**self.train_targets.squeeze()

        if self.variable == "E":
            ins = self.test_inputs[:, 0]
            truth = (1./r_truth) * ins

            pred_MC = (1./r_pred_MC) * ins[:, np.newaxis]

            pred_mean = (1./r_pred_mean) * ins
            pred_mode = (1./r_pred_mode) * ins

            pred_jetmean = (1./r_pred_jetmean) * ins
            pred_jetmode = (1./r_pred_jetmode) * ins

            if self.plot_train:
                train_ins = self.train_inputs[:, 0]
                train = (1./r_train) * train_ins

            # labels = ['$E_{\\text{true}}$',
            #           '$E_{\\text{pred}}$ MC',
            #           '$E_{\\text{pred}}$ Mean',
            #           '$E_{\\text{pred}}$ Mode',
            #           ]
        else:
            ins = self.test_inputs[:, 1]
            truth = (1./r_truth) * self.test_inputs[:, 1]
            
            pred_MC = (1./r_pred_MC) * ins[:, np.newaxis]

            pred_mean = (1./r_pred_mean) * ins
            pred_mode = (1./r_pred_mode) * ins

            pred_jetmean = (1./r_pred_jetmean) * ins
            pred_jetmode = (1./r_pred_jetmode) * ins

            if self.plot_train:
                train_ins = self.train_inputs[:, 1]
                train = (1./r_train) * train_ins

            
            # labels = ['$m_{\\text{true}}$',
            #           '$m_{\\text{pred}}$ MC',
            #           '$m_{\\text{pred}}$ Mean',
            #           '$m_{\\text{pred}}$ Mode',
            #           ]

        labels = [
            'True',
            'Sampling',
            'Mean',
            'Mode',
        ]
        data = [truth, pred_MC, pred_mean, pred_mode]
        colors_fig = [colors['bk'], colors['bl'], colors['rd'], colors['gn']]
        showratios=[False, True, True, True]
        ratio_line_styles = ["-", "-", "-", "-"]
        if self.plot_train:
            data.insert(1, train)
            colors_fig.insert(1, colors['cy'])
            showratios.insert(1, True)
            labels.insert(1, 'Train')
            ratio_line_styles.insert(1, "--")

        range = [max(np.min(truth)-100, 1), np.quantile(truth, 0.99999)]
        logging.info(f"{self.variable} range: {range}")

        fig1, axs =  make_hist_1dim_ratio(
            data       = data,
            labels     = labels,
            colors     = colors_fig,
            showratios = showratios,
            ratioref   = 0,
            xlabel     = "Jet Energy $E$ [GeV]" if self.variable == "E" else "Jet mass $m$ [GeV]",
            rlabel     = "$Pred/True$",
            xrange     = range,
            ticks      = [[], [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]] if self.variable != "E" else [[], []],
            logscales  = [False, self.variable != "E"],
            # nbins      = 100,
            nbins      = 50,
            # legend     = ['lower center', 0.65, 0.05, None] if self.variable == "E" else ['lower right', 0., 0., None],
            legend     = ['upper right', 0.95, 0.95, None] if self.variable == "E" else ['lower left', 0.2, 0.05, None],
            atlas_info = [0.04, 0.95, 'four-lines', 'left', 'top', 'none'],
            ratio_line_styles=ratio_line_styles,
            ratio_ylims=(0.9, 1.1),
            ratio_yticks=[0.9, 0.95, 1.0, 1.05, 1.1]
        )

        labels2 = [
            'True',
            'Sampling',
            'Jet Mean',
            'Jet Mode',
        ]

        data2 = [truth, pred_MC, pred_jetmean, pred_jetmode]
        if self.plot_train:
            data2.insert(1, train)
            labels2.insert(1, 'Train')

        fig2, axs =  make_hist_1dim_ratio(
            data       = data2,
            labels     = labels2,
            colors     = colors_fig,
            showratios = showratios,
            ratioref   = 0,
            xlabel     = "Jet Energy $E$ [GeV]" if self.variable == "E" else "Jet mass $m$ [GeV]",
            rlabel     = "$Pred/True$",
            xrange     = range,
            ticks      = [[], [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]] if self.variable != "E" else [[], []],
            logscales  = [False, self.variable != "E"],
            # nbins      = 100,
            nbins      = 50,
            legend     = ['upper right', 0.95, 0.95, None] if self.variable == "E" else ['lower left', 0.2, 0.05, None],
            atlas_info = [0.04, 0.95, 'four-lines', 'left', 'top', 'none'],
            ratio_line_styles=ratio_line_styles,
            ratio_ylims=(0.9, 1.1),
            ratio_yticks=[0.9, 0.95, 1.0, 1.05, 1.1]
        )

        range = [max(0.9 * np.min(truth), 1), np.quantile(truth, 0.99999)]
        logging.info(f"{self.variable} range: {range}")

        fig3, axs2 =  make_hist_1dim_ratio(
            data       = data,
            labels     = labels,
            colors     = colors_fig,
            showratios = showratios,
            ratioref   = 0,
            xlabel     = "Jet Energy $E$ [GeV]" if self.variable == "E" else "Jet mass $m$ [GeV]",
            rlabel     = '$Pred/True$',
            xrange     = range,
            ticks      = [[], [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]] if self.variable != "E" else [[], []],
            logscales  = [True, self.variable != "E"],
            # nbins      = 100,
            nbins      = 50,
            legend     = ['lower center', 0.65, 0.05, None] if self.variable == "E" else ['lower left', 0.2, 0.05, None],
            atlas_info = [0.04, 0.95, 'four-lines', 'left', 'top', 'none'],
            ratio_line_styles = ratio_line_styles,
            ratio_ylims=(0.9, 1.1),
            ratio_yticks=[0.9, 0.95, 1.0, 1.05, 1.1]
        )

        fig4, axs2 =  make_hist_1dim_ratio(
            data       = data2,
            labels     = labels2,
            colors     = colors_fig,
            showratios = showratios,
            ratioref   = 0,
            xlabel     = "Jet Energy $E$ [GeV]" if self.variable == "E" else "Jet mass $m$ [GeV]",
            rlabel     = '$Pred/True$',
            xrange     = range,
            ticks      = [[], [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]] if self.variable != "E" else [[], []],
            logscales  = [True, self.variable != "E"],
            # nbins      = 100,
            nbins      = 50,
            legend     = ['lower center', 0.65, 0.05, None] if self.variable == "E" else ['lower left', 0.2, 0.05, None],
            atlas_info = [0.04, 0.95, 'four-lines', 'left', 'top', 'none'],
            ratio_line_styles = ratio_line_styles,
            ratio_ylims=(0.9, 1.1),
            ratio_yticks=[0.9, 0.95, 1.0, 1.05, 1.1]
        )

        with PdfPages(os.path.join(self.plot_dir, name)) as pdf:
            pdf.savefig(fig1)
            pdf.savefig(fig2)
            pdf.savefig(fig3)
            pdf.savefig(fig4)
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)

    @torch.inference_mode()
    def E_M_predictions_input_histogram(self, name):
        logging.info("Plotting E_M predictions input")

        r_truth = 10**self.test_targets
        
        if self.variable == "E":
            input_data = self.test_inputs[:, 0]
        else:
            input_data = self.test_inputs[:, 1]

        jet_truth = input_data / r_truth
        jet_MC = (1. / self.samples.response_samples[:, self.target_dim]) * input_data[:, None]
        jet_mean = input_data / self.samples.response_mean[:, self.target_dim]
        jet_mode = input_data / self.samples.response_mode[:, self.target_dim]

        # assert np.isnan(jet_truth).any() == False, "Jet truth contais NaN"
        # assert np.isnan(jet_MC).any() == False, "Jet MC contais NaN"
        # assert np.isnan(jet_mean).any() == False, "Jet mean contais NaN"
        # assert np.isnan(jet_mode).any() == False, "Jet mode contais NaN"

        full_features = np.concatenate([self.test_inputs, self.additional_test_inputs], axis=1)
        
        # _keys_current = [f"\\text{{ {key} }}" for key in self.keys_current] + ["p_T", "\\eta"]
        _keys_current = [key for key in self.keys_current] + ["p_T", "\\eta"]
        _log_scales = self.log_scales + [True, False]

        with PdfPages(os.path.join(self.plot_dir, name)) as pdf:
            for feature in range(full_features.shape[1]):
                logging.info(f"Plotting {_keys_current[feature]}")

                input_data = full_features[:, feature]
                # assert np.isnan(input_data).any() == False, "input_data contais NaN"

                xmin = np.quantile(input_data, 0.001)
                xmax = np.quantile(input_data, 0.999)
                xbin = 100

                if _log_scales[feature]:
                    xedges = np.logspace(np.log10(xmin), np.log10(xmax), xbin+1)
                else: 
                    xedges = np.linspace(xmin, xmax, xbin+1)

                ymin  = np.quantile(jet_truth, 0.001)
                ymax  = np.quantile(jet_truth, 0.999)
                ybin  = 100

                if True:
                    yedges = np.logspace(np.log10(ymin), np.log10(ymax), ybin+1)
                else:
                    yedges = np.linspace(ymin, ymax, ybin+1)

                input_data_MC = np.broadcast_to(input_data[:, None], (input_data.shape[0], jet_MC.shape[1])).flatten()

                hist_target, _, _ = np.histogram2d(input_data, jet_truth, bins=[xedges, yedges], density=False)
                hist_pred_MC, _, _ = np.histogram2d(input_data_MC, jet_MC.flatten(), bins=[xedges, yedges], density=False)
                hist_mean, _, _ = np.histogram2d(input_data, jet_mean, bins=[xedges, yedges], density=False)
                hist_mode, _, _ = np.histogram2d(input_data, jet_mode, bins=[xedges, yedges], density=False)

                norm_column = True

                if norm_column:
                    norm = np.sum(hist_target, axis=1, keepdims=True)
                    norm[norm==0] = 1.
                    hist_target /= norm

                    norm = np.sum(hist_pred_MC, axis=1, keepdims=True)
                    norm[norm==0] = 1.
                    hist_pred_MC /= norm

                    norm = np.sum(hist_mean, axis=1, keepdims=True)
                    norm[norm==0] = 1.
                    hist_mean /= norm

                    norm = np.sum(hist_mode, axis=1, keepdims=True)
                    norm[norm==0] = 1.
                    hist_mode /= norm
                else:
                    hist_target /= np.sum(hist_target)
                    hist_pred_MC /= np.sum(hist_pred_MC)
                    hist_mean /= np.sum(hist_mean)
                    hist_mode /= np.sum(hist_mode)


                # assert np.isnan(hist_target).any() == False, "hist_target contais NaN"
                # assert np.isnan(hist_pred_MC).any() == False, "hist_pred_MC contais NaN"
                # assert np.isnan(hist_mean).any() == False, "hist_mean contais NaN"
                # assert np.isnan(hist_mode).any() == False, "hist_mode contais NaN"

                # weights=np.zeros_like(input_data) + 1./input_data.size
                # weights_MC=np.zeros_like(input_data_MC) + 1./input_data_MC.size

                vmin = min(
                    [
                        np.min(hist_target[np.nonzero(hist_target)]),
                        np.min(hist_pred_MC[np.nonzero(hist_pred_MC)]),
                        np.min(hist_mean[np.nonzero(hist_mean)]),
                        np.min(hist_mode[np.nonzero(hist_mode)])
                    ]
                )
                vmax = max(
                    [
                        np.max(hist_target[np.nonzero(hist_target)]),
                        np.max(hist_pred_MC[np.nonzero(hist_pred_MC)]),
                        np.max(hist_mean[np.nonzero(hist_mean)]),
                        np.max(hist_mode[np.nonzero(hist_mode)])
                    ]
                )

                fig, axs = plt.subplots(1, 4, figsize=figs_hist_twice)

                fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_hist2d)
                atlas_info = [0.04, 0.95, 'three-lines', 'left', 'top', 'white']
                atlas = atlas_info1 if atlas_info[2] == 'three-lines' else atlas_info2
                #ax.text(atlas_info[0], atlas_info[1], atlas, ha=atlas_info[3], va=atlas_info[4], transform=ax.transAxes, bbox=dict(boxstyle='round', fc=atlas_info[5], ec='none', alpha=0.75))
                
                hist = axs[0].pcolormesh(xedges, yedges, hist_target.T, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), rasterized=True)
                if True:
                    axs[0].set_ylabel(f"$\\log_{{ 10 }} \\text{{ {self.variable} }}_{{ \\text{{ reco }} }}$")
                else:
                    axs[0].set_ylabel(f"{self.variable}")
                axs[0].set_title("True")

                hist = axs[1].pcolormesh(xedges, yedges, hist_pred_MC.T, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), rasterized=True)
                axs[1].set_title("MC")

                hist = axs[2].pcolormesh(xedges, yedges, hist_mean.T, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), rasterized=True)
                axs[2].set_title("Mean")

                hist = axs[3].pcolormesh(xedges, yedges, hist_mode.T, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), rasterized=True)
                axs[3].set_title("Mode")

                for i in range(4):
                    if _log_scales[feature]:
                        axs[i].set_xlabel(f"$\\log_{{ 10 }} {_keys_current[feature]}$")
                    else:
                        axs[i].set_xlabel(f"${_keys_current[feature]}$")

                    if _log_scales[feature]:
                        axs[i].set_xscale('log')
                    axs[i].set_xlim((xmin, xmax))
                    axs[i].set_ylim((ymin, ymax))

                    if True:
                        axs[i].set_yscale('log')
                        axs[i].yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
                        axs[i].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

                    if i > 0:
                        axs[i].set_yticks([])

                divider = make_axes_locatable(axs[3])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(hist, cax=cax)
                cbar.set_label('Relative entries', loc='center')
                cbar.ax.tick_params(axis='y', which='both', direction='out')
    
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    @torch.inference_mode()
    def r_2d_histogram(self, name):
        """Create 2D histograms comparing true vs predicted response values."""

        logging.info("Plotting r_2d_histogram")

        r_truth = self.test_targets.squeeze()            

        with PdfPages(os.path.join(self.plot_dir, name)) as pdf:
            r_pred = self.samples.log_response_samples[:, self.target_dim, 0].squeeze()
            r_range = compute_range([r_truth, r_pred], quantile=0.00001)
            label = 'Model $\\log(r_%s)$ Post. MC' % self.variable
            fig, ax = make_hist_2dim(
                data       = [r_truth, r_pred],
                labels     = ['True $\\log(r_%s)$' % self.variable,
                            label],
                ranges     = [r_range, r_range],
                ticks      = [[],
                            []],
                logscales  = [False, False],
                nbins      = [100, 100],
                atlas_info = [0.96, 0.05, 'three-lines', 'right', 'bottom', 'none'])
            pdf.savefig(fig)
            plt.close(fig)
            r_pred = self.samples.log_response_mean[:, self.target_dim].squeeze()
            label = 'Model $\\log(r_%s)$ Post. Mean' % self.variable
            fig, ax = make_hist_2dim(
                data       = [r_truth, r_pred],
                labels     = ['True $\\log(r_%s)$' % self.variable,
                            label],
                ranges     = [r_range, r_range],
                ticks      = [[],
                            []],
                logscales  = [False, False],
                nbins      = [100, 100],
                atlas_info = [0.96, 0.05, 'three-lines', 'right', 'bottom', 'none'])
            pdf.savefig(fig)
            plt.close(fig)
            r_pred = self.samples.log_response_mode[:, self.target_dim]
            label = 'Model $\\log(r_%s)$ Post. Maximum' % self.variable
            fig, ax = make_hist_2dim(
                data       = [r_truth, r_pred],
                labels     = ['True $\\log(r_%s)$' % self.variable,
                            label],
                ranges     = [r_range, r_range],
                ticks      = [[],
                            []],
                logscales  = [False, False],
                nbins      = [100, 100],
                atlas_info = [0.96, 0.05, 'three-lines', 'right', 'bottom', 'none'])
            pdf.savefig(fig)
            plt.close(fig)

    @torch.inference_mode()
    def E_M_2d_histogram(self, name, mode="sample"):
        """Create 2D histograms comparing true vs predicted energy/mass values."""

        variable = self.variable

        r_truth = 10**self.test_targets.squeeze()
        
        r_pred_MC = self.samples.response_samples[:, self.target_dim, 0]
        r_pred_mean = self.samples.response_mean[:, self.target_dim]

        if variable == "E":
            ins = self.test_inputs[:, 0]
            truth = (1./r_truth) * ins
            if mode == "sample":
                pred = (1./r_pred_MC) * ins
                label = 'Model Energy $E$ MC'
            elif mode == "mean":
                pred = (1./r_pred_mean) * ins
                label = 'Model Energy $E$ Post. Mean'
            range = [np.min(truth)-10, np.quantile(truth, 0.999)]
        else:
            truth = (1./r_truth) * self.test_inputs[:, 1]
            ins = self.test_inputs[:, 1]
            if mode == "sample":
                pred = (1./r_pred_MC) * ins
                label = 'Model Mass $M$ MC'
            elif mode == "mean":
                pred = (1./r_pred_mean) * ins
                label = 'Model Mass $M$ Post. Mean'

            range = [np.min(truth)-10, np.quantile(truth, 0.999)]

        fig, ax = make_hist_2dim(
            data       = [truth, pred],
            labels     = ['True Energy $E$' if variable == "E" else 'True Mass $M$',
                        label],
            ranges     = [range, range],
            ticks      = [[],
                        []],
            logscales  = [False, False],
            nbins      = [100, 100],
            atlas_info = [0.96, 0.05, 'three-lines', 'right', 'bottom', 'none'])

        fig.savefig(os.path.join(self.plot_dir, name), format='pdf')
        plt.close(fig)

    @torch.inference_mode()
    def pred_inputs_histogram(self, name):
        """Create histograms showing correlations between inputs and predictions."""

        logging.info("Plotting prediction inputs histogram")

        r_truth = 10**self.test_targets.squeeze()
        r_pred = self.samples.response_samples[:, self.target_dim]

        full_features = np.concatenate([self.test_inputs, self.additional_test_inputs], axis=1)
        
        # _keys_current = [f"\\text{{ {key} }}" for key in self.keys_current] + ["p_T", "\\eta"]
        _keys_current = [key for key in self.keys_current] + ["p_{T, \\text{true}}", "\\eta_{\\text{true}}"]
        _log_scales = self.log_scales + [True, False]

        with PdfPages(os.path.join(self.plot_dir, name)) as pdf:
            for feature in range(full_features.shape[1]):
                input_data = full_features[:, feature]
                input_data_MC = np.broadcast_to(input_data[:, None], (input_data.shape[0], r_pred.shape[1])).flatten()

                xmin = np.quantile(input_data, 0.001)
                xmax = np.quantile(input_data, 0.999)
                xbin = 100

                if _log_scales[feature] and False:
                    xedges = np.logspace(np.log10(xmin), np.log10(xmax), xbin+1)
                else: 
                    xedges = np.linspace(xmin, xmax, xbin+1)

                ymin  = np.quantile(r_truth, 0.001)
                ymax  = np.quantile(r_truth, 0.999)
                ybin  = 100

                if False:
                    yedges = np.logspace(np.log10(ymin), np.log10(ymax), ybin+1)
                else: yedges = np.linspace(ymin, ymax, ybin+1)

                hist_target, _, _ = np.histogram2d(input_data, r_truth, bins=[xedges, yedges], density=False)
                hist_pred, _, _ = np.histogram2d(input_data_MC, r_pred.flatten(), bins=[xedges, yedges], density=False)

                vmin = min([np.min(hist_target[np.nonzero(hist_target)]/np.sum(hist_target)),
                            np.min(hist_pred[np.nonzero(hist_pred)]/np.sum(hist_pred))])
                vmax = max([np.max(hist_target[np.nonzero(hist_target)]/np.sum(hist_target)),
                            np.max(hist_pred[np.nonzero(hist_pred)]/np.sum(hist_pred))])
                del hist_target, hist_pred

                ykey  = 'r_ems' # response

                fig, axs = plt.subplots(1, 2, figsize=figs_hist_twice)

                fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_hist2d)
                atlas_info = [0.04, 0.95, 'three-lines', 'left', 'top', 'white']
                atlas = atlas_info1 if atlas_info[2] == 'three-lines' else atlas_info2
                #ax.text(atlas_info[0], atlas_info[1], atlas, ha=atlas_info[3], va=atlas_info[4], transform=ax.transAxes, bbox=dict(boxstyle='round', fc=atlas_info[5], ec='none', alpha=0.75))
                hist = axs[0].hist2d(input_data, r_truth, weights=np.zeros_like(input_data)+1./input_data.size, bins=[xedges, yedges], norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), rasterized=True)
                divider = make_axes_locatable(axs[0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(hist[3], cax=cax)
                cbar.set_label('Relative entries', loc='center')
                cbar.ax.tick_params(axis='y', which='both', direction='out')

                axs[0].set_xlabel(f"${_keys_current[feature]}$")
                axs[0].set_ylabel(f"True response $r_{self.variable}$")

                if _log_scales[feature] and False:
                    axs[0].set_xscale('log')
                axs[0].set_xlim((xmin, xmax))
                axs[0].set_ylim((ymin, ymax))

                if False:
                    axs[0].set_yscale('log')
                    axs[0].yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
                    axs[0].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

                hist = axs[1].hist2d(input_data_MC, r_pred.flatten(), weights=np.zeros_like(input_data_MC)+1./input_data_MC.size, bins=[xedges, yedges], norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), rasterized=True)
                divider = make_axes_locatable(axs[1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(hist[3], cax=cax)
                cbar.set_label('Relative entries', loc='center')
                cbar.ax.tick_params(axis='y', which='both', direction='out')

                axs[1].set_xlabel(f"${_keys_current[feature]}$")
                axs[1].set_ylabel(f"Predicted response $r_{self.variable}$")

                if _log_scales[feature] and False:
                    axs[1].set_xscale('log')
                axs[1].set_xlim((xmin, xmax))
                axs[1].set_ylim((ymin, ymax))

                if False:
                    axs[1].set_yscale('log')
                    axs[1].yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
                    axs[1].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    @torch.inference_mode()
    def pred_inputs_histogram_marginalized(self, name):
        """Create marginalized histograms showing input-prediction correlations."""

        logging.info("Plotting marginalized input-prediction histograms")

        r_truth = 10**self.test_targets.squeeze()
        r_pred = self.samples.response_samples[:, self.target_dim]

        full_features = np.concatenate([self.test_inputs, self.additional_test_inputs], axis=1)
        
        # _keys_current = [f"\\text{{ {key} }}" for key in self.keys_current] + ["p_T", "\\eta"]
        _keys_current = [key for key in self.keys_current] + ["p_{T, \\text{true}}", "\\eta_{\\text{true}}"]
        _log_scales = self.log_scales + [True, False]

        with PdfPages(os.path.join(self.plot_dir, name)) as pdf:

            # for feature in range(self.test_inputs.shape[1]):
            for feature in range(full_features.shape[1]):

                logging.info(f"Plotting marginalized plot for the feature {_keys_current[feature]}")

                # input_data = self.test_inputs[:, feature]
                input_data = full_features[:, feature]

                xmin = np.quantile(input_data, 0.001)
                xmax = np.quantile(input_data, 0.999)
                xbin = 50
                
                if _log_scales[feature]:
                    xedges = np.logspace(np.log10(xmin), np.log10(xmax), xbin+1)
                else: 
                    xedges = np.linspace(xmin, xmax, xbin+1)

                median_target, _, _ = binned_statistic(input_data, r_truth, statistic='median', bins=xedges)

                # DS: I think this is a fake uncertainty
                # median_samples = []
                # for sample in range(self.n_samples):
                #     median_sample, _, _ = binned_statistic(input_data, r_pred[:, sample], statistic='median', bins=xedges)
                #     median_samples.append(median_sample)
                # median_samples_mean = np.mean(np.array(median_samples), axis=0)
                # median_samples_std = np.std(np.array(median_samples), axis=0)
                median_samples, _, _ = binned_statistic(np.broadcast_to(input_data[..., None], r_pred.shape).flatten(), r_pred.flatten(), statistic='median', bins=xedges)

                # posterior_mean = np.mean(r_pred, axis=1)
                # median_posteriormean, _, _ = binned_statistic(input_data, posterior_mean, statistic='median', bins=xedges)
                median_posteriormean, _, _ = binned_statistic(input_data, self.samples.response_mean[:, self.target_dim], statistic='median', bins=xedges)

                # posterior_median = np.median(r_pred, axis=1)
                # median_posteriormedian, _, _ = binned_statistic(input_data, posterior_median, statistic='median', bins=xedges)
                # median_posteriormedian, _, _ = binned_statistic(input_data, self.samples.response_median, statistic='median', bins=xedges)

                median_posteriormaximum, _, _ = binned_statistic(input_data, self.samples.response_mode[:, self.target_dim], statistic='median', bins=xedges)

                bin_centers = (xedges[:-1] + xedges[1:]) / 2
                # ymin, ymax = np.min(median_target)-0.1, np.max(median_target)+0.1
                # ymin, ymax = np.min(median_target), np.max(median_target)
                # ymin, ymax = np.min(median_samples_mean - median_samples_std), np.max(median_samples_mean + median_samples_std)
                ymin, ymax = np.min(median_samples), np.max(median_samples)
                ydiff = ymax - ymin
                ymin = ymin - 0.1 * ydiff
                ymax = ymax + 0.1 * ydiff

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figs_ratio, sharex=True, gridspec_kw={'height_ratios': [3,1]})
                fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_ratio)
                fig.subplots_adjust(hspace=0.07)

                # Top panel
                # lt = ax1.step(bin_centers, median_target, color=colors['bk']) # , label=f"True $r_{self.variable}$")
                # ls = ax1.step(bin_centers, median_samples_mean, color=colors['bl']) # , label=f"$r_{self.variable}$ Sampled")
                # lmean = ax1.step(bin_centers, median_posteriormean, color=colors['rd']) # , label=f"$r_{self.variable}$ Mean")
                # # ax1.step(bin_centers, median_posteriormedian, color=colors['gn'], label=f"$r_{self.variable}$ Post. Median")
                # lmode = ax1.step(bin_centers, median_posteriormaximum, color=colors['gn']) # , label=f"$r_{self.variable}$ Mode")
                # ls_unc = ax1.fill_between(bin_centers, median_samples_mean - median_samples_std, median_samples_mean + median_samples_std, color=colors['bl'], alpha=0.4, step='pre')

                lt = ax1.stairs(median_target, edges=xedges, color=colors['bk']) # , label=f"True $r_{self.variable}$")
                # ls = ax1.stairs(median_samples_mean, edges=xedges, color=colors['bl']) # , label=f"$r_{self.variable}$ Sampled")
                ls = ax1.stairs(median_samples, edges=xedges, color=colors['bl']) # , label=f"$r_{self.variable}$ Sampled")
                lmean = ax1.stairs(median_posteriormean, edges=xedges, color=colors['rd']) # , label=f"$r_{self.variable}$ Mean")
                lmode = ax1.stairs(median_posteriormaximum, edges=xedges, color=colors['gn']) # , label=f"$r_{self.variable}$ Mode")
                # ls_unc = ax1.fill_between(
                #     get_fillbetween_edges(xedges),
                #     get_fillbetween_values(median_samples_mean - median_samples_std),
                #     get_fillbetween_values(median_samples_mean + median_samples_std),
                #     color=colors['bl'], alpha=0.4, step='pre'
                # )
                # ls_unc = ax1.fill_between(
                #     xedges,
                #     dup_last(median_samples_mean - median_samples_std),
                #     dup_last(median_samples_mean + median_samples_std),
                #     color=colors['bl'], alpha=0.4, step='post'
                # )
                
                legend_linewidth = 4
                handles = [
                    matplotlib.lines.Line2D([], [], color=lt.get_edgecolor(), linewidth=legend_linewidth),
                    # (matplotlib.lines.Line2D([], [], color=ls.get_edgecolor(), linewidth=legend_linewidth), ls_unc),
                    matplotlib.lines.Line2D([], [], color=ls.get_edgecolor(), linewidth=legend_linewidth),
                    matplotlib.lines.Line2D([], [], color=lmean.get_edgecolor(), linewidth=legend_linewidth),
                    matplotlib.lines.Line2D([], [], color=lmode.get_edgecolor(), linewidth=legend_linewidth)
                ]
                labels = [
                    f"True $r_{self.variable}$",
                    f"$r_{self.variable}$ Sampled",
                    f"$r_{self.variable}$ Mean",
                    f"$r_{self.variable}$ Mode"
                ]

                ax1.set_ylabel(f'Bin Median $r_{self.variable}$')
                ax1.set_xlim((xmin, xmax))
                ax1.set_ylim((ymin, ymax))  
                if _log_scales[feature]:
                    ax1.set_xscale('log')
                # ax1.legend(borderpad=0.4)
                # ax1.legend(loc="best")
                ax1.legend(handles=handles, labels=labels, loc="best")
                
                ax1.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))

                # Bottom ratio panel
                # ratio_samples = median_samples_mean / median_target
                ratio_samples = median_samples / median_target
                ratio_posteriormean = median_posteriormean / median_target
                # ratio_posteriormedian = median_posteriormedian / median_target
                ratio_posteriormaximum = median_posteriormaximum / median_target
                
                # ax2.step(bin_centers, ratio_samples, color=colors['bl'])
                # ax2.step(bin_centers, ratio_posteriormean, color=colors['rd'])
                # # ax2.step(bin_centers, ratio_posteriormedian, color=colors['gn'])
                # ax2.step(bin_centers, ratio_posteriormaximum, color=colors['br'])
                # ax2.fill_between(bin_centers, ratio_samples - median_samples_std / median_target, ratio_samples + median_samples_std / median_target, color=colors['bl'], alpha=0.4, step='pre')
                # ax2.axhline(1.0, color='black', linestyle='dashed')
                
                ax2.stairs(ratio_samples, edges=xedges, color=colors['bl'])
                ax2.stairs(ratio_posteriormean, edges=xedges, color=colors['rd'])
                ax2.stairs(ratio_posteriormaximum, edges=xedges, color=colors['br'])
                # ax2.fill_between(
                #     get_fillbetween_edges(bin_centers),
                #     get_fillbetween_values(ratio_samples - median_samples_std / median_target),
                #     get_fillbetween_values(ratio_samples + median_samples_std / median_target),
                #     color=colors['bl'], alpha=0.4, step='pre'
                # )
                # ax2.fill_between(
                #     xedges,
                #     dup_last(ratio_samples - median_samples_std / median_target),
                #     dup_last(ratio_samples + median_samples_std / median_target),
                #     color=colors['bl'], alpha=0.4, step='post'
                # )
                ax2.axhline(1.0, color='black', linestyle='dashed')
        
                
                # ax2.set_xlabel(keys_current[feature])
                # ax2.set_xlabel(f"${_keys_current[feature]}$")

                # if _log_scales[feature]:
                #     ax2.set_xlabel(f"$\\log_{{ 10 }} {_keys_current[feature]}$")
                # else:
                #     ax2.set_xlabel(f"${_keys_current[feature]}$")
                ax2.set_xlabel(f"${_keys_current[feature]}$")

                ax2.set_ylabel('Ratio')
                if _log_scales[feature]:
                    ax2.set_xscale('log')   
                # ax2.set_ylim(0.95, 1.05)
                # ax2.set_ylim(0.99, 1.01)

                # ratio_ymax = np.max(ratio_samples + median_samples_std / median_target)
                # ratio_ymin = np.min(ratio_samples - median_samples_std / median_target)
                ratio_ymax = np.max(ratio_samples)
                ratio_ymin = np.min(ratio_samples)
                ratio_diff = ratio_ymax - ratio_ymin
                ratio_ymax += 0.5 * ratio_diff
                ratio_ymin -= 0.5 * ratio_diff
                # ratio_diff_up = max(0.0001, ratio_ymax - 1.)
                # ratio_diff_down = max(0.0001, 1. - ratio_ymin)
                # ratio_ymax += 0.5 * ratio_diff_up
                # ratio_ymin -= 0.5 * ratio_diff_down

                ax2.set_ylim(ratio_ymin, ratio_ymax)

                ax2.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.01))

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    @torch.inference_mode()
    def plot_standard_deviations(self, name):
        """Plot distributions of prediction standard deviations."""
        r_truth = 10**self.test_targets.squeeze()
        r_pred = self.samples.response_samples[:, self.target_dim]
        r_pred_std = np.std(r_pred, axis=1)
        r_pred_mean = np.mean(r_pred, axis=1)

        with PdfPages(os.path.join(self.plot_dir, name)) as pdf:

            bins = np.linspace(np.quantile(r_pred_std, 0.0001), np.quantile(r_pred_std, 0.9999), 100)
            fig, ax = plt.subplots(1, 2, figsize=figs_hist_twice)
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_hist2d)
            ax[0].hist(r_pred_std, bins=bins, density=True, label='Posterior Std $R$')
            ax[0].set_xlabel('Standard deviation $R$')
            ax[0].set_ylabel('Density')
            #ax[0].set_yscale('log')
            ax[0].legend()

            bins = np.linspace(np.quantile(r_pred_mean, 0.0001), np.quantile(r_pred_mean, 0.9999), 100)
            ax[1].hist2d(r_pred_mean, r_pred_std, bins=[bins, bins])
            ax[1].set_xlabel('Posterior Mean $R$')
            ax[1].set_ylabel('Posterior Std $R$')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    @torch.inference_mode()
    def plot_GMM_weights(self, name):
        """Plot distributions of GMM component weights."""
        weight_key = [key for key in self.test_predictions.keys() if 'weights' in key][0]
        mu_key = [key for key in self.test_predictions.keys() if 'mu' in key][0]
        sigma_key = [key for key in self.test_predictions.keys() if 'sigma' in key][0]

        weights = self.test_predictions[weight_key].numpy()
        sorted_weights = np.sort(weights, axis=1)
        fig, ax = plt.subplots(1, 1, figsize=figs_hist2d)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_hist2d)
        bins = np.linspace(0, 1, 100)
        for i in range(weights.shape[1]):
            ax.hist(sorted_weights[:, i], bins=bins, density=True, label=f'Gaussian {i+1}', histtype='step')
        #ax.legend()
        ax.set_xlabel('Gaussian weight')
        ax.set_ylabel('Density')
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, name), format='pdf')
        plt.close(fig)
    
    @torch.inference_mode()
    def r_rel_over_input(self, name):
        """Create histograms depicting r_true / r_pred as a function of some input."""

        variable = self.variable

        logging.info("Plotting r relative over input")

        R_truth = 10.**self.test_targets
        
        R_pred_MC = self.samples.response_samples[:, self.target_dim]
        R_pred_mean = self.samples.response_mean[:, self.target_dim]
        R_pred_max_likelihood = self.samples.response_mode[:, self.target_dim]

        ratio_MC = (R_truth / R_pred_MC.T).T
        ratio_mean = R_truth / R_pred_mean
        ratio_mode = R_truth / R_pred_max_likelihood

        targets = [ratio_MC, ratio_mean, ratio_mode]
        labels = ["MC", "Mean", "Mode"]
        colors_fig = [colors['bl'], colors['rd'], colors['gn']]
        xlabels = [r'$p_T \; [\text{GeV}]$', r'$\eta$']
        # ylabel = r'$r_{\text{truth}} / r_{\text{pred}}$'
        ylabel = f'$r_{{ {variable}, \\text{{ truth}} }} / r_{{ {variable}, \\text{{ pred}} }}$'

        pT = self.additional_test_inputs[:, 0]
        eta = self.additional_test_inputs[:, 1]

        input_log_scale = [True, False]
        ratio_ylims = [
            [0.8, 1.2],
            [0.7, 1.3]
        ]
        ratio_yticks = [
            [0.8, 0.9, 1.0, 1.1, 1.2],
            [0.7, 0.85, 1.0, 1.15, 1.3]
        ]


        with PdfPages(os.path.join(self.plot_dir, name)) as pdf:
            for input, xlabel, log_scale, _ratio_ylims, _ratio_yticks in zip([pT, eta], xlabels, input_log_scale, ratio_ylims, ratio_yticks):
                r_range = compute_range([input], quantile=0.00001)

                fig, ax =  make_hist_1dim_over_input(
                    targets    = targets,
                    input      = input,
                    labels     = labels,
                    colors     = colors_fig,
                    xlabel     = xlabel,
                    ylabel     = ylabel,
                    xrange     = r_range,
                    # ticks      = [[], [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]],
                    logscales  = [log_scale, False],
                    nbins      = 100,
                    # legend     = ['lower center', 0.65, 0.05, None] if self.variable == "E" else ['upper right', 0.95, 0.95, None],
                    legend     = ['upper right', 0.95, 0.95, None],
                    atlas_info = [0.04, 0.95, 'four-lines', 'left', 'top', 'none'],
                    ratio_ylims=_ratio_ylims,
                    ratio_yticks=_ratio_yticks
                )

                pdf.savefig(fig)
                plt.close(fig)


def make_hist_1dim_over_input(targets, input, labels, colors, xlabel, ylabel, xrange, ticks=[[], []], logscales=[False, False],
                              nbins=100, legend=['upper right', 0.95, 0.95, None], atlas_info=[0.97, 0.03, 'three-lines', 'right', 'bottom', 'none'],
                              ratio_line_styles=None, ratio_ylims=(0.8, 1.2), ratio_yticks=[0.8, 0.9, 1.0, 1.1, 1.2]):
    if ratio_line_styles == None:
        ratio_line_styles = ['-' for _ in targets]

    if logscales[0]: 
        bins = np.logspace(np.log10(xrange[0]), np.log10(xrange[1]), nbins+1)
    else:            
        bins = np.linspace(         xrange[0],           xrange[1],  nbins+1)

    bin_ids = np.digitize(input, bins)
    hists = []
    hists_err = []
    for i, target_i in enumerate(targets):
        target_bins = [target_i[bin_ids == i + 1] for i in range(nbins)]
        target_bins_mean = [np.mean(target_bin) for target_bin in target_bins]
        target_bins_std = [np.std(target_bin, ddof=1) for target_bin in target_bins]
        hists.append(np.asarray(target_bins_mean))
        hists_err.append(np.asarray(target_bins_std))

    fig, ax = plt.subplots(1, 1, figsize=figs_hist1d)
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_hist1d)

    steps = []; fills = []
    for i, y_avg, y_err, color, label, ratio_line_style in zip(range(len(hists)), hists, hists_err, colors, labels, ratio_line_styles):
        step = ax.step(bins, dup_last(y_avg), alpha=1.00, linewidth=1.00, where='post', color=color, label=label, linestyle=ratio_line_style)
        ax.step(bins, dup_last(y_avg-y_err),  alpha=0.50, linewidth=0.50, where='post', color=color, linestyle=ratio_line_style)
        ax.step(bins, dup_last(y_avg+y_err),  alpha=0.50, linewidth=0.50, where='post', color=color, linestyle=ratio_line_style)
        fill = ax.fill_between(bins, dup_last(y_avg-y_err), dup_last(y_avg+y_err), alpha=0.20, step='post', facecolor=color)
        steps.append(step[0]); fills.append(fill)

    _, labels = ax.get_legend_handles_labels()
    handles = [(step, fill) for step, fill in zip(steps, fills)]
    ax.legend(handles, labels, alignment='left', title=legend[3], loc=legend[0], bbox_to_anchor=(legend[1], legend[2]))

    ax.axhline(1.50, zorder=0, linestyle='dotted', color='grey')
    ax.axhline(1.00, zorder=0, linestyle='dashed', color='black')
    ax.axhline(0.50, zorder=0, linestyle='dotted', color='grey')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, loc='center')
    ax.set_xlim((bins[0], bins[-1]))
    ax.set_ylim(ratio_ylims)
    ax.set_yticks(ratio_yticks)
    if logscales[0]:
        ax.set_xscale('log')
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if logscales[1]:
        ax.set_yscale('log')
        ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if len(ticks[0]) > 0: ax.set_xticks(ticks[0])
    if len(ticks[1]) > 0: ax.set_yticks(ticks[1])

    return fig, ax

    

def make_hist_1dim_ratio(data, labels, colors, showratios, ratioref, xlabel, rlabel, xrange, ticks=[[], []], logscales=[False, True],
                         nbins=100, integrals=[], legend=['upper right', 0.95, 0.95, None], atlas_info=[0.97, 0.03, 'three-lines', 'right', 'bottom', 'none'], switch_legend=False,
                         ratio_line_styles=None, ratio_ylims=(0.8, 1.2), ratio_yticks=[0.8, 0.9, 1.0, 1.1, 1.2]):
    if ratio_line_styles == None:
        ratio_line_styles = ['-' for _ in data]

    if logscales[0]: 
        bins = np.logspace(np.log10(xrange[0]), np.log10(xrange[1]), nbins+1)
    else:            
        bins = np.linspace(         xrange[0],           xrange[1],  nbins+1)

    hists = []
    hists_err = []
    for i, dataset in enumerate(data):
        if len(dataset.shape) == 1:
            hists.append(np.histogram(dataset, bins=bins)[0])
            hists_err.append(np.sqrt(hists[-1]))
            logging.info(f"Computing {labels[i]} uncertainty via Poisson")
        else:
            # DS: I am pretty sure that this is a fake uncertainty
            # hists_model = np.array([np.histogram(model, bins=bins)[0] for model in dataset.T])
            # hist_model_mean = np.mean(hists_model, axis=0)
            # hist_model_std = np.std(hists_model, axis=0)
            # hists.append(hist_model_mean)
            # hists_err.append(hist_model_std)
            # logging.info(f"Computing {labels[i]} uncertainty via std")
            
            _dataset = dataset.flatten()
            hists.append(np.histogram(_dataset, bins=bins)[0])
            hists_err.append(np.sqrt(hists[-1]))
            logging.info(f"Computing {labels[i]} uncertainty via Poisson")

    if len(integrals) == 0: 
        integrals =  [np.sum(y_avg) for y_avg in hists]
    scales    = [1/integral if integral != 0.0 else 1.0 for integral in integrals]

    fig, axs = plt.subplots(2, 1, figsize=figs_ratio, sharex=True, gridspec_kw={'height_ratios': [3,1]})
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_ratio); fig.subplots_adjust(hspace=0.07)
    atlas = atlas_info1 if atlas_info[2] == 'three-lines' else atlas_info2
    #axs[0].text(atlas_info[0], atlas_info[1], atlas, ha=atlas_info[3], va=atlas_info[4], transform=axs[0].transAxes, bbox=dict(boxstyle='round', fc=atlas_info[5], ec='none', alpha=0.75))

    steps = []; fills = []
    for i, y_avg, y_err, scale, color, label, ratio_line_style in zip(range(len(hists)), hists, hists_err, scales, colors, labels, ratio_line_styles):
        
        step = axs[0].step(bins, scale*dup_last(y_avg), alpha=1.00, linewidth=1.00, where='post', color=color, label=label, linestyle=ratio_line_style)
        axs[0].step(bins, scale*dup_last(y_avg-y_err),  alpha=0.50, linewidth=0.50, where='post', color=color, linestyle=ratio_line_style)
        axs[0].step(bins, scale*dup_last(y_avg+y_err),  alpha=0.50, linewidth=0.50, where='post', color=color, linestyle=ratio_line_style)
        fill = axs[0].fill_between(bins, scale*dup_last(y_avg-y_err), scale*dup_last(y_avg+y_err), alpha=0.20, step='post', facecolor=color)
        steps.append(step[0]); fills.append(fill)

        if showratios[i]:
            
            with np.errstate(all='ignore'):
                ratio_avg = get_ratio(scale*y_avg, scales[ratioref]*hists[ratioref])
                ratio_err = np.sqrt(get_ratio(y_err, y_avg)**2+get_ratio(hists_err[ratioref], hists[ratioref])**2)
            #ratio_nan = np.isnan(ratio_avg)
            #ratio_avg[ratio_nan] = 1.0
            #ratio_err[ratio_nan] = 0.0

            axs[1].step(bins, dup_last(ratio_avg),           alpha=1.00, linewidth=1.00, where='post', color=color, linestyle=ratio_line_style)
            axs[1].step(bins, dup_last(ratio_avg-ratio_err), alpha=0.50, linewidth=0.50, where='post', color=color, linestyle=ratio_line_style)
            axs[1].step(bins, dup_last(ratio_avg+ratio_err), alpha=0.50, linewidth=0.50, where='post', color=color, linestyle=ratio_line_style)
            axs[1].fill_between(bins, dup_last(ratio_avg-ratio_err), dup_last(ratio_avg+ratio_err), alpha=0.20, step='post', facecolor=color)

    if switch_legend:
        _, labels = axs[0].get_legend_handles_labels()
        handles = [(step, fill) for step, fill in zip(steps[::-1], fills[::-1])]
        axs[0].legend(handles, labels[::-1], alignment='left', title=legend[3], loc=legend[0], bbox_to_anchor=(legend[1], legend[2]))
    else:
        _, labels = axs[0].get_legend_handles_labels()
        handles = [(step, fill) for step, fill in zip(steps, fills)]
        axs[0].legend(handles, labels, alignment='left', title=legend[3], loc=legend[0], bbox_to_anchor=(legend[1], legend[2]))

    axs[1].axhline(1.50, zorder=0, linestyle='dotted', color='grey')
    axs[1].axhline(1.00, zorder=0, linestyle='dashed', color='black')
    axs[1].axhline(0.50, zorder=0, linestyle='dotted', color='grey')
    axs[1].set_xlabel(xlabel)
    # axs[0].set_ylabel('Relative entries')
    axs[0].set_ylabel('Normalized counts')
    axs[1].set_ylabel(rlabel, loc='center')
    axs[0].set_xlim((bins[0], bins[-1]))
    axs[1].set_ylim(ratio_ylims)
    axs[1].set_yticks(ratio_yticks)
    if logscales[0]:
        axs[0].set_xscale('log')
        axs[0].xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
        axs[0].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if logscales[1]:
        axs[0].set_yscale('log')
        axs[0].yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
        axs[0].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if len(ticks[0]) > 0: axs[1].set_xticks(ticks[0])
    if len(ticks[1]) > 0: axs[0].set_yticks(ticks[1])

    return fig, axs


def make_hist_2dim(data, labels, ranges, ticks=[[], []], logscales=[False, False], nbins=[100, 100],
                   atlas_info=[0.97, 0.03, 'three-lines', 'right', 'bottom', 'none'], showdiag=True, norm_columns=True, norm_rows=False, show_cbar_label=True):
    fig, ax = plt.subplots(1, 1, figsize=figs_hist2d)
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_hist2d)
    
    mesh, cax = _make_hist_2dim(
        ax=ax,
        data=data,
        labels=labels,
        ranges=ranges,
        ticks=ticks,
        logscales=logscales,
        nbins=nbins,
        atlas_info=atlas_info,
        showdiag=showdiag,
        norm_columns=norm_columns,
        norm_rows=norm_rows)
    
    cbar = fig.colorbar(mesh, cax=cax)
    if show_cbar_label:
        cbar.set_label('Relative entries', loc='center')
    cbar.ax.tick_params(axis='y', which='both', direction='out')

    return fig, ax

def _make_hist_2dim(ax, data, labels, ranges, ticks=[[], []], logscales=[False, False], nbins=[100, 100],
                   atlas_info=[0.97, 0.03, 'three-lines', 'right', 'bottom', 'none'], showdiag=True,
                   density=False, norm_columns=True, norm_rows=False, fontsize=None, vmin=None, vmax=None):

    if logscales[0]: xbins = np.logspace(np.log10(ranges[0][0]), np.log10(ranges[0][1]), nbins[0]+1)
    else:            xbins = np.linspace(         ranges[0][0],           ranges[0][1],  nbins[0]+1)
    if logscales[1]: ybins = np.logspace(np.log10(ranges[1][0]), np.log10(ranges[1][1]), nbins[0]+1)
    else:            ybins = np.linspace(         ranges[1][0],           ranges[1][1],  nbins[1]+1)

    #atlas = atlas_info1 if atlas_info[2] == 'three-lines' else atlas_info2
    #ax.text(atlas_info[0], atlas_info[1], atlas, ha=atlas_info[3], va=atlas_info[4], transform=ax.transAxes, bbox=dict(boxstyle='round', fc=atlas_info[5], ec='none', alpha=0.75))
    #hist = ax.hist2d(data[0], data[1], weights=np.zeros_like(data[0]) + 1.0 / data[0].size, bins=[xbins, ybins], norm=mpl.colors.LogNorm(), rasterized=True)

    if len(data[0].shape) == 1:
        hist, x_edges, y_edges = np.histogram2d(
            data[0],
            data[1],
            bins=[xbins, ybins],
            density=density,
            weights=np.full_like(data[0], 1.0 / data[0].size) if not density else None
        )
    else:
        hists = []
        for i in range(data[0].shape[1]):
            hist, x_edges, y_edges = np.histogram2d(
                data[0][:, i],
                data[1][:, i],
                bins=[xbins, ybins],
                density=density,
                weights=np.full_like(data[0][:, i], 1.0 / data[0][:, i].size) if not density else None
            )
            hists.append(hist)
        hist = np.array(hists).mean(axis=0)
    if norm_columns:
        hist /= hist.sum(axis=1, keepdims=True)
    elif norm_rows:
        hist /= hist.sum(axis=0, keepdims=True)

    norm_kwargs = {}
    if vmin != None:
        norm_kwargs["vmin"] = vmin
        norm_kwargs["vmax"] = vmax

    mesh = ax.pcolormesh(x_edges, y_edges, hist.T, norm=mpl.colors.LogNorm(**norm_kwargs), rasterized=True)
    if showdiag: 
        ax.plot([ranges[0][0], ranges[0][1]], [ranges[1][0], ranges[1][1]], linestyle='dashed', color='black')
    divider = make_axes_locatable(ax)
    cax  = divider.append_axes('right', size='5%', pad=0.05)
    ax.set_xlabel(labels[0], fontsize=fontsize)
    ax.set_ylabel(labels[1], fontsize=fontsize)
    ax.set_xlim((xbins[0], xbins[-1]))
    ax.set_ylim((ybins[0], ybins[-1]))
    if logscales[0]:
        ax.set_xscale('log')
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if logscales[1]:
        ax.set_yscale('log')
        ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if len(ticks[0]) > 0: ax.set_xticks(ticks[0])
    if len(ticks[1]) > 0: ax.set_yticks(ticks[1])

    return mesh, cax



def make_hist_2dim_ratio(data, data_ref, labels, ranges, ticks=[[], []], logscales=[False, False], nbins=[100, 100],
                   atlas_info=[0.97, 0.03, 'three-lines', 'right', 'bottom', 'none'], showdiag=True, sigma_max=3.):

    if logscales[0]: xbins = np.logspace(np.log10(ranges[0][0]), np.log10(ranges[0][1]), nbins[0]+1)
    else:            xbins = np.linspace(         ranges[0][0],           ranges[0][1],  nbins[0]+1)
    if logscales[1]: ybins = np.logspace(np.log10(ranges[1][0]), np.log10(ranges[1][1]), nbins[0]+1)
    else:            ybins = np.linspace(         ranges[1][0],           ranges[1][1],  nbins[1]+1)

    fig, ax = plt.subplots(1, 1, figsize=figs_hist2d)
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_hist2d)

    if len(data[0].shape) == 1:
        hist, x_edges, y_edges = np.histogram2d(data[0], data[1], bins=[xbins, ybins]) #, weights=np.full_like(data[0], 1.0 / data[0].size))
        hist_err = np.sqrt(hist)
    else:
        hists = []
        for i in range(data[0].shape[1]):
            hist, x_edges, y_edges = np.histogram2d(data[0][:, i], data[1][:, i], bins=[xbins, ybins]) #, weights=np.full_like(data[0][:, i], 1.0 / data[0][:, i].size))
            hists.append(hist)
        hists = np.array(hists)
        hist = hists.mean(axis=0)
        hist_err = hists.std(axis=0, ddof=1)

    if len(data_ref[0].shape) == 1:
        hist_ref, x_edges_ref, y_edges_ref = np.histogram2d(data_ref[0], data_ref[1], bins=[xbins, ybins]) #, weights=np.full_like(data_ref[0], 1.0 / data_ref[0].size))
        hist_ref_err = np.sqrt(hist_ref)
    else:
        hists_ref = []
        for i in range(data[0].shape[1]):
            hist_ref, x_edges, y_edges = np.histogram2d(data_ref[0][:, i], data_ref[1][:, i], bins=[xbins, ybins]) #, weights=np.full_like(data_ref[0][:, i], 1.0 / data_ref[0][:, i].size))
            hists_ref.append(hist_ref)
        hists_ref = np.array(hists_ref)
        hist_ref = hists_ref.mean(axis=0)
        hist_ref_err = hists_ref.std(axis=0, ddof=1)

    # min_nonzero_hist = min([hist_e for hist_e in hist.flatten() if hist_e != 0])
    # min_nonzero_hist_ref = min([hist_e for hist_e in hist_ref.flatten() if hist_e != 0])
    # min_nonzero = min(min_nonzero_hist, min_nonzero_hist_ref) * 1.e-3
    # logging.info(f"Plotting relative 2d hist for {labels[0]} vs {labels[1]}. Hist has min value {min_nonzero_hist} and hist_ref has min value {min_nonzero_hist_ref}")

    # hist = hist + min_nonzero
    # hist_ref = hist_ref + min_nonzero

    hist /= hist.sum()
    hist_ref /= hist_ref.sum()
    hist_err /= hist.sum()
    hist_ref_err /= hist_ref.sum()

    zero_mask_hist = hist == 0.
    zero_mask_hist_ref = hist_ref == 0.

    both_zero = np.logical_and(zero_mask_hist, zero_mask_hist_ref)
    only_hist = np.logical_and(zero_mask_hist, ~zero_mask_hist_ref)
    only_ref = np.logical_and(~zero_mask_hist, zero_mask_hist_ref)

    hist_ratio = np.zeros_like(hist)
    hist_ratio_err = np.zeros_like(hist)
    hist_sigma = np.zeros_like(hist)

    hist_ratio[~both_zero] = hist[~both_zero] / hist_ref[~both_zero]
    hist_ratio_err[~both_zero] = hist_ratio[~both_zero] * np.sqrt((hist_err[~both_zero]/hist[~both_zero])**2 + (hist_ref_err[~both_zero]/hist_ref[~both_zero])**2)
    hist_sigma[~both_zero] = (hist_ratio[~both_zero] - 1.) / hist_ratio_err[~both_zero]


    hist_sigma[both_zero] = 0.
    hist_sigma[only_hist] = -sigma_max
    hist_sigma[only_ref] = sigma_max

    hist = hist_sigma

    # mesh = ax.pcolormesh(x_edges, y_edges, hist.T, norm=mpl.colors.LogNorm(), rasterized=True)
    # mesh = ax.pcolormesh(x_edges, y_edges, hist.T, rasterized=True, vmin=0.5, vmax=1.5)
    mesh = ax.pcolormesh(x_edges, y_edges, hist.T, rasterized=True, vmin=-sigma_max, vmax=sigma_max)

    if showdiag: 
        ax.plot([ranges[0][0], ranges[0][1]], [ranges[1][0], ranges[1][1]], linestyle='dashed', color='black')
    divider = make_axes_locatable(ax)
    cax  = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(mesh, cax=cax)
    # cbar.set_label('Ratio', loc='center')
    cbar.ax.tick_params(axis='y', which='both', direction='out')
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_xlim((xbins[0], xbins[-1]))
    ax.set_ylim((ybins[0], ybins[-1]))
    if logscales[0]:
        ax.set_xscale('log')
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if logscales[1]:
        ax.set_yscale('log')
        ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if len(ticks[0]) > 0: ax.set_xticks(ticks[0])
    if len(ticks[1]) > 0: ax.set_yticks(ticks[1])

    return fig, ax



def plot_pred_correlation(name, targets, samples: SampleEvaluation, plot_dir, train_targets):
    logging.info("Plotting joint r correlations")

    # samples_E = samples[:, 0]
    # samples_m = samples[:, 1]

    # logR_pred_MC_E = samples_E.reshape(-1)
    # logR_pred_median_E, _ = samples_E.median(dim=1)
    # logR_pred_mean_E = samples_E.mean(dim=1)
    # logR_pred_max_likelihood_E = samples_E[np.arange(samples_E.shape[0]), np.argmax(log_likelihoods[:, 0], axis=1)]

    # logR_pred_MC_m = samples_m.reshape(-1)
    # logR_pred_median_m, _ = samples_m.median(dim=1)
    # logR_pred_mean_m = samples_m.mean(dim=1)
    # logR_pred_max_likelihood_m = samples_m[np.arange(samples_m.shape[0]), np.argmax(samples_m, axis=1)]

    targets_E = targets[:, 0]
    targets_m = targets[:, 1]

    # min_values_E = min(min(logR_pred_MC_E.tolist()), min(targets_E.tolist()))
    # max_values_E = max(max(logR_pred_MC_E.tolist()), max(targets_E.tolist()))
    # min_values_m = min(min(logR_pred_MC_m.tolist()), min(targets_m.tolist()))
    # max_values_m = max(max(logR_pred_MC_m.tolist()), max(targets_m.tolist()))

    min_values_E_log = min(targets_E.tolist())
    max_values_E_log = max(targets_E.tolist())
    min_values_m_log = min(targets_m.tolist())
    max_values_m_log = max(targets_m.tolist())

    delta_E = max_values_E_log - min_values_E_log
    delta_m = max_values_m_log - min_values_m_log

    min_values_E_log = min_values_E_log - 0.05 * delta_E
    min_values_m_log = min_values_m_log - 0.05 * delta_m
    max_values_E_log = max_values_E_log + 0.05 * delta_E
    max_values_m_log = max_values_m_log + 0.05 * delta_m

    min_values_E = 10.**min(targets_E.tolist())
    max_values_E = 10.**max(targets_E.tolist())
    min_values_m = 10.**min(targets_m.tolist())
    max_values_m = 10.**max(targets_m.tolist())

    delta_E = max_values_E - min_values_E
    delta_m = max_values_m - min_values_m

    min_values_E = min_values_E - 0.05 * delta_E
    min_values_m = min_values_m - 0.05 * delta_m
    max_values_E = max_values_E + 0.05 * delta_E
    max_values_m = max_values_m + 0.05 * delta_m

    targets_E = targets_E.numpy()
    targets_m = targets_m.numpy()

    # if train_targets != None:
    train_targets = train_targets.numpy()

    with PdfPages(os.path.join(plot_dir, name)) as pdf:
        for (target_data_E, target_data_m), plot_title in zip(
            [
                (targets_E, targets_m),
                (samples.log_response_samples[:, 0], samples.log_response_samples[:, 1]),
                (samples.log_response_median[:, 0], samples.log_response_median[:, 1]),
                (samples.log_response_mean[:, 0], samples.log_response_mean[:, 1]),
                (samples.log_response_mode[:, 0], samples.log_response_mode[:, 1])
            ],
            [
                "Truth", "Samples", "Median", "Mean", "Mode"
            ]
        ):
            logging.info(f"Plotting 2d hist: {plot_title}")
            fig, axs = make_hist_2dim(
                data=[target_data_E, target_data_m],
                labels=[r"$\text{log}_{10} r_E$", r"$\text{log}_{10} r_m$"],
                ranges=[[min_values_E_log, max_values_E_log], [min_values_m_log, max_values_m_log]],
                showdiag=False,
                nbins=[100,100],
                norm_columns=False,
                norm_rows=False,
                show_cbar_label=False
            )
            fig.suptitle(plot_title, y=0.9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        data_comparison = [
            (samples.log_response_samples[:, 0], samples.log_response_samples[:, 1]),
            (samples.log_response_median[:, 0], samples.log_response_median[:, 1]),
            (samples.log_response_mean[:, 0], samples.log_response_mean[:, 1]),
            (samples.log_response_mode[:, 0], samples.log_response_mode[:, 1])
        ]
        titles_comparison = [
            "Samples / True", "Median / True", "Mean / True", "Mode / True"
        ]
        # if train_targets != None:
        data_comparison.append((train_targets[:, 0], train_targets[:, 1]))
        titles_comparison.append("Train / True")

        data_E_ref = targets_E
        data_m_ref = targets_m

        for (target_data_E, target_data_m), plot_title in zip(data_comparison, titles_comparison):
            logging.info(f"Plotting 2d relative hist: {plot_title}")
            data_E = target_data_E
            data_m = target_data_m
            
            fig, axs = make_hist_2dim_ratio(
                data=[data_E, data_m],
                data_ref=[data_E_ref, data_m_ref],
                labels=[r"$\text{log}_{10} r_E$", r"$\text{log}_{10} r_m$"],
                ranges=[[min_values_E_log, max_values_E_log], [min_values_m_log, max_values_m_log]],
                showdiag=False,
                nbins=[100,100],
                sigma_max=3
            )
            fig.suptitle(plot_title, y=0.9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        for (target_data_E, target_data_m), plot_title in zip(data_comparison, titles_comparison):
            logging.info(f"Plotting 2d relative hist: {plot_title}")
            data_E = target_data_E
            data_m = target_data_m
            
            fig, axs = make_hist_2dim_ratio(
                data=[data_E, data_m],
                data_ref=[data_E_ref, data_m_ref],
                labels=[r"$\text{log}_{10} r_E$", r"$\text{log}_{10} r_m$"],
                ranges=[[min_values_E_log, max_values_E_log], [min_values_m_log, max_values_m_log]],
                showdiag=False,
                nbins=[100,100],
                sigma_max=0.1
            )
            fig.suptitle(plot_title, y=0.9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # targets_E = 10.**targets_E
        # targets_m = 10.**targets_m

        # R_pred_MC_E = 10.**logR_pred_MC_E
        # R_pred_median_E = 10.**logR_pred_median_E
        # R_pred_mean_E = 10.**logR_pred_mean_E
        # R_pred_max_likelihood_E = 10.**logR_pred_max_likelihood_E

        # R_pred_MC_m = 10.**logR_pred_MC_m
        # R_pred_median_m = 10.**logR_pred_median_m
        # R_pred_mean_m = 10.**logR_pred_mean_m
        # R_pred_max_likelihood_m = 10.**logR_pred_max_likelihood_m

        for (target_data_E, target_data_m), plot_title in zip(
            [
                (targets_E, targets_m),
                (samples.response_samples[:, 0], samples.response_samples[:, 1]),
                (samples.response_median[:, 0], samples.response_median[:, 1]),
                (samples.response_mean[:, 0], samples.response_mean[:, 1]),
                (samples.response_mode[:, 0], samples.response_mode[:, 1])
            ],
            [
                "Truth", "Samples", "Median", "Mean", "Mode"
            ]
        ):
            logging.info(f"Plotting 2d hist: {plot_title}")
            fig, axs = make_hist_2dim(
                data=[10.**target_data_E, 10.**target_data_m],
                labels=[r"$r_E$", r"$r_m$"],
                ranges=[[min_values_E, max_values_E], [min_values_m, max_values_m]],
                showdiag=False,
                nbins=[100,100],
                norm_columns=False,
                norm_rows=False,
                show_cbar_label=False
            )
            fig.suptitle(plot_title, y=0.9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # for (target_data_E, target_data_m), plot_title in zip(
        #     [
        #         (1./targets_E, 1./targets_m),
        #         (1./samples.response_samples[:, 0], 1./samples.response_samples[:, 1]),
        #         (1./samples.response_median[:, 0], 1./samples.response_median[:, 1]),
        #         (1./samples.response_mean[:, 0], 1./samples.response_mean[:, 1]),
        #         (1./samples.response_mode[:, 0], 1./samples.response_mode[:, 1])
        #     ],
        #     [
        #         "Truth", "Samples", "Median", "Mean", "Mode"
        #     ]
        # ):
        #     logging.info(f"Plotting 2d hist: {plot_title}")
        #     fig, axs = make_hist_2dim(
        #         data=[target_data_E, target_data_m],
        #         labels=[r"$1/r_E$", r"$1/r_m$"],
        #         ranges=[[1./max_values_E, 1./min_values_E], [1./max_values_m, 1./min_values_m]],
        #         showdiag=False,
        #         nbins=[100,100],
        #         norm_columns=False,
        #         norm_rows=False,
        #         show_cbar_label=False
        #     )
        #     fig.suptitle(plot_title, y=0.9)
        #     fig.tight_layout()
        #     pdf.savefig(fig)
        #     plt.close(fig)

def plot_pred_jet_correlation(name, targets, samples: SampleEvaluation, input_data, plot_dir, train_targets, train_input_data):
    logging.info("Plotting joint jet correlations")

    input_data: torch.FloatTensor = input_data[:, :2]

    jet_targets = 10.**targets
    jet_targets = input_data / jet_targets
    
    # if train_targets != None and train_input_data != None:
    train_input_data: torch.FloatTensor = train_input_data[:, :2]
    train_jet_targets = 10.**train_targets
    train_jet_targets = train_input_data / train_jet_targets
    train_jet_targets = train_jet_targets.numpy()
    # logging.info(f"test_targets: {targets.shape}, train_targets: {train_targets.shape}")

    input_data_samples = input_data.unsqueeze(-1).expand(-1, -1, samples.response_samples.shape[-1]).numpy()
    input_data = input_data.numpy()


    
    jet_samples = input_data_samples / samples.response_samples
    jet_median = input_data / samples.response_median
    jet_mean = input_data / samples.response_mean
    jet_mode = input_data / samples.response_mode
    jet_jetmean = input_data / samples.response_jet_mean
    jet_jetmode = input_data / samples.response_jet_mode

    # log_jet_samples = torch.log10(jet_samples)

    # logR_pred_MC_E = samples_E.reshape(-1)
    # logR_pred_median_E, _ = samples_E.median(dim=1)
    # logR_pred_mean_E = samples_E.mean(dim=1)
    # logR_pred_max_likelihood_E = samples_E[np.arange(samples_E.shape[0]), np.argmax(log_likelihoods[:, 0], axis=1)]

    # logR_pred_MC_m = samples_m.reshape(-1)
    # logR_pred_median_m, _ = samples_m.median(dim=1)
    # logR_pred_mean_m = samples_m.mean(dim=1)
    # logR_pred_max_likelihood_m = samples_m[np.arange(samples_m.shape[0]), np.argmax(samples_m, axis=1)]

    # min_values_E = min(min(logR_pred_MC_E.tolist()), min(targets_E.tolist()))
    # max_values_E = max(max(logR_pred_MC_E.tolist()), max(targets_E.tolist()))
    # min_values_m = min(min(logR_pred_MC_m.tolist()), min(targets_m.tolist()))
    # max_values_m = max(max(logR_pred_MC_m.tolist()), max(targets_m.tolist()))

    min_values_E = min(jet_targets[:, 0].tolist())
    max_values_E = max(jet_targets[:, 0].tolist())
    min_values_m = min(jet_targets[:, 1].tolist())
    max_values_m = max(jet_targets[:, 1].tolist())

    delta_E = max_values_E - min_values_E
    delta_m = max_values_m - min_values_m

    min_values_E = min_values_E - 0.05 * delta_E
    min_values_m = min_values_m - 0.05 * delta_m
    max_values_E = max_values_E + 0.05 * delta_E
    max_values_m = max_values_m + 0.05 * delta_m

    min_values_E_log = np.log10(min(jet_targets[:, 0].tolist()))
    max_values_E_log = np.log10(max(jet_targets[:, 0].tolist()))
    min_values_m_log = np.log10(min(jet_targets[:, 1].tolist()))
    max_values_m_log = np.log10(max(jet_targets[:, 1].tolist()))

    delta_E = max_values_E_log - min_values_E_log
    delta_m = max_values_m_log - min_values_m_log

    min_values_E_log = min_values_E_log - 0.05 * delta_E
    min_values_m_log = min_values_m_log - 0.05 * delta_m
    max_values_E_log = max_values_E_log + 0.05 * delta_E
    max_values_m_log = max_values_m_log + 0.05 * delta_m

    jet_targets = jet_targets.numpy()

    with PdfPages(os.path.join(plot_dir, name)) as pdf:
        for (target_data_E, target_data_m), plot_title in zip(
            [
                (jet_targets[:, 0], jet_targets[:, 1]),
                (jet_samples[:, 0], jet_samples[:, 1]),
                (jet_median[:, 0], jet_median[:, 1]),
                (jet_mean[:, 0], jet_mean[:, 1]),
                (jet_mode[:, 0], jet_mode[:, 1]),
                (jet_jetmean[:, 0], jet_jetmean[:, 1]),
                (jet_jetmode[:, 0], jet_jetmode[:, 1]),
            ],
            [
                "Truth", "Samples", "Median", "Mean", "Mode", "Jet Mean", "Jet Mode"
            ]
        ):
            logging.info(f"Plotting 2d hist: {plot_title}")
            fig, axs = make_hist_2dim(
                data=[np.log10(target_data_E), np.log10(target_data_m)],
                labels=[r"$\text{Jet energy } \log_{10} E$", r"$\text{Jet mass } \log_{10} m$"],
                ranges=[[min_values_E_log, max_values_E_log], [min_values_m_log, max_values_m_log]],
                showdiag=False,
                nbins=[100,100],
                norm_columns=False,
                norm_rows=False,
                show_cbar_label=False
            )
            fig.suptitle(plot_title, y=0.9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        
            fig, axs = make_hist_2dim(
                data=[target_data_E, target_data_m],
                labels=[r"$\text{Jet energy } E [\text{GeV}]$", r"$\text{Jet mass } m [\text{GeV}]$"],
                ranges=[[min_values_E, max_values_E], [min_values_m, max_values_m]],
                showdiag=False,
                nbins=[100,100],
                norm_columns=False,
                norm_rows=False,
            )
            fig.suptitle(plot_title, y=0.9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    
        data_comparison = [
            (jet_samples[:, 0], jet_samples[:, 1]),
            (jet_median[:, 0], jet_median[:, 1]),
            (jet_mean[:, 0], jet_mean[:, 1]),
            (jet_mode[:, 0], jet_mode[:, 1]),
            (jet_jetmean[:, 0], jet_jetmean[:, 1]),
            (jet_jetmode[:, 0], jet_jetmode[:, 1]),
        ]
        titles_comparison = [
            "Samples / True", "Median / True", "Mean / True", "Mode / True", "Jet Mean / True", "Jet Mode / True"
        ]
        # if train_targets != None:
        data_comparison.append((train_jet_targets[:, 0], train_jet_targets[:, 1]))
        titles_comparison.append("Train / True")

        data_E_ref = jet_targets[:, 0]
        data_m_ref = jet_targets[:, 1]
        log_data_E_ref = np.log10(data_E_ref)
        log_data_m_ref = np.log10(data_m_ref)

        for (target_data_E, target_data_m), plot_title in zip(data_comparison, titles_comparison):
            logging.info(f"Plotting 2d relative hist: {plot_title}")
            data_E = target_data_E
            data_m = target_data_m

            fig, axs = make_hist_2dim_ratio(
                data=[np.log10(data_E), np.log10(data_m)],
                data_ref=[log_data_E_ref, log_data_m_ref],
                labels=[r"$\text{Jet energy } \log_{10} E$", r"$\text{Jet mass } \log_{10} m$"],
                ranges=[[min_values_E_log, max_values_E_log], [min_values_m_log, max_values_m_log]],
                showdiag=False,
                nbins=[100,100],
                sigma_max=3.
            )
            fig.suptitle(plot_title, y=0.9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        for (target_data_E, target_data_m), plot_title in zip(data_comparison, titles_comparison):
            logging.info(f"Plotting 2d relative hist: {plot_title}")
            data_E = target_data_E
            data_m = target_data_m

            fig, axs = make_hist_2dim_ratio(
                data=[np.log10(data_E), np.log10(data_m)],
                data_ref=[log_data_E_ref, log_data_m_ref],
                labels=[r"$\text{Jet energy } \log_{10} E$", r"$\text{Jet mass } \log_{10} m$"],
                ranges=[[min_values_E_log, max_values_E_log], [min_values_m_log, max_values_m_log]],
                showdiag=False,
                nbins=[100,100],
                sigma_max=0.01
            )
            fig.suptitle(plot_title, y=0.9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

def plot_pred_jet_correlation_marginalized(name, targets, samples: SampleEvaluation, input_data, plot_dir, train_targets=None, train_input_data=None):
    logging.info("Plotting joint jet correlations marginalized")
    input_data: torch.FloatTensor = input_data[:, :2]

    jet_targets = 10.**targets
    jet_targets = input_data / jet_targets
    
    input_data_samples = input_data.unsqueeze(-1).expand(-1, -1, samples.response_samples.shape[-1]).numpy()
    input_data = input_data.numpy()
    
    jet_samples = input_data_samples / samples.response_samples
    jet_mean = input_data / samples.response_mean
    jet_mode = input_data / samples.response_mode
    jet_jetmean = input_data / samples.response_jet_mean
    jet_jetmode = input_data / samples.response_jet_mode

    min_values_E = min(jet_targets[:, 0].tolist())
    max_values_E = max(jet_targets[:, 0].tolist())
    min_values_m = min(jet_targets[:, 1].tolist())
    max_values_m = max(jet_targets[:, 1].tolist())

    delta_E = max_values_E - min_values_E
    delta_m = max_values_m - min_values_m

    min_values_E = min_values_E - 0.05 * delta_E
    min_values_m = min_values_m - 0.05 * delta_m
    max_values_E = max_values_E + 0.05 * delta_E
    max_values_m = max_values_m + 0.05 * delta_m

    min_values_E_log = np.log10(min(jet_targets[:, 0].tolist()))
    max_values_E_log = np.log10(max(jet_targets[:, 0].tolist()))
    min_values_m_log = np.log10(min(jet_targets[:, 1].tolist()))
    max_values_m_log = np.log10(max(jet_targets[:, 1].tolist()))

    delta_E = max_values_E_log - min_values_E_log
    delta_m = max_values_m_log - min_values_m_log

    min_values_E_log = min_values_E_log - 0.05 * delta_E
    min_values_m_log = min_values_m_log - 0.05 * delta_m
    max_values_E_log = max_values_E_log + 0.05 * delta_E
    max_values_m_log = max_values_m_log + 0.05 * delta_m

    jet_targets = jet_targets.numpy()

    with PdfPages(os.path.join(plot_dir, name)) as pdf:
        for (target_data_E, target_data_m), plot_title in zip(
            [
                (jet_targets[:, 0], jet_targets[:, 1]),
                (jet_samples[:, 0], jet_samples[:, 1]),
                (jet_mean[:, 0], jet_mean[:, 1]),
                (jet_mode[:, 0], jet_mode[:, 1]),
                (jet_jetmean[:, 0], jet_jetmean[:, 1]),
                (jet_jetmode[:, 0], jet_jetmode[:, 1]),
            ],
            [
                "Truth", "Sampling", "Mean", "Mode", "Jet Mean", "Jet Mode"
            ]
        ):
            logging.info(f"Plotting 2d hist: {plot_title}")
            fig, axs = make_hist_2dim(
                data=[np.log10(target_data_E), np.log10(target_data_m)],
                labels=[r"$\text{Jet energy } \log_{10} E$", r"$\text{Jet mass } \log_{10} m$"],
                ranges=[[min_values_E_log, max_values_E_log], [min_values_m_log, max_values_m_log]],
                showdiag=False,
                nbins=[100,100],
                norm_columns=True,
                norm_rows=False,
                show_cbar_label=False
            )
            fig.suptitle(f"{plot_title} Marginalized E", y=0.9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        for (target_data_E, target_data_m), plot_title in zip(
            [
                (jet_targets[:, 0], jet_targets[:, 1]),
                (jet_samples[:, 0], jet_samples[:, 1]),
                (jet_mean[:, 0], jet_mean[:, 1]),
                (jet_mode[:, 0], jet_mode[:, 1]),
                (jet_jetmean[:, 0], jet_jetmean[:, 1]),
                (jet_jetmode[:, 0], jet_jetmode[:, 1]),
            ],
            [
                "Truth", "Sampling", "Mean", "Mode", "Jet Mean", "Jet Mode"
            ]
        ):
            fig, axs = make_hist_2dim(
                data=[target_data_E, target_data_m],
                labels=[r"$\text{Jet energy } E$", r"$\text{Jet mass } m$"],
                ranges=[[min_values_E, max_values_E], [min_values_m, max_values_m]],
                showdiag=False,
                nbins=[100,100],
                norm_columns=True,
                norm_rows=False,
            )
            fig.suptitle(f"{plot_title} Marginalized E", y=0.9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        for (target_data_E, target_data_m), plot_title in zip(
            [
                (jet_targets[:, 0], jet_targets[:, 1]),
                (jet_samples[:, 0], jet_samples[:, 1]),
                (jet_mean[:, 0], jet_mean[:, 1]),
                (jet_mode[:, 0], jet_mode[:, 1]),
                (jet_jetmean[:, 0], jet_jetmean[:, 1]),
                (jet_jetmode[:, 0], jet_jetmode[:, 1]),
            ],
            [
                "Truth Marginalized", "Sampling", "Mean", "Mode", "Jet Mean", "Jet Mode"
            ]
        ):
            logging.info(f"Plotting 2d hist: {plot_title}")
            fig, axs = make_hist_2dim(
                data=[np.log10(target_data_E), np.log10(target_data_m)],
                labels=[r"$\text{Jet energy } \log_{10} E$", r"$\text{Jet mass } \log_{10} m$"],
                ranges=[[min_values_E_log, max_values_E_log], [min_values_m_log, max_values_m_log]],
                showdiag=False,
                nbins=[100,100],
                norm_columns=False,
                norm_rows=True,
                show_cbar_label=False
            )
            fig.suptitle(f"{plot_title} Marginalized m", y=0.9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        for (target_data_E, target_data_m), plot_title in zip(
            [
                (jet_targets[:, 0], jet_targets[:, 1]),
                (jet_samples[:, 0], jet_samples[:, 1]),
                (jet_mean[:, 0], jet_mean[:, 1]),
                (jet_mode[:, 0], jet_mode[:, 1]),
                (jet_jetmean[:, 0], jet_jetmean[:, 1]),
                (jet_jetmode[:, 0], jet_jetmode[:, 1])
            ],
            [
                "Truth Marginalized", "Sampling", "Mean", "Mode", "Jet Mean", "Jet Mode"
            ]
        ):
            fig, axs = make_hist_2dim(
                data=[target_data_E, target_data_m],
                labels=[r"$\text{Jet energy } E$", r"$\text{Jet mass } m$"],
                ranges=[[min_values_E, max_values_E], [min_values_m, max_values_m]],
                showdiag=False,
                nbins=[100,100],
                norm_columns=False,
                norm_rows=True,
            )
            fig.suptitle(f"{plot_title} Marginalized m", y=0.9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

