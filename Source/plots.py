import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import os
from matplotlib.backends.backend_pdf import PdfPages
import logging
from scipy.stats import binned_statistic

from Source.plot_utils import *
plt.style.use('/remote/gpu07/huetsch/JetCalibration/Source/plotting.mplstyle')


figs_hist1d = (1.40*2.953, 1.20*2.568)
figs_hist2d = (1.40*2.953, 1.20*2.568)
figs_hist_twice = (1.40*2.953*2, 1.20*2.568)
figs_ratio  = (1.40*2.953, 1.40*2.568)
rect_hist1d = (0.11, 0.11, 0.98, 0.97) # left, bottom, right, top
rect_hist2d = (0.11, 0.11, 0.87, 0.97) # left, bottom, right, top
rect_ratio  = (0.11, 0.09, 0.98, 0.97) # left, bottom, right, top
subs        = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
num_bins    = 100
dup_last    = lambda a: np.append(a, a[-1])


c_bk = '#0a0a0a' # black      RGB(10,  10,  10)
c_rd = '#ff0000' # red        RGB(255, 0, 0)
c_gn = '#00CC00' # green      RGB(0, 204, 0)
#c_gn = '#3BC14A' # green      RGB(0, 204, 0)
c_bl = '#0000ff' # blue       RGB(0, 0, 255)
c_yl = '#ffff00' # yellow     RGB(255, 255, 0)
c_cy = '#00ffff' # cyan       RGB(0, 255, 255)
c_db = '#006699' # dark blue  RGB( 0, 102, 153)
c_br = '#663300' # brown      RGB(102, 51, 0)
colors = {'bk': c_bk, 'rd': c_rd, 'gn': c_gn, 'bl': c_bl, 'yl': c_yl, 'cy': c_cy, 'db': c_db, 'br': c_br}



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




feature_range = { # dictionary containing all variables and their corresponding index, value range, binning and log-scaling
                'r_ems': [0,  +1.000e-01, +6.900e+00, 1200, False, +5.000e-02, +4.950e+00], # r_truth or r_target (120 bins if x-axis)
                'e_ems': [1,  +1.000e-02, +3.162e+03,   54,  True, +2.000e-01, +1.000e+03], # clusterE (cluster signal)
                'y_ems': [2,  -2.750e+00, +2.750e+00,   55, False, -2.400e+00, +2.400e+00], # clusterEta (cluster rapidity)
                'm_sig': [3,  +1.259e-03, +7.943e+02,   57,  True, +8.000e-02, +7.943e+02], # cluster_SIGNIFICANCE (signal significance)
                't_ems': [4,  -1.230e+02, +1.230e+02,  123, False, -5.800e+01, +4.400e+01], # cluster_time (cluster time)
                'm_tim': [5,  -1.000e+01, +1.990e+02,   50, False, +0.000e+00, +1.990e+02], # cluster_SECOND_TIME (cell-time variance)
                'm_lam': [6,  +0.000e+00, +6.000e+03,  100, False, +4.900e-03, +4.100e+03], # cluster_CENTER_LAMBDA (depth in calorimeter)
                'm_cog': [7,  +1.200e+03, +7.800e+03,  100, False, +1.200e+03, +6.600e+03], # cluster_CENTER_MAG (distance from vertex)
                'm_fem': [8,  -1.000e-02, +1.001e+00,  102, False, -1.500e-02, +1.015e+00], # cluster_ENG_FRAC_EM (signal fraction in EMC)
                'm_rho': [9,  +1.000e-09, +1.000e-03,   60,  True, +1.000e-09, +4.400e-04], # cluster_FIRST_ENG_DENS (signal density)
                'm_lon': [10, -1.000e-02, +1.001e+00,  102, False, -1.500e-02, +1.015e+00], # cluster_LONGITUDINAL (longitudinal signal dispersion)
                'm_lat': [11, -1.000e-02, +1.001e+00,  102, False, -1.500e-02, +1.015e+00], # cluster_LATERAL (lateral signal dispersion)
                'm_ptd': [12, -1.000e-02, +1.001e+00,  102, False, -1.500e-02, +1.015e+00], # cluster_PTD (signal compactness)
                'm_iso': [13, -1.000e-02, +1.001e+00,  102, False, -1.500e-02, +1.015e+00], # cluster_ISOLATION (cluster isolation)
                'p_npv': [14, -2.500e-01, +9.575e+01,   48, False, -5.000e-02, +5.850e+01], # nPrimVtx (number of reconstructed primary vertices)
                'p_mmu': [15, -2.500e-01, +9.575e+01,   48, False, -5.000e-02, +8.050e+01], # avgMu (number of pile-up interactions)
                'e_dep': [16, +1.000e-02, +3.162e+03,   54,  True, +2.000e-01, +1.000e+03], # e_truth or cluster_ENG_CALIB_TOT
                'e_dnn': [17, +1.000e-02, +3.162e+03,   54,  True, +2.000e-01, +1.000e+03], # e_model from deterministic NN (DNN)
                'e_bnn': [18, +1.000e-02, +3.162e+03,   54,  True, +2.000e-01, +1.000e+03], # e_model from Bayesian NN (BNN)
                'e_rde': [19, +1.000e-02, +3.162e+03,   54,  True, +2.000e-01, +1.000e+03], # e_model from Bayesian NN (RE)
                'r_dnn': [20, +1.000e-01, +6.900e+00, 1200, False, +5.000e-02, +4.950e+00], # r_model from deterministic NN (120 bins if x-axis)
                'r_bnn': [21, +1.000e-01, +6.900e+00, 1200, False, +5.000e-02, +4.950e+00], # r_model from Bayesian NN (120 bins if x-axis)
                'r_rde': [22, +1.000e-01, +6.900e+00, 1200, False, +5.000e-02, +4.950e+00], # r_model from Bayesian NN (120 bins if x-axis)
                'c_dnn': [23, -1.050e+00, +6.950e+00, 1600, False, -6.500e-01, +0.950e+00], # closure or prediction power: r_dnn/r_ems-1 (160 bins if x-axis)
                'c_bnn': [24, -1.050e+00, +6.950e+00, 1600, False, -6.500e-01, +0.950e+00], # closure or prediction power: r_bnn/r_ems-1 (160 bins if x-axis)
                'c_rde': [25, -1.050e+00, +6.950e+00, 1600, False, -6.500e-01, +0.950e+00], # closure or prediction power: r_bnn/r_ems-1 (160 bins if x-axis)
                'l_ems': [26, -1.050e+00, +6.950e+00, 1600, False, -9.500e-01, +1.950e+00], # signal linearity: e_ems/e_dep-1 = clusterE/cluster_ENG_CALIB_TOT-1
                'l_had': [27, -1.050e+00, +6.950e+00, 1600, False, -9.500e-01, +1.950e+00], # signal linearity: e_had/e_dep-1 = clusterECalib/cluster_ENG_CALIB_TOT-1
                'l_dnn': [28, -1.050e+00, +6.950e+00, 1600, False, -9.500e-01, +1.950e+00], # signal linearity: e_dnn/e_dep-1 = e_dnn/cluster_ENG_CALIB_TOT-1
                'l_bnn': [29, -1.050e+00, +6.950e+00, 1600, False, -9.500e-01, +1.950e+00], # signal linearity: e_bnn/e_dep-1 = e_bnn/cluster_ENG_CALIB_TOT-1
                'l_rde': [30, -1.050e+00, +6.950e+00, 1600, False, -9.500e-01, +1.950e+00], # signal linearity: e_bnn/e_dep-1 = e_bnn/cluster_ENG_CALIB_TOT-1
            }

feature_names = { # dictionary containing all variables and their corresponding LaTeX description (and unit if needed)
                'r_ems': [0,  'Target response', '$R_{\\text{clus}}^{\\text{EM}}$'],
                'e_ems': [1,  'Cluster signal', '$E_{\\text{clus}}^{\\text{EM}}$', '[$\\text{GeV}$]'],
                'y_ems': [2,  'Cluster rapidity', '$y_{\\text{clus}}^{\\text{EM}}$'],
                'm_sig': [3,  'Signal significance', '$\zeta_{\\text{clus}}^{\\text{EM}}$'],
                't_ems': [4,  'Cluster time', '$t_{\\text{clus}}$', '[$\\text{ns}$]'],
                'm_tim': [5,  'Cell-time variance', '$\\text{Var}_{\\text{clus}}(t_{\\text{cell}})$', '[$\\text{ns}^{2}$]'],
                'm_lam': [6,  'Depth in calorimeter', '$\lambda_{\\text{clus}}$', '[$\\text{mm}$]'],
                'm_cog': [7,  'Distance from vertex', '$\\vert\\vec{c}_{\\text{clus}}\\vert$', '[$\\text{mm}$]'],
                'm_fem': [8,  'Signal fraction in EMC', '$f_{\\text{emc}}$'],
                'm_rho': [9,  'Signal density', '$\\rho_{\\text{clus}}=\langle\\rho_{\\text{cell}}\\rangle$', '[$\\text{GeV}/\\text{mm}^{3}$]'],
                'm_lon': [10, 'Longitudinal signal dispersion', '$\langle\mathfrak{m}_{\\text{long}}^{2}\\rangle$'],
                'm_lat': [11, 'Lateral signal dispersion', '$\langle\mathfrak{m}_{\\text{lat}}^{2}\\rangle$'],
                'm_ptd': [12, 'Signal compactness', '$p_{\\text{T}}D$'],
                'm_iso': [13, 'Cluster isolation', '$f_{\\text{iso}}$'],
                'p_npv': [14, 'Number of reconstructed vertices', '$N_{\\text{PV}}$'],
                'p_mmu': [15, 'Number of pile-up interactions', '$\mu$'],
                'e_dep': [16, 'Deposited energy',   '$E_{\\text{clus}}^{\\text{dep}}$', '[$\\text{GeV}$]'],
                'e_dnn': [17, 'DNN-calibrated energy', '$E_{\\text{clus}}^{\\text{DNN}}$', '[$\\text{GeV}$]'],
                'e_bnn': [18, 'BNN-calibrated energy', '$E_{\\text{clus}}^{\\text{BNN}}$', '[$\\text{GeV}$]'],
                'e_rde': [19,  'RE-calibrated energy', '$E_{\\text{clus}}^{\\text{RE}}$',  '[$\\text{GeV}$]'],
                'r_dnn': [20, 'DNN-predicted response', '$R_{\\text{clus}}^{\\text{DNN}}$'],
                'r_bnn': [21, 'BNN-predicted response', '$R_{\\text{clus}}^{\\text{BNN}}$'],
                'r_rde': [22,  'RE-predicted response', '$R_{\\text{clus}}^{\\text{RE}}$'],
                'c_dnn': [23, 'Prediciton power', '$\Delta_{R}^{\\text{DNN}}$'],
                'c_bnn': [24, 'Prediciton power', '$\Delta_{R}^{\\text{BNN}}$'],
                'c_rde': [25, 'Prediciton power', '$\Delta_{R}^{\\text{RE}}$'],
                'l_ems': [26, 'Signal linearity', '$\Delta_{E}^{\\text{EM}}$'],
                'l_had': [27, 'Signal linearity', '$\Delta_{E}^{\\text{had}}$'],
                'l_dnn': [28, 'Signal linearity', '$\Delta_{E}^{\\text{DNN}}$'],
                'l_bnn': [29, 'Signal linearity', '$\Delta_{E}^{\\text{BNN}}$'],
                'l_rde': [30, 'Signal linearity', '$\Delta_{E}^{\\text{RE}}$'],
            }

# select target and input-feature keys from directory (response and 15 topo-cluster observables)
keys_feats = [key for key in feature_names if (feature_names[key][0] >= 1) and (feature_names[key][0] <= 16)]

keys_current = ['E', 'mass', 'rap', 'groomMRatio', 'Width', 'Split12', 'Split23', 'C2', 'D2', 'Tau32', 'Tau21', 'Qw', 'EMFracCaloBased', 'EM3FracCaloBased', 'Tile0FracCaloBased', 'EffNClustsCaloBased', 'NeutralEFrac', 'ChargePTFrac', 'ChargeMFrac', 'averageMu', 'NPV']
log_scales = [True, True, False, False, False, True, True, False, True, False, False, True, False, False, False, False, False, False, False, True, True]


class Plotter():
    def __init__(self, plot_dir, params, test_predictions, test_inputs, test_targets, samples, log_likelihoods, variable):
        self.plot_dir = plot_dir
        self.params = params
        self.test_predictions = test_predictions
        self.test_inputs = test_inputs.numpy()
        self.test_targets = test_targets.numpy()

        self.samples = samples.numpy()
        self.log_likelihoods = log_likelihoods.numpy()

        self.n_samples = self.samples.shape[-1]

        self.variable = variable


    def plot_loss_history(self, name, train_loss, val_loss):
        plt.plot(train_loss, label="Training loss")
        plt.plot(val_loss, label="Validation loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss history")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, name), format='pdf', bbox_inches="tight")
        plt.close()


    def plot_inputs_histogram(self, name):
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
            

    def r_predictions_histogram(self, name):

        logR_truth = self.test_targets.squeeze()
        logR_pred_MC = self.samples.squeeze()
        logR_pred_mean = self.samples.mean(axis=1)
        logR_pred_max_likelihood = self.samples[np.arange(self.samples.shape[0]), np.argmax(self.log_likelihoods, axis=1)]

        logr_range = compute_range([logR_truth, logR_pred_MC], quantile=0.00001)
        fig1, axs1 =  make_hist_1dim_ratio(
            data       = [logR_truth, logR_pred_MC, logR_pred_mean, logR_pred_max_likelihood],
            labels     = ['$R_{\\text{true}}$',
                        '$R_{\\text{pred}}$ MC',
                        '$R_{\\text{pred}}$ Mean',
                        '$R_{\\text{pred}}$ Mode'],
            colors     = [colors['bk'], colors['bl'], colors['rd'], colors['gn']],
            showratios = [False, True, True, True],
            ratioref   = 0,
            xlabel     = 'Log Response $\\log(R_{\\text{%s}})$' % self.variable,
            rlabel     = '$Pred/True$',
            xrange     = logr_range,
            ticks      = [[], [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]],
            logscales  = [False, True],
            nbins      = 100,
            legend     = ['lower center', 0.65, 0.05, None] if self.variable == "E" else ['upper right', 0.95, 0.95, None],
            #legend     = ['upper right', 0.95, 0.95, None],
            atlas_info = [0.04, 0.95, 'four-lines', 'left', 'top', 'none'])
        

        r_truth = 10**self.test_targets.squeeze()
        r_pred_MC = 10**self.samples.squeeze()
        r_pred_mean = (10**self.samples).mean(axis=1).squeeze()
        max_likelihood_fixed = (self.log_likelihoods - self.samples).argmax(axis=1)
        r_pred_max_likelihood_fixed = 10**self.samples[np.arange(self.samples.shape[0]), max_likelihood_fixed]

        
        r_range = compute_range([r_truth, r_pred_MC[:, 0]], quantile=0.00001)

        fig2, axs2 =  make_hist_1dim_ratio(
            data       = [r_truth, r_pred_MC, r_pred_mean, r_pred_max_likelihood_fixed],
            labels     = ['$R_{\\text{true}}$',
                        '$R_{\\text{pred}}$ MC',
                        '$R_{\\text{pred}}$ Mean',
                        '$R_{\\text{pred}}$ Mode'],
            colors     = [colors['bk'], colors['bl'], colors['rd'], colors['gn']],
            showratios = [False, True, True, True],
            ratioref   = 0,
            xlabel     = 'Response $R_{\\text{%s}}$' % self.variable,
            rlabel     = '$Pred/True$',
            xrange     = r_range,
            ticks      = [[], [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]],
            logscales  = [False, True],
            nbins      = 100,
            legend     = ['lower center', 0.65, 0.05, None] if self.variable == "E" else ['upper right', 0.95, 0.95, None],
            atlas_info = [0.04, 0.95, 'four-lines', 'left', 'top', 'none'])

        with PdfPages(os.path.join(self.plot_dir, name)) as pdf:
            pdf.savefig(fig1)
            pdf.savefig(fig2)
        plt.close(fig1)
        plt.close(fig2)

    def E_M_predictions_histogram(self, name):

        r_truth = 10**self.test_targets.squeeze()
        r_pred_MC = 10**self.samples.squeeze()
        r_pred_mean = (10**self.samples).mean(axis=1).squeeze()

        max_likelihood_fixed = (self.log_likelihoods - self.samples).argmax(axis=1)

        r_pred_max_likelihood_fixed = 10**self.samples[np.arange(self.samples.shape[0]), max_likelihood_fixed]

        if self.variable == "E":
            ins = self.test_inputs[:, 0]
            truth = (1./r_truth) * ins
            pred_MC = (1./r_pred_MC) * ins[:, np.newaxis]
            pred_mean = (1./r_pred_mean) * ins
            pred_max_likelihood_fixed = (1./r_pred_max_likelihood_fixed) * ins

            range = [np.min(truth)-100, np.quantile(truth, 0.999999)]
            labels = ['$E_{\\text{true}}$',
                      '$E_{\\text{pred}}$ MC',
                      '$E_{\\text{pred}}$ Mean',
                      '$E_{\\text{pred}}$ Mode',
                      ]
        else:
            truth = (1./r_truth) * self.test_inputs[:, 1]
            ins = self.test_inputs[:, 1]
            pred_MC = (1./r_pred_MC) * ins[:, np.newaxis]
            pred_mean = (1./r_pred_mean) * ins
            pred_max_likelihood_fixed = (1./r_pred_max_likelihood_fixed) * ins

            range = [np.min(truth)- 100, np.quantile(truth, 0.99999)]
            labels = ['$m_{\\text{true}}$',
                      '$m_{\\text{pred}}$ MC',
                      '$m_{\\text{pred}}$ Mean',
                      '$m_{\\text{pred}}$ Mode',
                      ]

        fig, axs =  make_hist_1dim_ratio(
            data       = [truth, pred_MC, pred_mean, pred_max_likelihood_fixed],
            labels     = labels,
            colors     = [colors['bk'], colors['bl'], colors['rd'], colors['gn']],
            showratios = [False, True, True, True],
            ratioref   = 0,
            xlabel     = "Jet Energy $E$" if self.variable == "E" else "Jet mass $m$",
            rlabel     = "$Pred/True$",
            xrange     = range,
            ticks      = [[], [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]],
            logscales  = [False, True],
            nbins      = 100,
            legend     = ['lower center', 0.65, 0.05, None],
            atlas_info = [0.04, 0.95, 'four-lines', 'left', 'top', 'none'])

        fig.savefig(os.path.join(self.plot_dir, name), format='pdf')
        plt.close(fig)




    def r_2d_histogram(self, name):
        r_truth = self.test_targets.squeeze()            

        with PdfPages(os.path.join(self.plot_dir, name)) as pdf:
            r_pred = self.samples[:, 0].squeeze()
            r_range = compute_range([r_truth, r_pred], quantile=0.00001)
            label = 'Model $\\log(R_%s)$ Post. MC' % self.variable
            fig, ax = make_hist_2dim(
                data       = [r_truth, r_pred],
                labels     = ['True $\\log(R_%s)$' % self.variable,
                            label],
                ranges     = [r_range, r_range],
                ticks      = [[],
                            []],
                logscales  = [False, False],
                nbins      = [100, 100],
                atlas_info = [0.96, 0.05, 'three-lines', 'right', 'bottom', 'none'])
            pdf.savefig(fig)
            plt.close(fig)
            r_pred = self.samples.mean(axis=1).squeeze()
            label = 'Model $\\log(R_%s)$ Post. Mean' % self.variable
            fig, ax = make_hist_2dim(
                data       = [r_truth, r_pred],
                labels     = ['True $\\log(R_%s)$' % self.variable,
                            label],
                ranges     = [r_range, r_range],
                ticks      = [[],
                            []],
                logscales  = [False, False],
                nbins      = [100, 100],
                atlas_info = [0.96, 0.05, 'three-lines', 'right', 'bottom', 'none'])
            pdf.savefig(fig)
            plt.close(fig)
            r_pred = self.samples[np.arange(self.samples.shape[0]), np.argmax(self.log_likelihoods, axis=1)].squeeze()
            label = 'Model $\\log(R_%s)$ Post. Maximum' % self.variable
            fig, ax = make_hist_2dim(
                data       = [r_truth, r_pred],
                labels     = ['True $\\log(R_%s)$' % self.variable,
                            label],
                ranges     = [r_range, r_range],
                ticks      = [[],
                            []],
                logscales  = [False, False],
                nbins      = [100, 100],
                atlas_info = [0.96, 0.05, 'three-lines', 'right', 'bottom', 'none'])
            pdf.savefig(fig)
            plt.close(fig)


    def E_M_2d_histogram(self, name, variable="E", mode="sample"):

        r_truth = 10**self.test_targets.squeeze()
        r_pred_MC = 10**self.samples.squeeze()[:, 0]
        r_pred_mean = (10**self.samples).mean(axis=1).squeeze()

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


    def pred_inputs_histogram(self, name):

        r_truth = 10**self.test_targets.squeeze()
        r_pred = 10**self.samples[:, 0].squeeze()

        with PdfPages(os.path.join(self.plot_dir, name)) as pdf:

            for feature in range(self.test_inputs.shape[1]):

                input_data = self.test_inputs[:, feature]

                xmin = np.quantile(input_data, 0.001)
                xmax = np.quantile(input_data, 0.999)
                xbin = 100

                if log_scales[feature] and False:
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
                hist_pred, _, _ = np.histogram2d(input_data, r_pred, bins=[xedges, yedges], density=False)

                vmin = min([np.min(hist_target[np.nonzero(hist_target)]/np.sum(hist_target)),
                            np.min(hist_pred[np.nonzero(hist_pred)]/np.sum(hist_pred))])
                vmax = max([np.max(hist_target[np.nonzero(hist_target)]/np.sum(hist_target)),
                            np.max(hist_pred[np.nonzero(hist_pred)]/np.sum(hist_pred))])

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

                axs[0].set_xlabel(keys_current[feature])
                axs[0].set_ylabel("True response $R$")

                if log_scales[feature] and False:
                    axs[0].set_xscale('log')
                axs[0].set_xlim((xmin, xmax))
                axs[0].set_ylim((ymin, ymax))

                if False:
                    axs[0].set_yscale('log')
                    axs[0].yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=subs, numticks=999))
                    axs[0].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

                hist = axs[1].hist2d(input_data, r_pred, weights=np.zeros_like(input_data)+1./input_data.size, bins=[xedges, yedges], norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), rasterized=True)
                divider = make_axes_locatable(axs[1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(hist[3], cax=cax)
                cbar.set_label('Relative entries', loc='center')
                cbar.ax.tick_params(axis='y', which='both', direction='out')

                axs[1].set_xlabel(keys_current[feature])
                axs[1].set_ylabel("Predicted response $R$")

                if log_scales[feature] and False:
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


    def pred_inputs_histogram_marginalized(self, name):

        r_truth = 10**self.test_targets.squeeze()
        r_pred = 10**self.samples.squeeze()

        loglikelihoods = self.log_likelihoods# / r_pred
        maximum_likelihood_index = np.argmax(loglikelihoods, axis=1)
        maximum_likelihood_r = r_pred[np.arange(r_pred.shape[0]), maximum_likelihood_index]

        with PdfPages(os.path.join(self.plot_dir, name)) as pdf:

            for feature in range(self.test_inputs.shape[1]):

                input_data = self.test_inputs[:, feature]

                xmin = np.quantile(input_data, 0.001)
                xmax = np.quantile(input_data, 0.999)
                xbin = 50
                if log_scales[feature] and False:
                    xedges = np.logspace(np.log10(xmin), np.log10(xmax), xbin+1)
                else: 
                    xedges = np.linspace(xmin, xmax, xbin+1)

                median_target, _, _ = binned_statistic(input_data, r_truth, statistic='median', bins=xedges)

                median_samples = []
                for sample in range(self.n_samples):
                    median_sample, _, _ = binned_statistic(input_data, r_pred[:, sample], statistic='median', bins=xedges)
                    median_samples.append(median_sample)

                median_samples_mean = np.mean(np.array(median_samples), axis=0)
                median_samples_std = np.std(np.array(median_samples), axis=0)

                posterior_mean = np.mean(r_pred, axis=1)
                median_posteriormean, _, _ = binned_statistic(input_data, posterior_mean, statistic='median', bins=xedges)

                posterior_median = np.median(r_pred, axis=1)
                median_posteriormedian, _, _ = binned_statistic(input_data, posterior_median, statistic='median', bins=xedges)

                median_posteriormaximum, _, _ = binned_statistic(input_data, maximum_likelihood_r, statistic='median', bins=xedges)

                bin_centers = (xedges[:-1] + xedges[1:]) / 2
                ymin, ymax = np.min(median_target)-0.1, np.max(median_target)+0.1

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figs_ratio, sharex=True, gridspec_kw={'height_ratios': [3,1]})
                fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_ratio)
                fig.subplots_adjust(hspace=0.07)

                # Top panel
                ax1.step(bin_centers, median_target, color=colors['bk'], label='True $R$')
                ax1.step(bin_centers, median_samples_mean, color=colors['bl'], label='$R$ Sampled')
                ax1.step(bin_centers, median_posteriormean, color=colors['rd'], label='$R$ Post. Mean')
                ax1.step(bin_centers, median_posteriormedian, color=colors['gn'], label='$R$ Post. Median')
                ax1.step(bin_centers, median_posteriormaximum, color=colors['br'], label='$R$ Post. Maximum')
                ax1.fill_between(bin_centers, median_samples_mean - median_samples_std, median_samples_mean + median_samples_std, color=colors['bl'], alpha=0.4, step='pre')
                ax1.set_ylabel('Bin Median $R$')
                ax1.set_xlim((xmin, xmax))
                ax1.set_ylim((ymin, ymax))  
                if log_scales[feature] and False:
                    ax1.set_xscale('log')
                ax1.legend(borderpad=0.4)
                ax1.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))

                # Bottom ratio panel
                ratio_samples = median_samples_mean / median_target
                ratio_posteriormean = median_posteriormean / median_target
                ratio_posteriormedian = median_posteriormedian / median_target
                ratio_posteriormaximum = median_posteriormaximum / median_target
                ax2.step(bin_centers, ratio_samples, color=colors['bl'])
                ax2.step(bin_centers, ratio_posteriormean, color=colors['rd'])
                ax2.step(bin_centers, ratio_posteriormedian, color=colors['gn'])
                ax2.step(bin_centers, ratio_posteriormaximum, color=colors['br'])
                ax2.fill_between(bin_centers, ratio_samples - median_samples_std / median_target, ratio_samples + median_samples_std / median_target, color=colors['bl'], alpha=0.4, step='pre')
                ax2.axhline(1.0, color='black', linestyle='dashed')
                ax2.set_xlabel(keys_current[feature])
                ax2.set_ylabel('Ratio')
                if log_scales[feature] and False:
                    ax2.set_xscale('log')   
                ax2.set_ylim(0.95, 1.05)
                ax2.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.01))

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


    def plot_standard_deviations(self, name):

        r_truth = 10**self.test_targets.squeeze()
        r_pred = 10**self.samples.squeeze()
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

    def plot_GMM_weights(self, name):

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



def make_hist_1dim_ratio(data, labels, colors, showratios, ratioref, xlabel, rlabel, xrange, ticks=[[], []], logscales=[False, True],
                         nbins=100, integrals=[], legend=['upper right', 0.95, 0.95, None], atlas_info=[0.97, 0.03, 'three-lines', 'right', 'bottom', 'none'], switch_legend=False):

    if logscales[0]: 
        bins = np.logspace(np.log10(xrange[0]), np.log10(xrange[1]), nbins+1)
    else:            
        bins = np.linspace(         xrange[0],           xrange[1],  nbins+1)

    hists = []
    hists_err = []
    for dataset in data:
        if len(dataset.shape) == 1:
            hists.append(np.histogram(dataset, bins=bins)[0])
            hists_err.append(np.sqrt(hists[-1]))
        else:
            hists_model = np.array([np.histogram(model, bins=bins)[0] for model in dataset.T])
            hist_model_mean = np.mean(hists_model, axis=0)
            hist_model_std = np.std(hists_model, axis=0)
            hists.append(hist_model_mean)
            hists_err.append(hist_model_std)

    if len(integrals) == 0: 
        integrals =  [np.sum(y_avg) for y_avg in hists]
    scales    = [1/integral if integral != 0.0 else 1.0 for integral in integrals]

    fig, axs = plt.subplots(2, 1, figsize=figs_ratio, sharex=True, gridspec_kw={'height_ratios': [3,1]})
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_ratio); fig.subplots_adjust(hspace=0.07)
    atlas = atlas_info1 if atlas_info[2] == 'three-lines' else atlas_info2
    #axs[0].text(atlas_info[0], atlas_info[1], atlas, ha=atlas_info[3], va=atlas_info[4], transform=axs[0].transAxes, bbox=dict(boxstyle='round', fc=atlas_info[5], ec='none', alpha=0.75))

    steps = []; fills = []
    for i, y_avg, y_err, scale, color, label in zip(range(len(hists)), hists, hists_err, scales, colors, labels):

        step = axs[0].step(bins, scale*dup_last(y_avg), alpha=1.00, linewidth=1.00, where='post', color=color, label=label)
        axs[0].step(bins, scale*dup_last(y_avg-y_err),  alpha=0.50, linewidth=0.50, where='post', color=color)
        axs[0].step(bins, scale*dup_last(y_avg+y_err),  alpha=0.50, linewidth=0.50, where='post', color=color)
        fill = axs[0].fill_between(bins, scale*dup_last(y_avg-y_err), scale*dup_last(y_avg+y_err), alpha=0.20, step='post', facecolor=color)
        steps.append(step[0]); fills.append(fill)

        if showratios[i]:
            
            with np.errstate(all='ignore'):
                ratio_avg = get_ratio(scale*y_avg, scales[ratioref]*hists[ratioref])
                ratio_err = np.sqrt(get_ratio(y_err, y_avg)**2+get_ratio(hists_err[ratioref], hists[ratioref])**2)
            #ratio_nan = np.isnan(ratio_avg)
            #ratio_avg[ratio_nan] = 1.0
            #ratio_err[ratio_nan] = 0.0

            axs[1].step(bins, dup_last(ratio_avg),           alpha=1.00, linewidth=1.00, where='post', color=color)
            axs[1].step(bins, dup_last(ratio_avg-ratio_err), alpha=0.50, linewidth=0.50, where='post', color=color)
            axs[1].step(bins, dup_last(ratio_avg+ratio_err), alpha=0.50, linewidth=0.50, where='post', color=color)
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
    axs[0].set_ylabel('Relative entries')
    axs[1].set_ylabel(rlabel, loc='center')
    axs[0].set_xlim((bins[0], bins[-1]))
    axs[1].set_ylim((0.8, 1.2))
    axs[1].set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
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
                   atlas_info=[0.97, 0.03, 'three-lines', 'right', 'bottom', 'none'], showdiag=True):

    if logscales[0]: xbins = np.logspace(np.log10(ranges[0][0]), np.log10(ranges[0][1]), nbins[0]+1)
    else:            xbins = np.linspace(         ranges[0][0],           ranges[0][1],  nbins[0]+1)
    if logscales[1]: ybins = np.logspace(np.log10(ranges[1][0]), np.log10(ranges[1][1]), nbins[0]+1)
    else:            ybins = np.linspace(         ranges[1][0],           ranges[1][1],  nbins[1]+1)

    fig, ax = plt.subplots(1, 1, figsize=figs_hist2d)
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_hist2d)
    #atlas = atlas_info1 if atlas_info[2] == 'three-lines' else atlas_info2
    #ax.text(atlas_info[0], atlas_info[1], atlas, ha=atlas_info[3], va=atlas_info[4], transform=ax.transAxes, bbox=dict(boxstyle='round', fc=atlas_info[5], ec='none', alpha=0.75))
    #hist = ax.hist2d(data[0], data[1], weights=np.zeros_like(data[0]) + 1.0 / data[0].size, bins=[xbins, ybins], norm=mpl.colors.LogNorm(), rasterized=True)
    if len(data[0].shape) == 1:
        hist, x_edges, y_edges = np.histogram2d(data[0], data[1], bins=[xbins, ybins], weights=np.full_like(data[0], 1.0 / data[0].size))
    else:
        hists = []
        for i in range(data[0].shape[1]):
            hist, x_edges, y_edges = np.histogram2d(data[0][:, i], data[1][:, i], bins=[xbins, ybins], weights=np.full_like(data[0][:, i], 1.0 / data[0][:, i].size))
            hists.append(hist)
        hist = np.array(hists).mean(axis=0)
    hist /= hist.sum(axis=1, keepdims=True)
    mesh = ax.pcolormesh(x_edges, y_edges, hist.T, norm=mpl.colors.LogNorm(), rasterized=True)
    if showdiag: 
        ax.plot([ranges[0][0], ranges[0][1]], [ranges[1][0], ranges[1][1]], linestyle='dashed', color='black')
    divider = make_axes_locatable(ax)
    cax  = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label('Relative entries', loc='center')
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
