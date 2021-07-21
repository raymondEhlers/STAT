'''
Class to steer Bayesian analysis and produce plots.
'''

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy
import statistics

import os
import sys
import pickle
import argparse

from src.design import Design
from src import emulator, mcmc, init

import run_analysis_base

################################################################
class MergeResults(run_analysis_base.RunAnalysisBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, config_file, model, output_dir, **kwargs):

        # Initialize base class
        super(MergeResults, self).__init__(config_file, model, output_dir, **kwargs)
        
        self.output_dir_holdout = os.path.join(self.output_dir_base, '{}/holdout'.format(model))
        self.plot_dir = os.path.join(self.output_dir_base, model)
        
    #---------------------------------------------------------------
    # Run analysis
    #---------------------------------------------------------------
    def run_analysis(self):
  
        # Initialize data and model from files
        self.initialize()
        
        # Initialize pickled config settings
        init.Init(self.workdir).Initialize(self)
        
        # Emulator validation: Store lists of true RAA, emulator RAA at each holdout point
        SystemCount = len(self.AllData["systems"])
        true_raa_aggregated = [[] for i in range(SystemCount)]
        emulator_raa_mean_aggregated = [[] for i in range(SystemCount)]
        emulator_raa_stdev_aggregated = [[] for i in range(SystemCount)]

        # Store a list of the chi2 of the holdout residual
        self.avg_residuals = []

        # Store list of closure test result
        T_qhat_closure_result_list = []
        T_qhat_closure_result_list2 = []
        T_qhat_closure_truth_list = []
        E_qhat_closure_result_list = []
        E_qhat_closure_result_list2 = []
        E_qhat_closure_truth_list = []
        theta_closure_list = []
        theta_closure_result_dict = {}
        theta_closure_result2_dict = {}
        for name in self.Names:
            theta_closure_result_dict[name] = []
            theta_closure_result2_dict[name] = []
    
        n_design_points = len(next(os.walk(self.output_dir_holdout))[1])
        print('iterating through {} results'.format(n_design_points))
        for i in range(0, n_design_points):
        
            # Load pkl file of results
            result_path = os.path.join(self.output_dir_holdout, '{}/result.pkl'.format(i))
            if not os.path.exists(result_path):
                print('Warning: {} does not exist'.format(result_path))
                continue
            with open(result_path, 'rb') as f:

                result_dict = pickle.load(f)

                # Holdout test
                true_raa = result_dict['true_raa']
                emulator_raa_mean = result_dict['emulator_raa_mean']
                emulator_raa_stdev = result_dict['emulator_raa_stdev']
                
                [true_raa_aggregated[i].append(true_raa[i]) for i in range(SystemCount)]
                [emulator_raa_mean_aggregated[i].append(emulator_raa_mean[i]) for i in range(SystemCount)]
                [emulator_raa_stdev_aggregated[i].append(emulator_raa_stdev[i]) for i in range(SystemCount)]

                
                # Closure test
                
                # qhat vs T
                T_array = result_dict['T_array']
                T_qhat_truth = result_dict['T_qhat_truth']
                T_qhat_mean = result_dict['T_qhat_mean']
                T_qhat_closure = result_dict['T_qhat_closure']
                T_qhat_closure2 = result_dict['T_qhat_closure2']
                T_qhat_closure_result_list.append(T_qhat_closure)
                T_qhat_closure_result_list2.append(T_qhat_closure2)
                T_qhat_closure_truth_list.append(T_qhat_truth)
                
                # qhat vs E
                E_array = result_dict['E_array']
                E_qhat_truth = result_dict['E_qhat_truth']
                E_qhat_mean = result_dict['E_qhat_mean']
                E_qhat_closure = result_dict['E_qhat_closure']
                E_qhat_closure2 = result_dict['E_qhat_closure2']
                E_qhat_closure_result_list.append(E_qhat_closure)
                E_qhat_closure_result_list2.append(E_qhat_closure2)
                E_qhat_closure_truth_list.append(E_qhat_truth)

                # ABCD closure
                theta = result_dict['theta']
                theta_closure_list.append(theta)
                for name in self.Names:
                    theta_closure_result_dict[name].append(result_dict['{}_closure'.format(name)])
                    theta_closure_result2_dict[name].append(result_dict['{}_closure2'.format(name)])

        # Plot summary of holdout tests
        #self.plot_avg_residuals()
        self.plot_emulator_validation(true_raa_aggregated, emulator_raa_mean_aggregated, emulator_raa_stdev_aggregated)
        
        self.plot_emulator_uncertainty_validation(true_raa_aggregated, emulator_raa_mean_aggregated, emulator_raa_stdev_aggregated)

        # Plot summary of qhat closure tests
        self.plot_closure_summary_qhat(T_array, T_qhat_closure_result_list,
                                       T_qhat_closure_truth_list, type='T', CR='90')
        self.plot_closure_summary_qhat(T_array, T_qhat_closure_result_list2,
                                       T_qhat_closure_truth_list, type='T', CR='60')
        self.plot_closure_summary_qhat(E_array, E_qhat_closure_result_list,
                                       E_qhat_closure_truth_list, type='E', CR='90')
        self.plot_closure_summary_qhat(E_array, E_qhat_closure_result_list2,
                                       E_qhat_closure_truth_list, type='E', CR='60')
                                       
        # Print theta closure summary
        for i,name in enumerate(self.Names):
            self.plot_closure_summary_theta(i, name, theta_closure_list, theta_closure_result_dict, CR='90')
            self.plot_closure_summary_theta(i, name, theta_closure_list, theta_closure_result2_dict, CR='60')

    #---------------------------------------------------------------
    # Plot summary of closure tests
    #
    # theta_closure_list is a list (per design point) of theta values
    #
    # theta_closure_result_dict is a dictionary (per ABCD) of lists (per design point)
    # [{A: [True, True, ...]}, {B: [True, False,  ...]}, ... ]
    #
    #---------------------------------------------------------------
    def plot_closure_summary_theta(self, i, name, theta_closure_list, theta_closure_result_dict, CR='90'):

        theta_i_list = [theta[i] for theta in  theta_closure_list]
        qhat_list = [self.qhat(T=0.3, E=100, parameters=theta) for theta in theta_closure_list]
        success_list = theta_closure_result_dict[name]

        # Construct 2D histogram of qhat vs theta[i],
        # where amplitude is fraction of successful closure tests

        theta_i_range = self.ranges_transformed[i]
        xbins = np.linspace(theta_i_range[0], theta_i_range[1], num=8)
        xwidth = (theta_i_range[0]+theta_i_range[1])/(7*2)

        ybins =  [0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15]
        ybins_center =  [(ybins[i+1]+ybins[i])/2 for i in range(len(ybins)-1)]

        x = np.array(theta_i_list)
        y = np.array(qhat_list)
        z = np.array(success_list)

        # Histogram of fraction of successes
        self.N_per_bin = 1
        H, xedges, yedges, binnumber= scipy.stats.binned_statistic_2d(x, y, z, statistic=np.mean,
                                                                      bins=[xbins, ybins])
        XX, YY = np.meshgrid(xedges, yedges)

        fig = plt.figure(figsize = (11,9))
        ax1=plt.subplot(111)
        plot1 = ax1.pcolormesh(XX, YY, H.T)
        fig.colorbar(plot1, ax=ax1)

        # Histogram of efficiency uncertainty
        Herr, xedges, yedges, binnumber= scipy.stats.binned_statistic_2d(x, y, z,
                                                                         statistic=self.efficiency_uncertainty_bayesian,
                                                                         bins=[xbins, ybins])

        plt.xlabel(name, size=14)
        plt.ylabel(r'$\left< \hat{q}/T^3 \right>_{T=300\;\rm{MeV}, E=100\;\rm{GeV}}$', size=14)
        plt.title('Fraction of closure tests contained in {}% CR'.format(CR), size=14)

        mean = np.mean(z)
        self.N_per_bin = 1
        unc = self.efficiency_uncertainty_bayesian(z)
        ax1.legend(title='mean: {:0.2f}{}{:0.2f}'.format(mean, r'$\pm$', unc),
                   title_fontsize=14, loc='upper right')
            
        for i in range(len(xbins)-1):
            for j in range(len(ybins)-1):
                zval = H[i][j]
                zerr = Herr[i][j]
                if np.isnan(zval) or np.isnan(zerr):
                    continue
                ax1.text(xbins[i]+xwidth, ybins_center[j], '{:0.2f}{}{:0.2f}'.format(zval, r'$\pm$',zerr),
                         size=8, ha='center', va='center',
                         bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

        # Save
        plt.savefig('{}/Closure_Summary2D_{}_{}.pdf'.format(self.plot_dir, name, CR), dpi = 192)
        plt.close('all')

    #---------------------------------------------------------------
    # Plot summary of closure tests
    #
    # qhat_closure_result_list is a list (per design point) of lists (of T values)
    # [ [True, True, ...], [True, False,  ...], ... ] where each sublist is a given design point
    
    # qhat_closure_truth_list is a list (per design point) of lists (of T values)
    # [ [qhat_T1, qhat_T2, ...], [qhat_T1, qhat_T2, ...], ... ] where each sublist is a given design point
    #---------------------------------------------------------------
    def plot_closure_summary_qhat(self, x_array, qhat_closure_result_list,
                                  qhat_closure_truth_list, type='T', CR='90'):
        
        # Construct 2D histogram of <qhat of design point> vs T,
        # where amplitude is fraction of successful closure tests
        
        # For each T and design point, compute <qhat of design point>,
        # T, and the fraction of successful closure tests
        x_list = []
        qhat_mean_list = []
        success_list = []
        for i,x in enumerate(x_array):
            for j,design in enumerate(qhat_closure_result_list):
            
                qhat_mean = statistics.mean(qhat_closure_truth_list[j])
                success = qhat_closure_result_list[j][i]
                
                x_list.append(x)
                qhat_mean_list.append(qhat_mean)
                success_list.append(success)
                
        # Now draw the mean success rate in 2D
        if type is 'T':
            xbins = np.linspace(0.15, 0.5, num=8)
            xwidth = 0.025
            self.N_per_bin = 50/7 # We have multiple T points per bin
        if type is 'E':
            xbins = np.linspace(20, 200, num=10)
            xwidth = 10
            self.N_per_bin = 50/9  # We have multiple E points per bin

        ybins =  [0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15]
        ybins_center =  [(ybins[i+1]+ybins[i])/2 for i in range(len(ybins)-1)]

        x = np.array(x_list)
        y = np.array(qhat_mean_list)
        z = np.array(success_list)
        
        # Histogram of fraction of successes
        H, xedges, yedges, binnumber= scipy.stats.binned_statistic_2d(x, y, z, statistic=np.mean,
                                                                      bins=[xbins, ybins])
        H = np.ma.masked_invalid(H) # masking where there was no data
        XX, YY = np.meshgrid(xedges, yedges)

        fig = plt.figure(figsize = (11,9))
        ax1=plt.subplot(111)
        plot1 = ax1.pcolormesh(XX, YY, H.T)
        fig.colorbar(plot1, ax=ax1)
        
        # Histogram of binomial uncertainty
        Herr, xedges, yedges, binnumber= scipy.stats.binned_statistic_2d(x, y, z,
                                                                         statistic=self.efficiency_uncertainty_bayesian,
                                                                         bins=[xbins, ybins])
        Herr = np.ma.masked_invalid(Herr)
        
        plt.xlabel('{} (GeV)'.format(type), size=14)
        if type is 'T':
            plt.ylabel(r'$\left< \hat{q}/T^3 \right>_{E=100\;\rm{GeV}}$', size=14)
        if type is 'E':
            plt.ylabel(r'$\left< \hat{q}/T^3 \right>_{T=300\;\rm{MeV}}$', size=14)
        plt.title('Fraction of closure tests contained in {}% CR'.format(CR), size=14)
        
        mean = np.mean(z)
        self.N_per_bin = 50 # Here, we take just one point per curve
        unc = self.efficiency_uncertainty_bayesian(z)
        ax1.legend(title='mean: {:0.2f}{}{:0.2f}'.format(mean, r'$\pm$', unc),
                   title_fontsize=14, loc='upper right')
            
        for i in range(len(xbins)-1):
            for j in range(len(ybins)-1):
                zval = H[i][j]
                zerr = Herr[i][j]
                if np.isnan(zval) or np.isnan(zerr):
                    continue
                ax1.text(xbins[i]+xwidth, ybins_center[j], '{:0.2f}{}{:0.2f}'.format(zval, r'$\pm$',zerr),
                         size=8, ha='center', va='center',
                         bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
       
        # Save
        plt.savefig('{}/Closure_Summary2D_{}_{}.pdf'.format(self.plot_dir, type, CR), dpi = 192)
        plt.close('all')

    #---------------------------------------------------------------
    # Compute binomial uncertainty from a list of True/False values
    # [True, True, False, True, ...]
    #---------------------------------------------------------------
    def efficiency_uncertainty_binomial(self, success_list):
    
        length = len(success_list)
        sum = np.sum(success_list)
        mean = 1.*sum/length
        
        # We have multiple T points per bin, which would underestimate the uncertainty
        # since neighboring points are highly correlated
        real_length = length / self.N_per_bin
        
        variance = real_length*mean*(1-mean)
        sigma = np.sqrt(variance)

        return sigma/real_length
        
    #---------------------------------------------------------------
    # Compute bayesian uncertainty on efficiency from a list of True/False values
    # [True, True, False, True, ...]
    # http://phys.kent.edu/~smargeti/STAR/D0/Ullrich-Errors.pdf
    #---------------------------------------------------------------
    def efficiency_uncertainty_bayesian(self, success_list):
    
        length = len(success_list)
        sum = np.sum(success_list)
        mean = 1.*sum/length
        
        # We have multiple T points per bin, which would underestimate the uncertainty
        # since neighboring points are highly correlated
        real_length = length / self.N_per_bin
        
        k = mean*real_length
        n = real_length
        variance = (k+1)*(k+2)/((n+2)*(n+3)) - (k+1)*(k+1)/((n+2)*(n+2))
        return np.sqrt(variance)

    #---------------------------------------------------------------
    # Plot emulator validation
    #
    # true_raa and emulator_raa are lists (per system) of lists (per design point) of lists
    # e.g. true_raa[i] = [[RAA_0, RAA_1,...], [RAA_0, RAA_1, ...], ...]
    #
    #---------------------------------------------------------------
    def plot_emulator_validation(self, true_raa, emulator_raa_mean, emulator_raa_stdev):

        # Loop through emulators
        for cent in range(0,2):
        
            # Construct a figure with two plots
            plt.figure(1, figsize=(10, 6))
            ax_scatter = plt.axes([0.1, 0.13, 0.6, 0.8]) # [left, bottom, width, height]
            ax_residual = plt.axes([0.81, 0.13, 0.15, 0.8])
            
            markers = ['o', 's', 'D']
            
            SystemCount = len(self.AllData["systems"])
            for i in range(SystemCount):
            
                system = self.AllData['systems'][i]
                if 'AuAu' in system:
                    if cent == 0:
                        system_label = 'Au-Au \;200\; GeV, 0-10\%'
                    if cent == 1:
                        system_label = 'Au-Au \;200\; GeV, 40-50\%'
                else:
                    if '2760' in system:
                        if cent == 0:
                            system_label = 'Pb-Pb \;2.76\; TeV, 0-5\%'
                        if cent == 1:
                            system_label = 'Pb-Pb \;2.76\; TeV, 30-40\%'

                    elif '5020' in system:
                        if cent == 0:
                            system_label = 'Pb-Pb \;5.02\; TeV, 0-10\%'
                        if cent == 1:
                            system_label = 'Pb-Pb \;5.02\; TeV, 30-50\%'

                #color = sns.color_palette('colorblind')[i]
                color = self.colors[i]
                
                # Optionally: Remove outlier points from emulator validation plot
                remove_outliers = False
                if remove_outliers:
                    if self.model == 'LBT':
                        remove = [79, 124, 135]
                    if self.model == 'MATTER':
                        remove = [59, 60, 61, 62]
                    if self.model == 'MATTER+LBT1':
                        remove = [0, 2, 5, 12, 17, 28, 31, 34, 37, 46, 50, 56, 63, 65, 69]
                    if self.model == 'MATTER+LBT2':
                        remove = [2, 3, 14, 19, 20, 21, 27, 28, 33, 56]
                    for index in sorted(remove, reverse=True):
                        del true_raa[i][index]
                        del emulator_raa_mean[i][index]
                        del emulator_raa_stdev[i][index]

                true_raa_flat_i = [item for sublist in true_raa[i] for item in sublist[cent]]
                emulator_raa_mean_flat_i = [item for sublist in emulator_raa_mean[i] for item in sublist[cent]]
                emulator_raa_stdev_flat_i = [item for sublist in emulator_raa_stdev[i] for item in sublist[cent]]

                # Get RAA points
                true_raa_i = np.array(true_raa_flat_i)
                emulator_raa_mean_i = np.array(emulator_raa_mean_flat_i)
                emulator_raa_stdev_i = np.array(emulator_raa_stdev_flat_i)
                normalized_residual_i = np.divide(true_raa_i-emulator_raa_mean_i, emulator_raa_stdev_i)

                # Draw scatter plot
                ax_scatter.scatter(true_raa_i, emulator_raa_mean_i, s=5, marker=markers[i],
                                   color=color, alpha=0.7, label=r'$\rm{{{}}}$'.format(system_label), linewidth=0)
                #ax_scatter.set_ylim([0, 1.19])
                #ax_scatter.set_xlim([0, 1.19])
                ax_scatter.set_xlabel(r'$R_{\rm{AA}}^{\rm{true}}$', fontsize=20)
                ax_scatter.set_ylabel(r'$R_{\rm{AA}}^{\rm{emulator}}$', fontsize=20)
                ax_scatter.legend(title=self.model, title_fontsize=16,
                                  loc='upper left', fontsize=14, markerscale=5)
                plt.setp(ax_scatter.get_xticklabels(), fontsize=14)
                plt.setp(ax_scatter.get_yticklabels(), fontsize=14)
                                  
                # Draw line with slope 1
                ax_scatter.plot([0,1], [0,1], sns.xkcd_rgb['almost black'], alpha=0.3,
                                linewidth=3, linestyle='--')
                
                # Print mean value of emulator uncertainty
                stdev_mean_relative = np.divide(emulator_raa_stdev_i, true_raa_i)
                stdev_mean = np.mean(stdev_mean_relative)
                text = r'$\left< \sigma_{{\rm{{emulator}}}}^{{\rm{{{}}}}} \right> = {:0.1f}\%$'.format(system_label, 100*stdev_mean)
                ax_scatter.text(0.4, 0.17-0.09*i, text, fontsize=16)
              
                # Draw normalization residuals
                max = 3
                bins = np.linspace(-max, max, 30)
                x = (bins[1:] + bins[:-1])/2
                h = ax_residual.hist(normalized_residual_i, color=color, histtype='step',
                                 orientation='horizontal', linewidth=3, alpha=0.8, density=True, bins=bins)
                ax_residual.scatter(h[0], x, color=color, s=10, marker=markers[i])
                ax_residual.set_ylabel(r'$\left(R_{\rm{AA}}^{\rm{true}} - R_{\rm{AA}}^{\rm{emulator}}\right) / \sigma_{\rm{emulator}}$',
                                       fontsize=20)
                plt.setp(ax_residual.get_xticklabels(), fontsize=14)
                plt.setp(ax_residual.get_yticklabels(), fontsize=14)
                                       
                # Print out indices of points that deviate significantly
                if remove_outliers:
                    stdev = np.std(normalized_residual_i)
                    for j,true_sublist in enumerate(true_raa[i]):
                        emulator_sublist = emulator_raa_mean[i][j]
                        for k,true_raa_value in enumerate(true_sublist):
                            emulator_raa_value = emulator_sublist[k]
                            normalized_residual = (true_raa_value-emulator_raa_value)/true_raa_value
                            if np.abs(normalized_residual) > 3*stdev:
                                print('Index {} has poor  emulator validation...'.format(j))
                  
            plt.savefig('{}/EmulatorValidation_{}.pdf'.format(self.plot_dir, cent))
            plt.close('all')

    #---------------------------------------------------------------
    # Plot emulator uncertainty validation
    #
    # true_raa and emulator_raa are lists (per system) of lists (per design point) of lists
    # e.g. true_raa[i] = [[RAA_0, RAA_1,...], [RAA_0, RAA_1, ...], ...]
    #
    #---------------------------------------------------------------
    def plot_emulator_uncertainty_validation(self, true_raa, emulator_raa_mean, emulator_raa_stdev):

        # Loop through emulators
        for cent in range(0,2):
        
            # Construct a figure with two plots
            plt.figure(1, figsize=(10, 6))
            ax_scatter = plt.axes([0.1, 0.13, 0.6, 0.8]) # [left, bottom, width, height]
            ax_residual = plt.axes([0.81, 0.13, 0.15, 0.8])
            
            SystemCount = len(self.AllData["systems"])
            for i in range(SystemCount):
            
                system = self.AllData['systems'][i]
                if 'AuAu' in system:
                    if cent == 0:
                        system_label = 'Au-Au \;200\; GeV, 0-10\%'
                    if cent == 1:
                        system_label = 'Au-Au \;200\; GeV, 40-50\%'
                else:
                    if '2760' in system:
                        if cent == 0:
                            system_label = 'Pb-Pb \;2.76\; TeV, 0-5\%'
                        if cent == 1:
                            system_label = 'Pb-Pb \;2.76\; TeV, 30-40\%'

                    elif '5020' in system:
                        if cent == 0:
                            system_label = 'Pb-Pb \;5.02\; TeV, 0-10\%'
                        if cent == 1:
                            system_label = 'Pb-Pb \;5.02\; TeV, 30-50\%'

                #color = sns.color_palette('colorblind')[i]
                color = self.colors[i]
                
                # Optionally: Remove outlier points from emulator validation plot
                remove_outliers = False
                if remove_outliers:
                    if self.model == 'LBT':
                        remove = [79, 124, 135]
                    if self.model == 'MATTER':
                        remove = [59, 60, 61, 62]
                    if self.model == 'MATTER+LBT1':
                        remove = [0, 2, 5, 12, 17, 28, 31, 34, 37, 46, 50, 56, 63, 65, 69]
                    if self.model == 'MATTER+LBT2':
                        remove = [2, 3, 14, 19, 20, 21, 27, 28, 33, 56]
                    for index in sorted(remove, reverse=True):
                        del true_raa[i][index]
                        del emulator_raa_mean[i][index]
                        del emulator_raa_stdev[i][index]

                true_raa_flat_i = [item for sublist in true_raa[i] for item in sublist[cent]]
                emulator_raa_mean_flat_i = [item for sublist in emulator_raa_mean[i] for item in sublist[cent]]
                emulator_raa_stdev_flat_i = [item for sublist in emulator_raa_stdev[i] for item in sublist[cent]]

                # Get RAA points
                true_raa_i = np.array(true_raa_flat_i)
                emulator_raa_mean_i = np.array(emulator_raa_mean_flat_i)
                emulator_raa_stdev_i = np.array(emulator_raa_stdev_flat_i)
                normalized_residual_i = np.divide(true_raa_i-emulator_raa_mean_i, emulator_raa_stdev_i)

                # Draw scatter plot
                ax_scatter.scatter(true_raa_i, emulator_raa_stdev_i, s=5,
                                   color=color, alpha=0.7, label=r'$\rm{{{}}}$'.format(system_label), linewidth=0)
                #ax_scatter.set_ylim([0, 1.19])
                #ax_scatter.set_xlim([0, 1.19])
                ax_scatter.set_xlabel(r'$R_{\rm{AA}}^{\rm{true}}$', fontsize=18)
                ax_scatter.set_ylabel(r'$\sigma_{\rm{emulator}}$', fontsize=18)
                ax_scatter.legend(title=self.model, title_fontsize=16,
                                  loc='upper left', fontsize=14, markerscale=5)
              
                # Draw normalization residuals
                max = 3
                bins = np.linspace(-max, max, 30)
                ax_residual.hist(normalized_residual_i, color=color, histtype='step',
                                 orientation='horizontal', linewidth=3, alpha=0.8, density=True, bins=bins)
                ax_residual.set_ylabel(r'$\left(R_{\rm{AA}}^{\rm{true}} - R_{\rm{AA}}^{\rm{emulator}}\right) / \sigma_{\rm{emulator}}$',
                                       fontsize=16)
                                       
                # Print out indices of points that deviate significantly
                if remove_outliers:
                    stdev = np.std(normalized_residual_i)
                    for j,true_sublist in enumerate(true_raa[i]):
                        emulator_sublist = emulator_raa_mean[i][j]
                        for k,true_raa_value in enumerate(true_sublist):
                            emulator_raa_value = emulator_sublist[k]
                            normalized_residual = (true_raa_value-emulator_raa_value)/true_raa_value
                            if np.abs(normalized_residual) > 3*stdev:
                                print('Index {} has poor  emulator validation...'.format(j))
                  
            plt.savefig('{}/EmulatorUncertaintyValidation_{}.pdf'.format(self.plot_dir, cent))
            plt.close('all')

##################################################################
if __name__ == '__main__':
  
    # Define arguments
    parser = argparse.ArgumentParser(description='Jetscape STAT analysis')
    parser.add_argument('-o', '--outputdir', action='store',
                        type=str, metavar='outputdir',
                        default='./STATGallery')
    parser.add_argument('-c', '--configFile', action='store',
                        type=str, metavar='configFile',
                        default='analysis_config.yaml',
                        help='Path of config file')
    parser.add_argument('-m', '--model', action='store',
                        type=str, metavar='model',
                        default='LBT',
                        help='model')

    # Parse the arguments
    args = parser.parse_args()
    
    print('')
    print('Configuring MergeResults...')
    
    # If invalid configFile is given, exit
    if not os.path.exists(args.configFile):
        print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
        sys.exit(0)
    
    analysis = MergeResults(config_file = args.configFile, model=args.model,
                            output_dir=args.outputdir)
    analysis.run_model()
