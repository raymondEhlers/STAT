'''
Class to steer Bayesian analysis and produce plots.
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pymc3

import os
import subprocess
import sys
import pickle
import argparse

from src.design import Design
from src import emulator, mcmc, init

import run_analysis_base

################################################################
class RunAnalysis(run_analysis_base.RunAnalysisBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, config_file, model, output_dir, exclude_index, **kwargs):

    # Initialize base class
    super(RunAnalysis, self).__init__(config_file, model, output_dir, exclude_index, **kwargs)

    # Write dictionary of results to pickle
    self.output_dict = {}

  #---------------------------------------------------------------
  # Run analysis
  #---------------------------------------------------------------
  def run_analysis(self):

    # Initialize data and model from files
    self.initialize()

    # Initialize pickled config settings
    init.Init(self.workdir).Initialize(self)

    # If exclude_index < 0, perform standard analysis
    if self.exclude_index < 0:
      self.run_single_analysis()

    # Otherwise, hold out a specific training point from the emulator training
    else:

      # Check if exclude_index exists
      n_design_points = len(self.AllData['design'])
      if self.exclude_index > n_design_points-1:
        print('Design point {} does not exist for {}, which has {} design points'.format(self.exclude_index, self.model, n_design_points))
        os.system('rm -r {}'.format(self.workdir))
        return

      # For each emulator:
      # Store lists of true RAA, emulator RAA at each holdout point
      # (For each system+centrality -- over all pt)
      self.SystemCount = len(self.AllData["systems"])
      self.true_raa = [[[] for _ in range(0, 2)] for _ in range(self.SystemCount)]
      self.emulator_raa_mean = [[[] for _ in range(0, 2)] for _ in range(self.SystemCount)]
      self.emulator_raa_stdev = [[[] for _ in range(0, 2)] for _ in range(self.SystemCount)]

      # Initialize data structures, with the updated holdout information
      print('Running holdout test {} / {}'.format(self.exclude_index, n_design_points))
      self.initialize(exclude_index = self.exclude_index)

      # Transform holdout coordinates
      self.holdout_design = self.AllData['holdout_design']
      if self.model in ['MATTER+LBT1', 'MATTER+LBT2']:
          holdout_design_temp = np.copy(self.holdout_design)
          holdout_design_temp[0] = self.holdout_design[0] * self.holdout_design[1]
          holdout_design_temp[1] = self.holdout_design[0] - self.holdout_design[0] * self.holdout_design[1]
          self.holdout_design = holdout_design_temp
      self.output_dict['theta'] = self.holdout_design
      print('theta: {}'.format(self.AllData['holdout_design']))
      print('theta_transformed: {}'.format(self.holdout_design))

      if len(self.AllData['design']) != n_design_points - 1:
        sys.exit('Only {} design points remain, but there should be {}!'.format(
                  len(self.AllData['design']), n_design_points - 1))

      # Perform analysis (with holdout and closure tests)
      self.run_single_analysis(holdout_test=True, closure_test=True)

      plt.close('all')

  #---------------------------------------------------------------
  # Run analysis
  #---------------------------------------------------------------
  def run_single_analysis(self, holdout_test = False, closure_test = False):

    # Create output dir
    self.plot_dir = os.path.join(self.workdir, 'plots')
    if not os.path.exists(self.plot_dir):
      os.makedirs(self.plot_dir)

    # Re-train emulator, if requested
    if self.retrain_emulator:
      # Clean cache for emulator
      for system in self.AllData["systems"]:
          if os.path.exists(os.path.join(self.cache_dir, '{}.pkl'.format(system))):
              os.remove(os.path.join(self.cache_dir, '{}.pkl'.format(system)))
              print('removed {}'.format('{}/{}.pkl'.format(self.cache_dir, system)))

      # Re-train emulator
      #import IPython; IPython.embed()
      subprocess.run(
        #["python3", "-m src.emulator", "--retrain", f"--npc {self.n_pc}", f"--nrestarts {self.n_restarts}", f"--alpha {self.alpha}", f"-o {self.workdir}"],
        f"python3 -m src.emulator --retrain --npc {self.n_pc} --nrestarts {self.n_restarts} --alpha {self.alpha} -o {self.workdir}",
        shell=True, check=True,
      )

    # Load trained emulator
    self.EmulatorAuAu200 = emulator.Emulator.from_cache('AuAu200', self.workdir)
    self.EmulatorPbPb2760 = emulator.Emulator.from_cache('PbPb2760', self.workdir)
    self.EmulatorPbPb5020 = emulator.Emulator.from_cache('PbPb5020', self.workdir)

    # Construct plots characterizing the emulator
    self.plot_design(holdout_test = holdout_test)
    self.plot_RAA(self.AllData["design"], 'Design')

    if holdout_test:
      self.plot_emulator_RAA_residuals(holdout_test = True)
      if not closure_test:
        return
    else:
      self.plot_emulator_RAA_residuals()

    # Run MCMC
    if self.rerun_mcmc:
      if os.path.exists(os.path.join(self.cache_dir, 'mcmc_chain.hdf')):
        print('removed mcmc_chain.hdf')
        os.remove(os.path.join(self.cache_dir, 'mcmc_chain.hdf'))
      os.system('python -m src.mcmc --nwalkers {} --nburnsteps {} -o {} {} '.format(self.n_walkers, self.n_burn_steps, self.workdir, self.n_steps))

    # Load MCMC chain
    self.chain = mcmc.Chain(self.workdir)
    self.MCMCSamples = self.chain.load()

    # Plot dependence of MC sampling on number of steps
    self.plot_MCMC_samples()

    # Transform coordinates
    if self.model in ['MATTER+LBT1', 'MATTER+LBT2']:
      self.TransformedSamples = np.copy(self.MCMCSamples)
      self.TransformedSamples[:,0] = self.MCMCSamples[:,0] * self.MCMCSamples[:,1]
      self.TransformedSamples[:,1] = self.MCMCSamples[:,0] - self.MCMCSamples[:,0] * self.MCMCSamples[:,1]
    else:
      self.TransformedSamples = np.copy(self.MCMCSamples)

    # Plot posterior distributions of parameters
    self.plot_correlation(suffix = '', holdout_test = holdout_test)
    if self.model in  ['MATTER+LBT1', 'MATTER+LBT2']:
      self.plot_correlation(suffix = '_Transformed', holdout_test = holdout_test)

    # Plot RAA for samples of the posterior parameter space
    sample_points = self.MCMCSamples[ np.random.choice(range(len(self.MCMCSamples)), 100), :]
    self.plot_RAA(sample_points, 'Posterior')

    if not holdout_test and not closure_test:
        self.plot_qhat(E=100.)
        self.plot_qhat(T=0.3)

    plt.close('all')

    # Write result to pkl
    if holdout_test:
      self.output_dict['true_raa'] = self.true_raa
      self.output_dict['emulator_raa_mean'] = self.emulator_raa_mean
      self.output_dict['emulator_raa_stdev'] = self.emulator_raa_stdev

    # Plot qhat/T^3 for the holdout point
    if closure_test:
      self.plot_closure_test_qhat(E=100.)
      self.plot_closure_test_qhat(T=0.3)

    # Write result to pkl
    with open(os.path.join(self.workdir, 'result.pkl'), 'wb') as f:
      pickle.dump(self.output_dict, f)

    plt.close('all')

  #---------------------------------------------------------------
  # Plot qhat/T^3 for the holdout point
  #---------------------------------------------------------------
  def plot_qhat(self, E=None, T=None):

    # Plot 90% credible interval of qhat solution
    # --> Construct distribution of qhat by sampling each ABCD point
    if E:
        xlabel = 'T (GeV)'
        x_array = np.linspace(0.16, 0.5)
        qhat_posteriors = [[self.qhat(T=T, E=E, parameters=parameters)
                            for parameters in self.TransformedSamples]
                            for T in x_array]

    if T:
        xlabel = 'E (GeV)'
        x_array = np.linspace(5, 200)
        qhat_posteriors = [[self.qhat(T=T, E=E, parameters=parameters)
                            for parameters in self.TransformedSamples]
                            for E in x_array]

    # Get list of mean qhat values for each T or E
    qhat_mean = [np.mean(qhat_values) for qhat_values in qhat_posteriors]
    plt.plot(x_array, qhat_mean, sns.xkcd_rgb['denim blue'],
             linewidth=2., linestyle='--', label='Mean')
    plt.xlabel(xlabel)
    plt.ylabel(r'$\hat{q}/T^3$')

    ymin = 0
    ymax = 2*max(qhat_mean)
    axes = plt.gca()
    axes.set_ylim([ymin, ymax])

    # Get credible interval for each T or E
    # Specifically: highest posterior density interval (HPDI) via pymc3
    h = [pymc3.stats.hpd(np.array(qhat_values), self.confidence[0]) for qhat_values in qhat_posteriors]
    credible_low = [i[0] for i in h]
    credible_up =  [i[1] for i in h]
    plt.fill_between(x_array, credible_low, credible_up, color=sns.xkcd_rgb['light blue'],
                     label='{}% Credible Interval'.format(int(self.confidence[0]*100)))

    # Draw legend
    first_legend = plt.legend(title=self.model, title_fontsize=15,
                             loc='upper right', fontsize=12)
    ax = plt.gca().add_artist(first_legend)

    if E:
        label = 'T'
    if T:
        label = 'E'
    plt.savefig('{}/qhat_{}.pdf'.format(self.plot_dir, label), dpi = 192)
    plt.close('all')

  #---------------------------------------------------------------
  # Plot qhat/T^3 for the holdout point
  #---------------------------------------------------------------
  def plot_closure_test_qhat(self, E=None, T=None):

    # Plot 90% credible interval of qhat solution
    # --> Construct distribution of qhat by sampling each ABCD point

    # Plot 1D closure tests for qhat vs. T, for fixed E
    if E:
        xlabel = 'T (GeV)'
        x_array = np.linspace(0.16, 0.5)

        # Plot truth value
        qhat_truth = [self.qhat(T=T, E=E, parameters=self.holdout_design) for T in x_array]
        plt.plot(x_array, qhat_truth, sns.xkcd_rgb['pale red'],
                 linewidth=2., label='Truth')

        # Plot 90% credible interval of qhat solution
        # --> Construct distribution of qhat by sampling each ABCD point
        qhat_posteriors = [[self.qhat(T=T, E=E, parameters=parameters)
                            for parameters in self.TransformedSamples]
                            for T in x_array]

    # Plot 1D closure tests for qhat vs. E, for fixed T
    if T:
        xlabel = 'E (GeV)'
        x_array = np.linspace(5, 200)

        # Plot truth value
        qhat_truth = [self.qhat(T=T, E=E, parameters=self.holdout_design) for E in x_array]
        plt.plot(x_array, qhat_truth, sns.xkcd_rgb['pale red'],
                 linewidth=2., label='Truth')

        # Plot 90% credible interval of qhat solution
        # --> Construct distribution of qhat by sampling each ABCD point
        qhat_posteriors = [[self.qhat(T=T, E=E, parameters=parameters)
                            for parameters in self.TransformedSamples]
                            for E in x_array]

    plt.xlabel(xlabel)
    plt.ylabel(r'$\hat{q}/T^3$')

    ymin = 0
    ymax = 2*max(qhat_truth)
    axes = plt.gca()
    axes.set_ylim([ymin, ymax])

    # Get list of mean qhat values for each T
    qhat_mean = [np.mean(qhat_values) for qhat_values in qhat_posteriors]
    plt.plot(x_array, qhat_mean, sns.xkcd_rgb['denim blue'],
             linewidth=2., linestyle='--', label='Extracted mean')

    # Get credible interval for each T
    # Specifically: highest posterior density interval (HPDI) via pymc3
    h = [pymc3.stats.hpd(np.array(qhat_values), self.confidence[0]) for qhat_values in qhat_posteriors]
    credible_low = [i[0] for i in h]
    credible_up =  [i[1] for i in h]
    plt.fill_between(x_array, credible_low, credible_up, color=sns.xkcd_rgb['light blue'],
                     label='{}% Credible Interval'.format(int(self.confidence[0]*100)))

    # Store also 60% CR
    h2 = [pymc3.stats.hpd(np.array(qhat_values), self.confidence[1]) for qhat_values in qhat_posteriors]
    credible_low2 = [i[0] for i in h2]
    credible_up2 =  [i[1] for i in h2]

    # Store whether truth value is contained within credible region
    qhat_closure = [((qhat_truth[i] < credible_up[i]) and (qhat_truth[i] > credible_low[i])) for i,_ in enumerate(x_array)]
    qhat_closure2 = [((qhat_truth[i] < credible_up2[i]) and (qhat_truth[i] > credible_low2[i])) for i,_ in enumerate(x_array)]

    # Draw legend
    first_legend = plt.legend(title=self.model, title_fontsize=15,
                             loc='upper right', fontsize=12)
    ax = plt.gca().add_artist(first_legend)

    if E:
        label = 'T'
    if T:
        label = 'E'
    plt.savefig('{}/Closure_{}.pdf'.format(self.plot_dir, label), dpi = 192)
    plt.close('all')

    # Plot distribution of posterior qhat values for a given T
    plt.hist(qhat_posteriors[0], bins=50,
            histtype='step', color='green')

    plt.savefig('{}/ClosureDist.pdf'.format(self.plot_dir), dpi = 192)
    plt.close('all')

    # Write result to pkl
    verbose = True
    if E:
        self.output_dict['T_array'] = x_array
        self.output_dict['T_qhat_truth'] = qhat_truth             # Truth
        self.output_dict['T_qhat_mean'] = qhat_mean               # Extracted mean
        self.output_dict['T_qhat_closure'] = qhat_closure         # Extracted posteriors
        self.output_dict['T_qhat_closure2'] = qhat_closure2       # Extracted posteriors
        if verbose:
          self.output_dict['T_credible_up'] = credible_up           # Extracted posteriors
          self.output_dict['T_credible_low'] = credible_low         # Extracted posteriors
          self.output_dict['T_credible_up2'] = credible_up2         # Extracted posteriors
          self.output_dict['T_credible_low2'] = credible_low2       # Extracted posteriors

    if T:
        self.output_dict['E_array'] = x_array
        self.output_dict['E_qhat_truth'] = qhat_truth             # Truth
        self.output_dict['E_qhat_mean'] = qhat_mean               # Extracted mean
        self.output_dict['E_qhat_closure'] = qhat_closure         # Extracted posteriors
        self.output_dict['E_qhat_closure2'] = qhat_closure2       # Extracted posteriors
        if verbose:
          self.output_dict['E_credible_up'] = credible_up           # Extracted posteriors
          self.output_dict['E_credible_low'] = credible_low         # Extracted posteriors
          self.output_dict['E_credible_up2'] = credible_up2         # Extracted posteriors
          self.output_dict['E_credible_low2'] = credible_low2       # Extracted posteriors

  #---------------------------------------------------------------
  # Plot design points
  #---------------------------------------------------------------
  def plot_design(self, holdout_test = False):

    # Tranform {A+C, A/(A+C), B, D, Q}  to {A,B,C,D,Q}
    design_points = self.AllData['design']
    if self.model in ['MATTER+LBT1', 'MATTER+LBT2']:
      transformed_design_points = np.copy(design_points)
      transformed_design_points[:,0] = design_points[:,0] * design_points[:,1]
      transformed_design_points[:,1] = design_points[:,0] - design_points[:,0] * design_points[:,1]
    else:
      transformed_design_points = np.copy(design_points)

    # Plot A vs. C example
    i = 2
    j = 0
    plt.locator_params(nbins=8)
    plt.scatter(transformed_design_points[:, j], transformed_design_points[:, i],
                c=sns.xkcd_rgb['denim blue'], alpha=0.5)
    plt.title('Design Points of Inputs {},{}'.format(self.Names[i], self.Names[j]), fontsize=16, weight='bold')
    plt.xlabel(self.Names[j], fontsize=20)
    plt.ylabel(self.Names[i], fontsize=20, rotation=0, labelpad=15)
    plt.savefig('{}/DesignPoints_AC.pdf'.format(self.plot_dir), dpi = 192)
    plt.close('all')

    # Plot grid of 2D projections
    NDimension = len(self.AllData["labels"])
    figure, axes = plt.subplots(figsize = (3 * NDimension, 3 * NDimension), ncols = NDimension, nrows = NDimension)
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):

            if i==NDimension-1 or (j==0 and i>3) or (j==NDimension-1 and i>3):
                ax.set_xlabel(self.Names[j], fontsize=20)
            if j==0:
                ax.set_ylabel(self.Names[i], fontsize=20)

            if i==j:
                ax.hist(transformed_design_points[:,i], bins=10,
                        range=self.Ranges_transformed[:,i], histtype='step', color=sns.xkcd_rgb['denim blue'])
                ax.set_xlim(*self.Ranges_transformed[:,j])
            if i>j:
                ax.scatter(transformed_design_points[:, j], transformed_design_points[:, i],
                            c=sns.xkcd_rgb['denim blue'], alpha=0.5)
                ax.set_xlim(*self.Ranges_transformed[:,j])
                ax.set_ylim(*self.Ranges_transformed[:,i])

                if holdout_test:
                  ax.plot(self.holdout_design[j], self.holdout_design[i], 'ro')

            if i<j:
                ax.axis('off')
    plt.savefig('{}/DesignPoints.pdf'.format(self.plot_dir), dpi = 192)
    plt.close('all')

  #---------------------------------------------------------------
  # Plot RAA of the model at a set of points in the parameter space
  #---------------------------------------------------------------
  def plot_RAA(self, points, name):

    TempPrediction = {"AuAu200": self.EmulatorAuAu200.predict(points),
                     "PbPb2760": self.EmulatorPbPb2760.predict(points),
                     "PbPb5020": self.EmulatorPbPb5020.predict(points)}

    SystemCount = len(self.AllData["systems"])

    figure, axes = plt.subplots(figsize = (15, 5 * SystemCount), ncols = 2, nrows = SystemCount)

    for s1 in range(0, SystemCount):  # Collision system
        for s2 in range(0, 2): # Centrality range
            axes[s1][s2].set_xlabel(r"$p_{T}$")
            axes[s1][s2].set_ylabel(r"$R_{AA}$")

            # Plot data points
            S1 = self.AllData["systems"][s1]
            O  = self.AllData["observables"][0][0]
            S2 = self.AllData["observables"][0][1][s2]

            DX = self.AllData["data"][S1][O][S2]['x']
            DY = self.AllData["data"][S1][O][S2]['y']
            DE = np.sqrt(self.AllData["data"][S1][O][S2]['yerr']['stat'][:,0]**2 + self.AllData["data"][S1][O][S2]['yerr']['sys'][:,0]**2)

            # Plot emulator predictions at design points
            for i, y in enumerate(TempPrediction[S1][O][S2]):
                axes[s1][s2].plot(DX, y, 'b-', alpha=0.1, label="Posterior" if i==0 else '')
            axes[s1][s2].errorbar(DX, DY, yerr = DE, fmt='ro', label="Measurements")

    figure.savefig('{}/RAA_{}.pdf'.format(self.plot_dir, name), dpi = 192)
    plt.close('all')

  #---------------------------------------------------------------
  # Plot residuals of RAA between the emulator and the true model values, at the design points
  #---------------------------------------------------------------
  def plot_emulator_RAA_residuals(self, holdout_test = False):

    # Get training points
    if holdout_test:
      Examples = [self.AllData['holdout_design']]
    else:
      Examples = self.AllData['design']
    Examples = np.array(Examples, copy=False, ndmin=2)

    # Get emulator predictions at training points
    TempPrediction = {"AuAu200": self.EmulatorAuAu200.predict(Examples, return_cov=True),
                     "PbPb2760": self.EmulatorPbPb2760.predict(Examples, return_cov=True),
                     "PbPb5020": self.EmulatorPbPb5020.predict(Examples, return_cov=True)}

    SystemCount = len(self.AllData["systems"])
    figure, axes = plt.subplots(figsize = (15, 5 * SystemCount), ncols = 2, nrows = SystemCount)

    # Loop through system and centrality range
    for s1 in range(0, SystemCount):  # Collision system
        for s2 in range(0, 2): # Centrality range
            axes[s1][s2].set_xlabel(r"$p_{T}$")
            axes[s1][s2].set_ylabel(r"$(R_{AA}^{emulator} - R_{AA}^{model}) / R_{AA}^{model}$")

            # Get keys for given system, centrality
            S1 = self.AllData["systems"][s1]
            O  = self.AllData["observables"][0][0]
            S2 = self.AllData["observables"][0][1][s2]

            # Get MC values at training points
            if holdout_test:
              model_x = self.AllData['holdout_model'][S1][O][S2]['x'] # pt-bin values
              model_y = self.AllData['holdout_model'][S1][O][S2]['Y'] # 1d array of model Y-values at holdout point
            else:
              model_x = self.AllData['model'][S1][O][S2]['x'] # pt-bin values
              model_y = self.AllData['model'][S1][O][S2]['Y'] # 2d array of model Y-values at each training point

            # Get emulator predictions at training points
            mean_prediction, cov_prediction = TempPrediction[S1]

            # Get interpolation uncertainty
            cov = cov_prediction[(O,S2),(O,S2)][0]
            variance = np.diagonal(cov)
            stdev_prediction = np.sqrt(variance)

            # Plot difference between model and emulator
            emulator_y = mean_prediction[O][S2] # 2d array of emulator Y-values at each training point
            for i, y in enumerate(emulator_y):

              if holdout_test:
                model_y_1d = model_y
                [self.true_raa[s1][s2].append(raa) for raa in model_y_1d]
                [self.emulator_raa_mean[s1][s2].append(raa) for raa in emulator_y[i]]
                [self.emulator_raa_stdev[s1][s2].append(stdev) for stdev in stdev_prediction]
              else:
                model_y_1d = model_y[i]

              deltaRAA = (emulator_y[i] - model_y_1d) / model_y_1d
              if holdout_test:
                  deltaRAA_stdev = stdev_prediction[i] / model_y_1d

              axes[s1][s2].plot(model_x, deltaRAA, 'b-', alpha=0.1, label="Posterior" if i==0 else '')
              if holdout_test:
                axes[s1][s2].fill_between(model_x, -deltaRAA_stdev, deltaRAA_stdev,
                                          lw=0, color=sns.xkcd_rgb['light blue'], alpha=.3, zorder=20)

    figure.savefig('{}/RAA_Residuals_Design.pdf'.format(self.plot_dir), dpi = 192)
    plt.close('all')

  #---------------------------------------------------------------
  # Plot residuals of each PC
  #---------------------------------------------------------------
  def plot_PC_residuals(self):

    for system in self.AllData["systems"]:
        # Get emulators for a given system (one per PC) from cache
        gps = emulator.emulators[system].gps
        nrows = len(gps)
        ncols = gps[0].X_train_.shape[1]

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4., nrows*4.) )
        ymax = np.ceil(max(np.fabs(g.y_train_).max() for g in gps))
        ylim = (-ymax, ymax)

        design = Design(system, self.workdir)
        test_points = [r*design.min + (1 - r)*design.max for r in [.2, .5, .8]]

        # Loop through emulators (one per PC)
        for ny, (gp, row) in enumerate(zip(gps, axes)):

            # Get list of training y-values
            y = gp.y_train_

            # Loop through training parameters {A+C,A/(A+C),B,D,Q}
            for nx, (x, label, xlim, ax) in enumerate(zip(gp.X_train_.T, design.labels, design.range, row)):

                # Plot training points
                ax.plot(x, y, 'o', ms=3., color='.75', zorder=10)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_xlabel(label)
                ax.set_ylabel('PC {}'.format(ny))

                # Plot emulator prediction (and stdev) for three different
                x = np.linspace(xlim[0], xlim[1], 100)
                X = np.empty((x.size, ncols))
                for k, test_point in enumerate(test_points):
                    X[:] = test_point
                    X[:, nx] = x
                    mean, std = gp.predict(X, return_std=True)

                    color = plt.cm.tab10(k)
                    ax.plot(x, mean, lw=.2, color=color, zorder=30)
                    ax.fill_between(x, mean - std, mean + std, lw=0, color=color, alpha=.3, zorder=20)

        plt.savefig('{}/EmulatorPCs_{}.pdf'.format(self.plot_dir, system), dpi = 192)
        plt.close('all')

  #---------------------------------------------------------------
  # Check that burn-in is sufficient
  #---------------------------------------------------------------
  def plot_MCMC_samples(self):

    with self.chain.dataset() as d:
      W = d.shape[0]     # number of walkers
      S = d.shape[1]     # number of steps
      N = d.shape[2]     # number of paramters
      T = int(S / 200)   # "thinning"
      A = 20 / W
      figure, axes = plt.subplots(figsize = (15, 2 * N), ncols = 1, nrows = N)
      for i, ax in enumerate(axes):
        for j in range(0, W):
          ax.plot(range(0, S, T), d[j, ::T, i], alpha = A)
      plt.savefig('{}/MCMCSamples.pdf'.format(self.plot_dir), dpi = 192)
      plt.close('all')

  #---------------------------------------------------------------
  # Plot posterior parameter distributions, either in  transformed
  # or non-transformed coordinates
  #---------------------------------------------------------------
  def plot_correlation(self, suffix = '', holdout_test = False):

    if 'Transformed' in suffix:
      Names = self.Names
      samples = self.TransformedSamples
      color = 'blue'
      colormap = 'Blues'
      ranges = self.Ranges_transformed
      if holdout_test:
          holdout_design = self.holdout_design
    else:
      Names = self.Names_untransformed
      samples = self.MCMCSamples
      color = 'green'
      colormap = 'Greens'
      ranges = self.Ranges
      if holdout_test:
          holdout_design = self.AllData['holdout_design']

    NDimension = len(self.AllData["labels"])
    figure, axes = plt.subplots(figsize = (3 * NDimension, 3 * NDimension), ncols = NDimension, nrows = NDimension)
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if i==j:

                # Draw 1D projection
                ax.hist(samples[:,i], bins=50,
                        range=ranges[:,i], histtype='step', color=color)
                ax.set_xlabel(Names[i])
                ax.set_xlim(*ranges[:,j])
                ymax = ax.get_ylim()[1]

                # If holdout test, draw the highest posterior density interval (HPDI)
                if holdout_test:
                    credible_interval = pymc3.stats.hpd(np.array(samples[:,i]), self.confidence[0])
                    ax.fill_between(credible_interval, [ymax,ymax], color=sns.xkcd_rgb['almost black'], alpha=0.1)

                    if self.model in ['LBT', 'MATTER'] or 'Transformed' in suffix:
                        # Store whether truth value is contained within credible region
                        theta_truth = holdout_design[i]
                        theta_closure = (theta_truth < credible_interval[1]) and (theta_truth > credible_interval[0])

                        credible_interval2 = pymc3.stats.hpd(np.array(samples[:,i]), self.confidence[1])
                        theta_closure2 = (theta_truth < credible_interval2[1]) and (theta_truth > credible_interval2[0])

                        name = self.Names[i]
                        self.output_dict['{}_closure'.format(name)] = theta_closure
                        self.output_dict['{}_closure2'.format(name)] = theta_closure2

            # Draw 2D correlations
            if i>j:
                ax.hist2d(samples[:, j], samples[:, i],
                          bins=50, range=[ranges[:,j], ranges[:,i]],
                          cmap=colormap)
                ax.set_xlabel(Names[j])
                ax.set_ylabel(Names[i])
                ax.set_xlim(*ranges[:,j])
                ax.set_ylim(*ranges[:,i])

                if holdout_test:
                  ax.plot(holdout_design[j], holdout_design[i], 'ro')

            if i<j:
                ax.axis('off')
    plt.savefig('{}/Posterior_Correlations{}.pdf'.format(self.plot_dir, suffix), dpi = 192)
    plt.close('all')

  #---------------------------------------------------------------
  def plot_avg_residuals(self):

    design_points = self.AllData['design']
    if self.model in ['MATTER+LBT1', 'MATTER+LBT2']:
      transformed_design_points = np.copy(design_points)
      transformed_design_points[:,0] = design_points[:,0] * design_points[:,1]
      transformed_design_points[:,1] = design_points[:,0] - design_points[:,0] * design_points[:,1]
    else:
      transformed_design_points = np.copy(design_points)

    if len(self.avg_residuals) < len(self.AllData['design']):
      transformed_design_points = transformed_design_points[0:self.n_max_holdout_tests]

    NDimension = len(self.AllData["labels"])
    figure, axes = plt.subplots(figsize = (3 * NDimension, 3 * NDimension), ncols = NDimension, nrows = NDimension)
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if i==j:
                ax.hist(transformed_design_points[:,i], bins=50, weights=self.avg_residuals,
                        range=self.Ranges_transformed[:,i], histtype='step', color='blue')
                ax.set_xlabel(self.Names[i])
                ax.set_xlim(*self.Ranges_transformed[:,j])
            if i>j:
                ax.hist2d(transformed_design_points[:, j], transformed_design_points[:, i], weights=self.avg_residuals,
                          bins=50, range=[self.Ranges_transformed[:,j], self.Ranges_transformed[:,i]],
                          cmap='Blues')
                ax.set_xlabel(self.Names[j])
                ax.set_ylabel(self.Names[i])
                ax.set_xlim(*self.Ranges_transformed[:,j])
                ax.set_ylim(*self.Ranges_transformed[:,i])
            if i<j:
                ax.axis('off')
    plt.savefig('{}/Average_Residuals.pdf'.format(self.plot_dir), dpi = 192)
    plt.close('all')

##################################################################
if __name__ == '__main__':

    # Define arguments
    parser = argparse.ArgumentParser(description='Jetscape STAT analysis')
    parser.add_argument('-c', '--configFile', action='store',
                        type=str, metavar='configFile',
                        default='analysis_config_QM.yaml',
                        help='Path of config file')
    parser.add_argument('-m', '--model', action='store',
                        type=str, metavar='model',
                        default='LBT',
                        help='model')
    parser.add_argument('-o', '--outputdir', action='store',
                        type=str, metavar='outputdir',
                        default='./STATGallery')
    parser.add_argument('-i', '--excludeIndex', action='store',
                        type=int, metavar='excludeIndex',
                        default=-1,
                        help='Index of design point to exclude from emulator')

    # Parse the arguments
    args = parser.parse_args()

    print('')
    print('Configuring RunAnalysis...')

    # If invalid configFile is given, exit
    if not os.path.exists(args.configFile):
      print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
      sys.exit(0)

    analysis = RunAnalysis(config_file=args.configFile,
                           model=args.model,
                           output_dir=args.outputdir,
                           exclude_index=args.excludeIndex)
    analysis.run_model()
