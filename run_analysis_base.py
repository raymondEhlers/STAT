'''
Base class to steer Bayesian analysis and produce plots.
'''

import enum
import os
import pickle
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy
import yaml

import reader

class ParametrizationType(enum.Enum):
    """qHat parametrization types.

    Based on designations in the qHat parametrization.

    """

    pbpb_paper = 5
    exponential = 6
    binomial = 7


def _running_alpha_s(mu_square: float, alphas: float) -> float:
  active_flavor = 3
  square_lambda_QCD_HTL = np.exp( -12.0 * np.pi/( (33 - 2 * active_flavor) * alphas) );
  ans = 12.0 * np.pi/( (33.0 - 2.0 * active_flavor) * np.log(mu_square/square_lambda_QCD_HTL) );
  if mu_square < 1.0:
    ans = alphas

  print(
    f"Fixed-alphaS={alphas}, Lambda_QCD_HTL={np.sqrt(square_lambda_QCD_HTL)}, mu2={mu_square}, Running alpha_s={ans}"
  )
  return ans;

def _integral_PDF(xB: float, a: float, b: float) -> float:
  Xmin, Xmax = 0.01, 0.99
  X, dX = 0, 0
  ans = 0
  N = 100
  ix = 0
  dX = (1.0-0.0) / N
  if xB > Xmax:
    ans = 0
  else:
    ix = xB/dX
    for i in range(ix, N):
      X = (i+0.5) * dX
      if X<Xmin:
        X = Xmin
      ans = ans + np.pow(X, a) * np.pow(1-X, b) * dX
  return ans

def _virtuality_qhat_function(qhat_parametrization_type: ParametrizationType, ener_loc: float, mu_square: float, Q0: float, C1: float, C2: float, C3: float, C4: float) -> float:
  ans = 0
  xB = 0
  xB0 = 0

  if (mu_square <= Q0 * Q0):
    return 1.0

  # By parametrization
  if qhat_parametrization_type == ParametrizationType.pbpb_paper:
    ans =  1.0 + C1*np.log(Q0*Q0)*np.log(Q0*Q0) + C2*np.pow(np.log(Q0*Q0),4)
    ans = ans/( 1.0 + C1*np.log(mu_square)*np.log(mu_square) + C2*np.pow(np.log(mu_square),4))
  elif qhat_parametrization_type == ParametrizationType.exponential:
    # number 6 in MATTER code
    xB = mu_square / (2.0 * ener_loc)
    xB0 = Q0*Q0 / (2.0 * ener_loc)
    if xB <= xB0:
       return 1.0
    if C3 > 0.0 and xB < 0.99:
      ans = (np.exp(C3 * (1.0-xB) ) - 1.0 )/(1.0 + C1*np.log(mu_square / 0.04) + C2*np.log(mu_square/0.04)*np.log(mu_square/0.04) )
      ans = ans*(1.0 + C1*np.log(Q0 * Q0 / 0.04) + C2*np.log(Q0 * Q0 / 0.04)*np.log(Q0*Q0/0.04) )/( np.exp(C3 * (1.0-xB0)) - 1.0 )
    elif C3 == 0.0 and xB < 0.99:
      ans = (1.0-xB)/(1.0 + C1 * np.log(mu_square / 0.04) + C2 * np.log(mu_square / 0.04) * np.log(mu_square / 0.04) )
      ans = ans * (1.0 + C1 * np.log(Q0 * Q0 / 0.04) + C2 * np.log(Q0 * Q0 / 0.04) * np.log(Q0 * Q0 / 0.04) )/(1 - xB0)
    else:
      return 0.0
  elif qhat_parametrization_type == ParametrizationType.binominal:
    # Number 7 in MATTER code
    xB  = mu_square/(2.0 * ener_loc)
    xB0 = Q0*Q0/(2.0 * ener_loc)
    if xB<=xB0:
      return 1.0
    ans = _integral_PDF(xB, C3, C4) / (1.0 + C1*np.log(mu_square/0.04) + C2*np.log(mu_square/0.04)*np.log(mu_square/0.04) )
    integral_norm = _integral_PDF(xB0, C3, C4)
    if integral_norm > 0.0:
      ans=ans*(1.0 + C1*np.log(mu_square/0.04) + C2*np.log(mu_square/0.04)*np.log(mu_square/0.04) )/integral_norm
    else:
      return 0
  else:
    raise NotImplementedError(f"Unknown parametrization type {qhat_parametrization_type}")
  return ans;

################################################################
class RunAnalysisBase():

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, config_file, model, output_dir, exclude_index=-1, **kwargs):
    super(RunAnalysisBase, self).__init__(**kwargs)

    self.model = model
    self.output_dir_base = output_dir
    self.exclude_index = exclude_index

    # Set various paths
    if exclude_index < 0:
        subdir = 'main'
    else:
        subdir =  'holdout/{}'.format(exclude_index)
    self.workdir = os.path.join(output_dir, '{}/{}'.format(model,subdir))

    if not os.path.exists(self.workdir):
        os.makedirs(self.workdir)

    self.pkl_path = os.path.join(self.workdir,'default.p')
    self.cache_dir = os.path.join(self.workdir,'cache')
    self.cachedir_emulator = os.path.join(self.cache_dir , 'emulator')

    # Initialize yaml settings
    self.initialize_config(config_file)

    self.colors = [sns.xkcd_rgb['denim blue'],sns.xkcd_rgb['pale red'],sns.xkcd_rgb['medium green']]

  #---------------------------------------------------------------
  # Initialize config
  #---------------------------------------------------------------
  def initialize_config(self, config_file):

    # Read config file
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    # Get model parameters
    model_dict = config['models'][self.model]
    self.input_dir = model_dict['input_dir']
    self.alpha = model_dict['alpha']
    self.Names = [r'{}'.format(s) for s in model_dict['parameter_names']]
    if 'parameter_names_untransformed' in model_dict:
        self.Names_untransformed = [r'{}'.format(s) for s in model_dict['parameter_names_untransformed']]
    else:
        self.Names_untransformed = self.Names
    self.parametrization_type = model_dict["parametrization_type"]

    min = model_dict['min']
    max = model_dict['max']
    self.ranges_transformed = [tuple([min[i], max[i]]) for i in range(len(min))]
    if 'min_untransformed' in model_dict:
        min_untransformed = model_dict['min_untransformed']
        max_untransformed = model_dict['max_untransformed']
        self.ranges = [tuple([min_untransformed[i], max_untransformed[i]]) for i in range(len(min))]
    else:
        self.ranges = self.ranges_transformed
    self.Ranges = np.array(self.ranges).T
    self.Ranges_transformed = np.array(self.ranges_transformed).T

    self.debug_level = config['debug_level']

    # Design parameters
    self.generate_design = config['generate_design']
    self.n_points_per_dimension = config['n_points_per_dimension']

    # Emulator parameters
    self.retrain_emulator = config['retrain_emulator']
    self.n_pc = config['n_pc']
    self.n_restarts = config['n_restarts']

    # MCMC parameters
    self.rerun_mcmc = config['rerun_mcmc']
    self.n_walkers = config['n_walkers']
    self.n_burn_steps = config['n_burn_steps']
    self.n_steps = config['n_steps']

    # Closure test parameters
    self.confidence = config['confidence']

  #---------------------------------------------------------------
  # Run analysis
  #---------------------------------------------------------------
  def run_model(self):

    # Initialize a few settings
    self.output_dir = os.path.join(self.output_dir_base, self.model)
    print(self)

    if self.generate_design:

        # Generate a separate design for each collision system
        #   (since each emulator, i.e. PC, will get different
        #    design points in the iterative process)
        ndim = len(self.ranges)
        npoints = ndim * self.n_points_per_dimension
        min = self.Ranges[0]
        max = self.Ranges[1]

        # Generate design in [0,1]x[0,1]x...
        design_unit = self.generate_lhs(npoints=npoints, ndim=ndim, seed=0)

        # Scale to variable ranges
        design = min + (max - min)*design_unit

        # Write to file
        # Version 1.0
        header = 'Version 1.0 \nParameter'
        for i in range(ndim):
            header += ' {}'.format(self.Names[i])
        np.savetxt(os.path.join(self.workdir, 'design.dat'), design, header=header)

        # Plot the design
        self.plot_dir = os.path.join(self.workdir, 'plots')
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        self.AllData = {}
        self.RawData['Design'] = reader.ReadDesign(os.path.join(self.workdir,'Design.dat'))
        self.AllData["design"] = self.RawData['Design']['Design']
        self.AllData["labels"] = self.RawData['Design']['Parameter']
        self.plot_design()

    else:

        # Run user-defined function
        self.run_analysis()

  #---------------------------------------------------------------
  # Generate latin hypercube using R lhs package (based on src.Design)
  #---------------------------------------------------------------
  def generate_lhs(self, npoints=0, ndim=0, seed=0):

    proc = subprocess.run(
            ['R', '--slave'],
            input="""
            library('lhs')
            set.seed({})
            write.table(maximinLHS({}, {}), col.names=FALSE, row.names=FALSE)
            """.format(seed, npoints, ndim).encode(),
            stdout=subprocess.PIPE,
            check=True
          )

    lhs = np.array([l.split() for l in proc.stdout.splitlines()], dtype=float)
    return lhs

  #---------------------------------------------------------------
  # Run user-defined function
  #---------------------------------------------------------------
  def run_analysis(self):

    raise NotImplementedError('You must implement run_analysis()!')

  #---------------------------------------------------------------
  # Return value of qhat/T^3
  def qhat(self, T=0, E=0, Q=0, parameters=None):

    if self.model == "MATTER_LBT_QM_exponential":
      # Parameters
      alpha_s_fix, Q0, C1, C2, tau_0, C3 = parameters
      active_flavor = 3
      # Extracted from JetScapeConstants
      C_a = 3.0

      # From GeneralQhatFunction
      debye_mass_square = alpha_s_fix * 4 * np.pi * np.pow(T, 2.0) * (6.0 + active_flavor) / 6.0
      scale_net = 2 * E * T
      if scale_net < 1.0:
        scale_net = 1.0
      # TODO: Determine how to figure out alpha_s, since it varies in the MATTER calculation. I guess it needs to be
      #       integrated over?
      qhat = (C_a * 50.4864 / np.pi) * _running_alpha_s(mu_sqaure=scale_net, alpha_s=alpha_s) * alpha_s_fix * np.pow(T, 3) * np.log(scale_net / debye_mass_square)
      qhat = qhat * _virtuality_qhat_function(
        qhat_parametrization_type=ParametrizationType.exponential, ener_loc=E,
        # mu_square is the virtuality.
        # see: https://github.com/JETSCAPE/JETSCAPE-COMP/blob/e83b8ac71f8d71b9ad8ed71935f85d8951a16cb9/src/jet/Matter.cc#L805
        mu_square=Q,
        Q0=Q0, C1=C1, C2=C2, C3=C3,
        # C4 is unused for this parametrization, so just set to 0
        C4=0,
      )
      return qhat
    else:
      Lambda = 0.2
      C_R = 4./3.
      coeff = 42 * C_R * scipy.special.zeta(3) / np.pi * np.square(4*np.pi/9)
      Q=E
      if self.model == 'MATTER+LBT1':
        A = parameters[0]
        B = parameters[2]
        C = parameters[1]
        D = parameters[3]
        Q0 = parameters[4]
        term1 = A * (np.log(E/Lambda) - np.log(B)) / np.square(np.log(E/Lambda))  * np.heaviside(Q-Q0, 0.)
        term2 = C * (np.log(E/T) - np.log(D)) / np.square(np.log(E*T/(Lambda*Lambda)))
      elif self.model == 'MATTER+LBT2':
        A = parameters[0]
        C = parameters[1]
        D = parameters[2]
        Q0 = parameters[3]
        term1 = A * (np.log(Q/Lambda) - np.log(Q0/Lambda)) / np.square(np.log(Q/Lambda)) * np.heaviside(Q-Q0, 0.)
        term2 = C * (np.log(E/T) - np.log(D)) / np.square(np.log(E*T/(Lambda*Lambda)))
      elif self.model in ['LBT', 'MATTER']:
        A = parameters[0]
        B = parameters[1]
        C = parameters[2]
        D = parameters[3]
        term1 = A * (np.log(E/Lambda) - np.log(B)) / np.square(np.log(E/Lambda))
        term2 = C * (np.log(E/T) - np.log(D)) / np.square(np.log(E*T/(Lambda*Lambda)))

      return coeff * (term1 + term2)

  #---------------------------------------------------------------
  # Initialize data
  #---------------------------------------------------------------
  def initialize(self, exclude_index = -1):

    self.init_files()
    self.init_model(exclude_index)

  #---------------------------------------------------------------
  # Initialize data to dictionary
  #---------------------------------------------------------------
  def init_model(self, exclude_index = -1):

    # Initialize empty dictionary
    self.AllData = {}

    # Basic information
    self.AllData["systems"] = ["AuAu200", "PbPb2760", "PbPb5020"]
    self.AllData["keys"] = self.RawData['Design']['Parameter']
    self.AllData["labels"] = self.RawData['Design']['Parameter']
    self.AllData["ranges"] = self.ranges
    self.AllData["observables"] = [('R_AA', ['C0', 'C1'])]

    # If a holdout point is passed, exclude it from the design and prediction
    if exclude_index >= 0:
      self.exclude_holdout(exclude_index)

    # Data points
    self.Data = self.recursive_defaultdict()
    for system in self.RawData['Data'].keys():
      for observable in self.RawData['Data'][system].keys():
        for centrality in self.RawData['Data'][system][observable].keys():
          self.Data['Prediction'][system][observable][centrality] = self.RawData['Data'][system][observable][centrality]['Data']

    # Model predictions
    self.Prediction = self.recursive_defaultdict()
    for system in self.RawData['Prediction'].keys():
      for observable in self.RawData['Prediction'][system].keys():
        for centrality in self.RawData['Prediction'][system][observable].keys():
          self.Prediction['Prediction'][system][observable][centrality]['Y'] = self.RawData['Prediction'][system][observable][centrality]['Prediction']
          self.Prediction['Prediction'][system][observable][centrality]['x'] = self.RawData['Data'][system][observable][centrality]['Data']

    # Covariance matrices - the indices are [system][measurement1][measurement2], each one is a block of matrix
    SysLength = {"sys,lumi,high": 9999, "sys,TAA,high": 9999, "default": 0.2}
    self.Covariance = reader.InitializeCovariance(self.RawData['Data'])

    # Diagonal terms
    self.Covariance = self.recursive_defaultdict()
    for system in self.RawData['Data'].keys():
      for observable in self.RawData['Data'][system].keys():
        for centrality in self.RawData['Data'][system][observable].keys():
          self.Covariance[system][(observable, centrality)][(observable, centrality)] = reader.EstimateCovariance(self.RawData['Data'][system][observable][centrality],
                                                                                                                  self.RawData['Data'][system][observable][centrality],
                                                                                                                  SysLength=SysLength)

    # TODO: Off-diagonal terms
    self.Covariance["AuAu200"][("R_AA", "C0")][("R_AA", "C1")]  = reader.EstimateCovariance(self.RawData1, self.RawData2, SysLength = {"sys,lumi,high": 9999, "default": -1}, SysStrength = {"sys,lumi,high": 1, "default": 0})
    self.Covariance["AuAu200"][("R_AA", "C1")][("R_AA", "C0")]  = reader.EstimateCovariance(self.RawData2, self.RawData1, SysLength = {"sys,lumi,high": 9999, "default": -1}, SysStrength = {"sys,lumi,high": 1, "default": 0})
    self.Covariance["PbPb2760"][("R_AA", "C0")][("R_AA", "C1")] = reader.EstimateCovariance(self.RawData3, self.RawData4, SysLength = {"sys,lumi,high": 9999, "default": -1}, SysStrength = {"sys,lumi,high": 1, "default": 0})
    self.Covariance["PbPb2760"][("R_AA", "C1")][("R_AA", "C0")] = reader.EstimateCovariance(self.RawData4, self.RawData3, SysLength = {"sys,lumi,high": 9999, "default": -1}, SysStrength = {"sys,lumi,high": 1, "default": 0})
    self.Covariance["PbPb5020"][("R_AA", "C0")][("R_AA", "C1")] = reader.EstimateCovariance(self.RawData5, self.RawData6, SysLength = {"sys,lumi,high": 9999, "default": -1}, SysStrength = {"sys,lumi,high": 1, "default": 0})
    self.Covariance["PbPb5020"][("R_AA", "C1")][("R_AA", "C0")] = reader.EstimateCovariance(self.RawData6, self.RawData5, SysLength = {"sys,lumi,high": 9999, "default": -1}, SysStrength = {"sys,lumi,high": 1, "default": 0})

    # This is how we can supply external pre-generated matrices
    # Covariance["AuAu200"][("R_AA", "C0")][("R_AA", "C0")] = RawCov1["Matrix"]
    #Covariance["AuAu200"][("R_AA", "C0")][("R_AA", "C0")] = RawCov11E["Matrix"]
    #Covariance["AuAu200"][("R_AA", "C1")][("R_AA", "C1")] = RawCov22E["Matrix"]
    #Covariance["PbPb2760"][("R_AA", "C0")][("R_AA", "C0")] = RawCov33E["Matrix"]
    #Covariance["PbPb2760"][("R_AA", "C1")][("R_AA", "C1")] = RawCov44E["Matrix"]
    #Covariance["PbPb5020"][("R_AA", "C0")][("R_AA", "C0")] = RawCov55E["Matrix"]
    #Covariance["PbPb5020"][("R_AA", "C1")][("R_AA", "C1")] = RawCov66E["Matrix"]

    # Assign data to the dictionary
    self.RawData['Design'][system][observable][centrality]
    self.AllData["design"] = self.RawData['Design']['Design']
    self.AllData["model"] = self.Prediction
    self.AllData["data"] = self.RawData['Data']
    self.AllData["cov"] = self.Covariance

    # Save to the desired pickle file
    with open(self.pkl_path, 'wb') as handle:
      pickle.dump(self.AllData, handle, protocol = pickle.HIGHEST_PROTOCOL)
    print('Wrote {}'.format(self.pkl_path))

    if self.debug_level > 0:
      print(self.AllData["design"].shape)
      print(self.AllData["model"]["AuAu200"]["R_AA"]["C0"]["Y"].shape)
      print(self.AllData["model"]["AuAu200"]["R_AA"]["C1"]["Y"].shape)
      print(self.AllData["model"]["PbPb2760"]["R_AA"]["C0"]["Y"].shape)
      print(self.AllData["model"]["PbPb2760"]["R_AA"]["C1"]["Y"].shape)
      print(self.AllData["model"]["PbPb5020"]["R_AA"]["C0"]["Y"].shape)
      print(self.AllData["model"]["PbPb5020"]["R_AA"]["C1"]["Y"].shape)

      print("AuAu200, C0")
      print(self.Covariance["AuAu200"][("R_AA", "C0")][("R_AA", "C0")])
      print("AuAu200, C1")
      print(self.Covariance["AuAu200"][("R_AA", "C1")][("R_AA", "C1")])
      print("PbPb2760, C0")
      print(self.Covariance["PbPb2760"][("R_AA", "C0")][("R_AA", "C0")])
      print("PbPb2760, C1")
      print(self.Covariance["PbPb2760"][("R_AA", "C1")][("R_AA", "C1")])

      SystemCount = len(self.AllData["systems"])
      figure, axes = plt.subplots(figsize = (15, 5 * SystemCount), ncols = 2, nrows = SystemCount)
      axes[0][0].imshow((self.Covariance["AuAu200"][("R_AA", "C0")][("R_AA", "C0")]))
      axes[0][1].imshow((self.Covariance["AuAu200"][("R_AA", "C1")][("R_AA", "C1")]))
      axes[1][0].imshow((self.Covariance["PbPb2760"][("R_AA", "C0")][("R_AA", "C0")]))
      axes[1][1].imshow((self.Covariance["PbPb2760"][("R_AA", "C1")][("R_AA", "C1")]))
      axes[2][0].imshow((self.Covariance["PbPb5020"][("R_AA", "C0")][("R_AA", "C0")]))
      axes[2][1].imshow((self.Covariance["PbPb5020"][("R_AA", "C1")][("R_AA", "C1")]))
      figure.tight_layout()

  #---------------------------------------------------------------
  # Exclude a holdout point from the design and prediction, and store it
  #---------------------------------------------------------------
  def exclude_holdout(self, exclude_index = None):

    # Store the holdout point design and prediction
    HoldoutDesign = self.RawData['Design']['Design'][exclude_index]

    # Model predictions
    HoldoutPrediction = self.recursive_defaultdict()
    for system in self.RawData['Data'].keys():
      for observable in self.RawData['Data'][system].keys():
        for centrality in self.RawData['Data'][system][observable].keys():
          HoldoutPrediction[system][observable][centrality]['Y'] = self.RawData['Prediction'][system][observable][centrality]['Prediction'][exclude_index]
          HoldoutPrediction[system][observable][centrality]['x'] = self.RawData['Data'][system][observable][centrality]['Data']['x']

    # Store the holdout point in the dictionary
    self.AllData['holdout_design'] = HoldoutDesign
    self.AllData['holdout_model'] = HoldoutPrediction

    # Remove the holdout point from the design
    self.RawData['Design']['Design'] = np.delete(self.RawData['Design']['Design'], exclude_index, axis = 0)

    # Remove the holdout point from the prediction
    for system in self.RawData['Data'].keys():
      for observable in self.RawData['Data'][system].keys():
        for centrality in self.RawData['Data'][system][observable].keys():
          self.RawData['Prediction'][system][observable][centrality]['Prediction'] = np.delete(self.RawData['Prediction'][system][observable][centrality]['Prediction'], exclude_index, axis=0)

    # For closure test: set the data to be equal to the held-out point
    for system in self.RawData['Data'].keys():
      for observable in self.RawData['Data'][system].keys():
        for centrality in self.RawData['Data'][system][observable].keys():
          self.RawData['Data'][system][observable][centrality]['Data']['y'] = self.RawData['Prediction'][system][observable][centrality]['Prediction'][exclude_index]

  #---------------------------------------------------------------
  # Initialize data
  #---------------------------------------------------------------
  def init_files(self):

    self.RawData = self.recursive_defaultdict()

    #design_input_files = Path(self.input_dir).glob("Design*")
    #for input_file in design_input_files:
    #  parametrization_type = str(input_file.name).split("_")[1]
    #  self.RawData["Design"][parametrization_type] = reader.ReadDesign(input_file)

    # Read txt files of experimental data, covariance matrices, design points, and predictions
    for input_file in sorted(Path(self.input_dir).glob("*.dat")):
      if "Design" in input_file.name:
        # Handle design points.
        parametrization_type = str(input_file.name).split("_")[1]
        # Only store design points that we are interested in processing.
        if parametrization_type == self.parametrization_type:
          self.RawData["Design"][parametrization_type] = reader.ReadDesign(input_file)
      else:
        system, observable, centrality = self.filename_to_labels(input_file.name)

        if 'Data' in input_file.name:
          result = reader.ReadData(input_file)
          # If the parametrization type is stored in the system, we would need to duplicate the data here.
          # However, we won't do this for our first pass, so it's not necessary
          # Store a copy of the data for each parametrization
          #for parametrization_type in self.parametrization_types:
          #  _system = f"{system}{parametrization_type}"
          #  self.RawData['Data'][_system][observable][centrality] = result
          self.RawData['Data'][system][observable][centrality] = result
        elif 'Covariance' in input_file.name:
          self.RawData['RawCov1L'][system][observable][centrality] = reader.ReadCovariance(input_file)
        elif 'Prediction' in input_file.name:
          # Only store if the prediction is relevant for the parametrization type which we're using
          if parametrization_type in input_file.name:
            self.RawData['Prediction'][system][observable][centrality] = reader.ReadPrediction(input_file)

  #---------------------------------------------------------------
  # Initialize data
  #---------------------------------------------------------------
  def filename_to_labels(self, filename):
    """Convert filename to label

    Note:
      Each convention that's labeled "HACK" can be resolved by standardizing the names
      between the predictions and the data. However, this is simplest for now (RJE, March 2022).
    """

    items = filename[:-4].split('_')

    system = items[2]
    # HACK: Remove the parametrization type from the system name.
    system = system.replace(self.parametrization_type, "")

    centrality_index = -2 if "Data" in filename else -1
    centrality = items[centrality_index]
    # HACK: Normalize the names of the Predictions (which use "to") and the Data, which uses "-".
    # "=" is arbitrarily selected as the convention.
    if "to" in centrality:
      centrality = centrality.replace("to", "-")

    # HACK: Normalize experiment name to upper case
    items[1] = items[1].upper()

    if 'hadron' in filename:
      # HACK: Rename pt -> RAA in Prediction name, since this is actually what we're looking at
      if items[4] == "pt":
        items[4] = "RAA"
      observable = f'{items[1]}_{items[3]}_{items[4]}_{items[5]}'
    elif 'jet' in filename:
      # HACK: Rename pt -> RAA in Prediction name, since this is actually what we're looking at
      if items[5] == "pt":
        items[5] = "RAA"
      # HACK: Rename R0.3 -> R03 for consistency.
      if "R" in items[6] and "." in items[6]:
        items[6] = items[6].replace(".", "")
      observable = f'{items[1]}_{items[3]}_{items[4]}_{items[5]}_{items[6]}'

    return system, observable, centrality

  #---------------------------------------------------------------
  # Create a nested defaultdict
  #---------------------------------------------------------------
  def recursive_defaultdict(self):
    return defaultdict(self.recursive_defaultdict)

  #---------------------------------------------------------------
  # Return formatted string of class members
  #---------------------------------------------------------------
  def __str__(self):
    s = []
    variables = self.__dict__.keys()
    for v in variables:
      s.append('{} = {}'.format(v, self.__dict__[v]))
    return "[i] {} with \n .  {}".format(self.__class__.__name__, '\n .  '.join(s))
