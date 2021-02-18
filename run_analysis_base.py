'''
Base class to steer Bayesian analysis and produce plots.
'''

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy

import os
import sys
import pickle
import yaml
import subprocess

import reader

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
    self.alpha = model_dict['alpha']
    self.Names = [r'{}'.format(s) for s in model_dict['parameter_names']]
    if 'parameter_names_untransformed' in model_dict:
        self.Names_untransformed = [r'{}'.format(s) for s in model_dict['parameter_names_untransformed']]
    else:
        self.Names_untransformed = self.Names
      
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
        self.RawDesign = reader.ReadDesign(os.path.join(self.workdir,'Design.dat'))
        self.AllData["design"] = self.RawDesign["Design"]
        self.AllData["labels"] = self.RawDesign["Parameter"]
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
  def qhat(self, T=0, E=0, parameters=None):
  
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
    self.AllData["keys"] = self.RawDesign["Parameter"]
    self.AllData["labels"] = self.RawDesign["Parameter"]
    self.AllData["ranges"] = self.ranges
    self.AllData["observables"] = [('R_AA', ['C0', 'C1'])]

    # If a holdout point is passed, exclude it from the design and prediction
    if exclude_index >= 0:
      self.exclude_holdout(exclude_index)

    # Data points
    self.Data = {"AuAu200": {"R_AA": {"C0": self.RawData1["Data"], "C1": self.RawData2["Data"]}},
        "PbPb2760": {"R_AA": {"C0": self.RawData3["Data"], "C1": self.RawData4["Data"]}},
        "PbPb5020": {"R_AA": {"C0": self.RawData5["Data"], "C1": self.RawData6["Data"]}}}

    # Model predictions
    self.Prediction = {"AuAu200": {"R_AA": {"C0": {"Y": self.RawPrediction1["Prediction"], "x": self.RawData1["Data"]['x']},
                                       "C1": {"Y": self.RawPrediction2["Prediction"], "x": self.RawData2["Data"]['x']}}},
                 "PbPb2760": {"R_AA": {"C0": {"Y": self.RawPrediction3["Prediction"], "x": self.RawData3["Data"]['x']},
                                       "C1": {"Y": self.RawPrediction4["Prediction"], "x": self.RawData4["Data"]['x']}}},
                 "PbPb5020": {"R_AA": {"C0": {"Y": self.RawPrediction5["Prediction"], "x": self.RawData5["Data"]['x']},
                                       "C1": {"Y": self.RawPrediction6["Prediction"], "x": self.RawData6["Data"]['x']}}}}

    # Covariance matrices - the indices are [system][measurement1][measurement2], each one is a block of matrix
    SysLength = {"sys,lumi,high": 9999, "sys,TAA,high": 9999, "default": 0.2}
    self.Covariance = reader.InitializeCovariance(self.Data)
    
    # Diagonal terms
    self.Covariance["AuAu200"][("R_AA", "C0")][("R_AA", "C0")]  = reader.EstimateCovariance(self.RawData1, self.RawData1, SysLength=SysLength)
    self.Covariance["AuAu200"][("R_AA", "C1")][("R_AA", "C1")]  = reader.EstimateCovariance(self.RawData2, self.RawData2, SysLength=SysLength)
    self.Covariance["PbPb2760"][("R_AA", "C0")][("R_AA", "C0")] = reader.EstimateCovariance(self.RawData3, self.RawData3, SysLength=SysLength)
    self.Covariance["PbPb2760"][("R_AA", "C1")][("R_AA", "C1")] = reader.EstimateCovariance(self.RawData4, self.RawData4, SysLength=SysLength)
    self.Covariance["PbPb5020"][("R_AA", "C0")][("R_AA", "C0")] = reader.EstimateCovariance(self.RawData5, self.RawData5, SysLength=SysLength)
    self.Covariance["PbPb5020"][("R_AA", "C1")][("R_AA", "C1")] = reader.EstimateCovariance(self.RawData6, self.RawData6, SysLength=SysLength)
    
    # Off-diagonal terms
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
    self.AllData["design"] = self.RawDesign["Design"]
    self.AllData["model"] = self.Prediction
    self.AllData["data"] = self.Data
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
    HoldoutDesign = self.RawDesign['Design'][exclude_index]
    HoldoutPrediction1 = self.RawPrediction1['Prediction'][exclude_index]
    HoldoutPrediction2 = self.RawPrediction2['Prediction'][exclude_index]
    HoldoutPrediction3 = self.RawPrediction3['Prediction'][exclude_index]
    HoldoutPrediction4 = self.RawPrediction4['Prediction'][exclude_index]
    HoldoutPrediction5 = self.RawPrediction5['Prediction'][exclude_index]
    HoldoutPrediction6 = self.RawPrediction6['Prediction'][exclude_index]
    
    # Model predictions
    HoldoutPrediction = {"AuAu200": {"R_AA": {"C0": {"Y": HoldoutPrediction1, "x": self.RawData1["Data"]['x']},
                                       "C1": {"Y": HoldoutPrediction2, "x": self.RawData2["Data"]['x']}}},
                 "PbPb2760": {"R_AA": {"C0": {"Y": HoldoutPrediction3, "x": self.RawData3["Data"]['x']},
                                       "C1": {"Y": HoldoutPrediction4, "x": self.RawData4["Data"]['x']}}},
                 "PbPb5020": {"R_AA": {"C0": {"Y": HoldoutPrediction5, "x": self.RawData5["Data"]['x']},
                                       "C1": {"Y": HoldoutPrediction6, "x": self.RawData6["Data"]['x']}}}}
     
    # Store the holdout point in the dictionary
    self.AllData['holdout_design'] = HoldoutDesign
    self.AllData['holdout_model'] = HoldoutPrediction
    
    # Remove the holdout point from the design
    self.RawDesign['Design'] = np.delete(self.RawDesign['Design'], exclude_index, axis = 0)

    # Remove the holdout point from the prediction
    self.RawPrediction1['Prediction'] = np.delete(self.RawPrediction1['Prediction'], exclude_index, axis=0)
    self.RawPrediction2['Prediction'] = np.delete(self.RawPrediction2['Prediction'], exclude_index, axis=0)
    self.RawPrediction3['Prediction'] = np.delete(self.RawPrediction3['Prediction'], exclude_index, axis=0)
    self.RawPrediction4['Prediction'] = np.delete(self.RawPrediction4['Prediction'], exclude_index, axis=0)
    self.RawPrediction5['Prediction'] = np.delete(self.RawPrediction5['Prediction'], exclude_index, axis=0)
    self.RawPrediction6['Prediction'] = np.delete(self.RawPrediction6['Prediction'], exclude_index, axis=0)
    
    # For closure test: set the data to be equal to the held-out point
    self.RawData1["Data"]["y"] = HoldoutPrediction1
    self.RawData2["Data"]["y"] = HoldoutPrediction2
    self.RawData3["Data"]["y"] = HoldoutPrediction3
    self.RawData4["Data"]["y"] = HoldoutPrediction4
    self.RawData5["Data"]["y"] = HoldoutPrediction5
    self.RawData6["Data"]["y"] = HoldoutPrediction6

  #---------------------------------------------------------------
  # Initialize data
  #---------------------------------------------------------------
  def init_files(self):
  
    # Read data files
    if self.model == 'MATTER':
      self.RawData1   = reader.ReadData('input/MATTERTruncated/Data_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawData2   = reader.ReadData('input/MATTERTruncated/Data_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawData3   = reader.ReadData('input/MATTERTruncated/Data_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawData4   = reader.ReadData('input/MATTERTruncated/Data_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawData5   = reader.ReadData('input/MATTERTruncated/Data_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawData6   = reader.ReadData('input/MATTERTruncated/Data_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    elif self.model == 'LBT':
      self.RawData1   = reader.ReadData('input/LBT/Data_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawData2   = reader.ReadData('input/LBT/Data_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawData3   = reader.ReadData('input/LBT/Data_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawData4   = reader.ReadData('input/LBT/Data_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawData5   = reader.ReadData('input/LBT/Data_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawData6   = reader.ReadData('input/LBT/Data_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    elif self.model == 'MATTER+LBT1':
      self.RawData1 = reader.ReadData('input/MATTERLBT1/Data_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawData2 = reader.ReadData('input/MATTERLBT1/Data_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawData3 = reader.ReadData('input/MATTERLBT1/Data_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawData4 = reader.ReadData('input/MATTERLBT1/Data_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawData5 = reader.ReadData('input/MATTERLBT1/Data_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawData6 = reader.ReadData('input/MATTERLBT1/Data_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    elif self.model == 'MATTER+LBT2':
      self.RawData1 = reader.ReadData('input/MATTERLBT2/Data_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawData2 = reader.ReadData('input/MATTERLBT2/Data_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawData3 = reader.ReadData('input/MATTERLBT2/Data_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawData4 = reader.ReadData('input/MATTERLBT2/Data_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawData5 = reader.ReadData('input/MATTERLBT2/Data_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawData6 = reader.ReadData('input/MATTERLBT2/Data_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    else:
      sys.exit('Unknown model {}! Options are: MATTER, LBT, MATTER+LBT1, MATTER+LBT2'.format(self.model))

    # Read covariance
    self.RawCov11L = reader.ReadCovariance('input/LBT/Covariance_PHENIX_AuAu200_RAACharged_0to10_2013_PHENIX_AuAu200_RAACharged_0to10_2013_Jake.dat')
    self.RawCov22L = reader.ReadCovariance('input/LBT/Covariance_PHENIX_AuAu200_RAACharged_40to50_2013_PHENIX_AuAu200_RAACharged_40to50_2013_Jake.dat')
    self.RawCov33L = reader.ReadCovariance('input/LBT/Covariance_ATLAS_PbPb2760_RAACharged_0to5_2015_ATLAS_PbPb2760_RAACharged_0to5_2015_Jake.dat')
    self.RawCov44L = reader.ReadCovariance('input/LBT/Covariance_ATLAS_PbPb2760_RAACharged_30to40_2015_ATLAS_PbPb2760_RAACharged_30to40_2015_Jake.dat')
    self.RawCov55L = reader.ReadCovariance('input/LBT/Covariance_CMS_PbPb5020_RAACharged_0to10_2017_CMS_PbPb5020_RAACharged_0to10_2017_Jake.dat')
    self.RawCov66L = reader.ReadCovariance('input/LBT/Covariance_CMS_PbPb5020_RAACharged_30to50_2017_CMS_PbPb5020_RAACharged_30to50_2017_Jake.dat')

    # Read design points
    if self.model == 'MATTER':
      self.RawDesign = reader.ReadDesign('input/MATTERTruncated/Design.dat')
    elif self.model == 'LBT':
      self.RawDesign = reader.ReadDesign('input/LBT/Design.dat')
    elif self.model == 'MATTER+LBT1':
      self.RawDesign = reader.ReadDesign('input/MATTERLBT1/Design.dat')
    elif self.model == 'MATTER+LBT2':
      self.RawDesign= reader.ReadDesign('input/MATTERLBT2/Design.dat')

    # Read model prediction
    if self.model == 'MATTER':
      self.RawPrediction1   = reader.ReadPrediction('input/MATTERTruncated/Prediction_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawPrediction2   = reader.ReadPrediction('input/MATTERTruncated/Prediction_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawPrediction3   = reader.ReadPrediction('input/MATTERTruncated/Prediction_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawPrediction4   = reader.ReadPrediction('input/MATTERTruncated/Prediction_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawPrediction5   = reader.ReadPrediction('input/MATTERTruncated/Prediction_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawPrediction6   = reader.ReadPrediction('input/MATTERTruncated/Prediction_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    elif self.model == 'LBT':
      self.RawPrediction1   = reader.ReadPrediction('input/LBT/Prediction_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawPrediction2   = reader.ReadPrediction('input/LBT/Prediction_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawPrediction3   = reader.ReadPrediction('input/LBT/Prediction_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawPrediction4   = reader.ReadPrediction('input/LBT/Prediction_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawPrediction5   = reader.ReadPrediction('input/LBT/Prediction_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawPrediction6   = reader.ReadPrediction('input/LBT/Prediction_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    elif self.model == 'MATTER+LBT1':
      self.RawPrediction1 = reader.ReadPrediction('input/MATTERLBT1/Prediction_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawPrediction2 = reader.ReadPrediction('input/MATTERLBT1/Prediction_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawPrediction3 = reader.ReadPrediction('input/MATTERLBT1/Prediction_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawPrediction4 = reader.ReadPrediction('input/MATTERLBT1/Prediction_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawPrediction5 = reader.ReadPrediction('input/MATTERLBT1/Prediction_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawPrediction6 = reader.ReadPrediction('input/MATTERLBT1/Prediction_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    elif self.model == 'MATTER+LBT2':
      self.RawPrediction1 = reader.ReadPrediction('input/MATTERLBT2/Prediction_PHENIX_AuAu200_RAACharged_0to10_2013.dat')
      self.RawPrediction2 = reader.ReadPrediction('input/MATTERLBT2/Prediction_PHENIX_AuAu200_RAACharged_40to50_2013.dat')
      self.RawPrediction3 = reader.ReadPrediction('input/MATTERLBT2/Prediction_ATLAS_PbPb2760_RAACharged_0to5_2015.dat')
      self.RawPrediction4 = reader.ReadPrediction('input/MATTERLBT2/Prediction_ATLAS_PbPb2760_RAACharged_30to40_2015.dat')
      self.RawPrediction5 = reader.ReadPrediction('input/MATTERLBT2/Prediction_CMS_PbPb5020_RAACharged_0to10_2017.dat')
      self.RawPrediction6 = reader.ReadPrediction('input/MATTERLBT2/Prediction_CMS_PbPb5020_RAACharged_30to50_2017.dat')
    
  #---------------------------------------------------------------
  # Return formatted string of class members
  #---------------------------------------------------------------
  def __str__(self):
    s = []
    variables = self.__dict__.keys()
    for v in variables:
      s.append('{} = {}'.format(v, self.__dict__[v]))
    return "[i] {} with \n .  {}".format(self.__class__.__name__, '\n .  '.join(s))
