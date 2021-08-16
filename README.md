# LSTM Temperature Model
This README explains how to use the provided data and code in this data release to replicate the training and prediction steps for the LSTM stream temperature models described in the paper. The code may be run on Mac or Linux operating systems where the Anaconda package manager is installed. A CUDA-compatible GPU and 256GB of memory are strongly recommended.

## Configure software environment
Create and activate the conda environment with all necessary software packages, using the condaenv_lstm_linux.yml file. These instructions recreate the python environment that was used to generate the results in the manuscript, and they should be sufficient to run the models yourself.

```
# [in a shell]
conda update -n base -c defaults conda
conda env create -f condaenv_lstm_linux.yml -n lstm_resdag
conda activate lstm_resdag
```

## Download and extract data
Steps to acquire the data to run this code:

Download files from ScienceBase. This can be done using the ScienceBase browser interface or using the sciencebasepy python package as follows:
```
# [in python, inside the LSTM_temp2 directory]
import os
import sciencebasepy
from zipfile import ZipFile
sb = sciencebasepy.SbSession()
sb.login(username, password) # enter your username and password. this is only necessary before the data are made public
os.mkdir('datarelease')
sb.get_item_files(sb.get_item('606db85fd34e670a7d5f61f0'), 'datarelease') # basins and gages
sb.get_item_files(sb.get_item('6083384fd34efe46ec0a2333'), 'datarelease') # temperature observations
sb.get_item_files(sb.get_item('6084cab2d34eadd49d31aeab'), 'datarelease') # model drivers and basin attributes
sb.get_item_files(sb.get_item('6084cb2ed34eadd49d31aeaf'), 'datarelease') #temperature predictions

# Extract the zipfiles
ZipFile('datarelease/gages.zip', 'r').extractall('datarelease')
ZipFile('datarelease/basins.zip', 'r').extractall('datarelease')
ZipFile('datarelease/predictions.zip', 'r').extractall('datarelease')
```

## Set up model inputs

Now create the input files that the model code will ingest:
```
import os
import pandas as pd
import shapefile
import numpy as np

#site attributes
coords_shp = shapefile.Reader('datarelease/gages.shp')
coords = pd.DataFrame(coords_shp.records(), columns=['site_no', 'site_name', 'lat', 'long']).\
    rename(columns={'long': 'lon'})
attr = pd.read_csv('datarelease/basin_attributes.csv', dtype={'site_no': 'str'}).\
    merge(coords, how='outer')
attr['site_no'] = pd.to_numeric(attr['site_no']) #attr now includes coordinates

#forcing data
temperature = pd.read_csv('datarelease/temperature_observations.csv')
forcing = pd.read_csv('datarelease/forcings.csv')

forcing_temp = forcing.merge(temperature, how = 'outer')
forcing_temp_renamed = forcing_temp.rename(columns={"discharge(cfs)" : "00060_Mean", "wtemp(C)" : "00010_Mean", "dayl(s/d)" : "dayl(s)", "prcp(mm/d)" :"prcp(mm/day)"})

```
### Explanation of different experiments

At this point, you can decide which experiment to replicate. `experiments.csv` in the data release describes the various experiments run in this data release. These columns are most important:

- train_filter: Apply this filter to forcing data to filter out training and test data
- test_filter: apply this filter to get *only* the test data. Test/train time period split is handled internally in the modeling code when running, use this to extract sites for a particular test set when checking performance.
- hidden_sizes:  The sizes of hidden layers of the neural network.  Set these accordingly in StreamTemp-Integ.py
- batch_size: Size of training batch.  Also set in StreamTemp-Integ.py

To filter the sites and forcing data based on the first experiment in experiments.csv, and write the input files:
```
os.makedirs('input/forcing', exist_ok = True)
os.makedirs('input/attr', exist_ok = True)
attr_filtered = attr.query('dag >= 60')
forcing_temp_renamed_filter = forcing_temp_renamed[forcing_temp_renamed['site_no'].isin(attr_filtered['site_no'])]
forcing_temp_renamed_filter.reset_index().to_feather('input/forcing/forcing_99.feather', version = 1)
attr_filtered.reset_index().to_feather('input/attr/attr_99.feather', version = 1)
```

Now you are almost ready to run the model code.

## Run model

First, set the forcing file name, site attributes name, batch size and hidden layer 
sizes in StreamTemp-Integ.py on the line numbers shown.  Note that you only specify the base filename here;
the directories (`input/forcing` and `input/attr`) are specified elsewhere.

```
19 forcing_list = ['forcing_99.feather']
20 attr_list = ['attr_99.feather']
21              
22 Batch_list = [ 50]
23 Hidden_list = [100]
 ```

Now you can train and make predictions with the model.

From a terminal:
`python StreamTemp-Integ.py`

Or from an iPython notebook:
`%run StreamTemp-Integ.py`

## Examine outputs
Output files will be written to the `output` folder in a subdirectory named according to various model hyperparameters, the start/end dates of the training data, and the random number seed used.  `obs.npy` contains observations for the test time period, and `pred.npy` contains the test period predictions.  The model script is set up to run train the model for six different random number seeds (for size different random neural network weight initializations).

### Average together initializations

```
#Change these file names to the appropriate ones
initialization_preds = ['output/epochs2000_batch47_rho365_hiddensize100_Tstart20101001_Tend20141001_1/All-2010-2016/pred.npy',
                       'output/epochs2000_batch47_rho365_hiddensize100_Tstart20101001_Tend20141001_2/All-2010-2016/pred.npy',
                       'output/epochs2000_batch47_rho365_hiddensize100_Tstart20101001_Tend20141001_3/All-2010-2016/pred.npy',
                       'output/epochs2000_batch47_rho365_hiddensize100_Tstart20101001_Tend20141001_4/All-2010-2016/pred.npy',
                       'output/epochs2000_batch47_rho365_hiddensize100_Tstart20101001_Tend20141001_5/All-2010-2016/pred.npy',
                       'output/epochs2000_batch47_rho365_hiddensize100_Tstart20101001_Tend20141001_6/All-2010-2016/pred.npy']
                       
def load_initializations(files):
    init_array = []
    for i in range(len(files)):
        print(files[i])
        init_array.append(np.load(files[i])[:,:,0])
    return(init_array)
        
init_array = load_initializations(initialization_preds)
init_means = np.mean(init_array, axis = 0) 
```

### To compare to observations

```
#load in observations, dropping degenerate third dimension
obs = np.load('output/epochs2000_batch47_rho365_hiddensize100_Tstart20101001_Tend20141001_1/All-2010-2016/pred.npy')[:,:,0]
#generate variety of metrics, or write own code as desired
from hydroDL.post import plot, stat
err = stat.statError(init_means, obs)

```

### To compare to specific test sites

```
#filter forcings to sites, based on test_filter column in experiments.csv
attr_filtered_test = attr.query('dag == 60')
attr_filtered
forcing_temp_renamed_filter_test = forcing_temp_renamed[forcing_temp_renamed['site_no'].isin(attr_filtered_test['site_no'])]
forcing_temp_renamed_filter_test.reset_index().to_feather('input/forcing/forcing_99_test.feather', version = 1)
attr_filtered.reset_index().to_feather('input/attr/attr_99_test.feather', version = 1)
```

On line 41 of StreamTemp-Integ.py, set the `Action` variable to [2] to only make prediction, not train
` 41         Action = [0, 2]`

Set `forcing_list` and `attr_list` (lines 19-20, seen above) to the new file names, and run `StreamTemp-Integ.py` as described above.
 
Now compare the predictions and observations as before:
```
obs = np.load('TempDemo/FirstRun/epochs2000_batch50_rho365_hiddensize100_Tstart20101001_Tend20141001_8/All-2010-2016/obs.npy')[:,:,0]
pred = np.load('TempDemo/FirstRun/epochs2000_batch50_rho365_hiddensize100_Tstart20101001_Tend20141001_8/All-2010-2016/pred.npy')[:,:,0]

pred.shape
obs.shape

```
