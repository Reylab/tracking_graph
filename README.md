# Tracking_Graph
![GitHub License](https://img.shields.io/github/license/reylab/tracking_graph)

Tracking_Graph is a tool for track units across spike sorting solutions. This package is part of the PhD 
This project is part of my PhD research at [University Name]

## Installation

Install Tracking_Graph using pip:

```bash
pip install tracking_graph
```

## Usage

### Creating a Graph

```python
from tracking_graph import run_tg, get_tg_groups, EuclideanClassifier

# Create a classifier
modelcreator = EuclideanClassifier.creator(std_mult=3)  # Classify spikes with length > 3 std

# Run Tracking_Graph
G = run_tg(
    we_list,  # List of WaveformExtractor objects
    outputfile='/home/user/examplefolder/tg_data.hdf5',
    max_len=2,  # Maximum edge length, must be at least 1
    modelcreator=modelcreator
)

# Compute results programmatically (can be replaced by the GUI)
groups, sG, discarded = get_tg_groups(
    G,
    mintrack=3,  # Minimum number of segments for a cluster
    merge=True  # Apply criteria to merge splits
)

# Create a final results table
import pandas as pd

df = []
for gi, g in enumerate(groups):
    for c in g:
        df.append({'segment': c.segment,
                    'cluster': c.unit,
                    'tg_unit': gi})

results_table = pd.DataFrame(df)
```

### Exploring Simplified Graphs with GUI

Launch the graphical interface (Streamlit server) using:

```bash
tg_gui
```

### Additional Tools

Tracking_Graph provides a wrapper to load aligned waveforms from Wave_Clus clustering results, addressing limitations in SpikeInterface's waveform interpolation:

```python
from tracking_graph.spikeinterface_addons import Waveclus_Waveforms

path_times_file = '/home/user/examplefolder/times_example.mat'  # Full path to Wave_Clus result
we = Waveclus_Waveforms(path_times_file)  # Object with basic WaveformExtractor interface
```

## Limitations

- Tracking_Graph requires `spikeinterface<=0.100` due to its dependency on the `WaveformExtractor` class.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

##  About
For more projects and information, visit my GitHub profile: [ferchaure](https://github.com/ferchaure)