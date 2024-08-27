
from spikeinterface import extractors
from pathlib import Path
from spikeinterface.extractors.matlabhelpers import MatlabHelper
import numpy as np

class Waveclus_Waveforms():
    """
    This little class is a ducktype and temporal solution to load waveforms 
    from WaveClus with the basic interface of a WaveformExtractor
    """
    def __init__(self, finput) -> None:

        if Path(finput).is_dir():
            spfile = finput/Path('raw1_spikes.mat')
            timesfile = finput/Path('times_results.mat')
        else:
            spfile = finput
            timesfile = finput
        self1 = MatlabHelper(spfile)
        wc_snippets = self1._getfield("spikes")
        self2 = MatlabHelper(timesfile)
        cluster_classes = self2._getfield("cluster_class")
        classes = cluster_classes[:, 0]
        #remove not classified spikes
        self.wc_snippets = wc_snippets[classes>0,:]
        self.classes = classes[classes>0]
        self.sorting = extractors.WaveClusSortingExtractor(timesfile)

    def get_template(self,u,mode='average'):
        
        if isinstance(u,list):
            ix = np.zeros_like(self.classes).astype(bool)
            for el in u:
                ix= np.logical_or(ix,self.classes==el)
        else:
            ix=self.classes==u
        if mode=='average':
            return np.expand_dims(self.wc_snippets[ix,:],2).mean(0)
        if mode=='std':
            return np.expand_dims(self.wc_snippets[ix,:],2).std(0)
        return None

    def get_waveforms(self,u):
        return np.expand_dims(self.wc_snippets[self.classes==u,:],axis=2)
