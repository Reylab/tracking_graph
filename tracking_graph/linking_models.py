import numpy as np

class LinkingModel():
    def __init__(self, data_inputs):
        pass
    def fit(self, we):
        assert False,'not implemented'
        
    def get_model_assignation_rates(self, we):
        assert False,'not implemented'

    @classmethod
    def creator(cls, **kwargs):
        def creator():
            return cls(**kwargs)
        return creator
    
    def get_unit_ids(self):
        return self.clusters.copy()

class EuclideanClassifier(LinkingModel):
    def __init__(self, std_mult):
        self.var_mul = std_mult**2

    def fit(self, we):
        sorting = we.sorting
        self.model = {}
        for u in sorting.get_unit_ids():
            w_std = np.sum((we.get_template(u,mode='std')**2))
            w_template = we.get_template(u,mode='average')
            self.model[u] = {'mean':w_template, 'sumvar': w_std}
        self.nknown_cells = len(self.model)
        self.clusters = sorting.get_unit_ids().copy()
        
    def get_model_assignation_rates(self, waveforms):
        assign = np.zeros(self.nknown_cells)
        distances = np.empty(self.nknown_cells)
        for i in range(waveforms.shape[0]):
            for m_i, m_unit_info in enumerate(self.model.values()):
                d = np.sum((m_unit_info['mean']-waveforms[i,:,:])**2)
                if d <= m_unit_info['sumvar']*self.var_mul:
                    distances[m_i] = d
                else:
                    distances[m_i] = np.inf
            min_m_i = np.argmin(distances)
            if distances[min_m_i] != np.inf:
               assign[min_m_i] = assign[min_m_i] + 1
        return assign/waveforms.shape[0]

class MahalanobisClassifier(LinkingModel):
    """
    For single channel recordings
    """
    def __init__(self, std_mult):
        self.var_mul = std_mult**2

    def fit(self, we):
        sorting = we.sorting

        for u in sorting.get_unit_ids():
            s = we.get_waveforms(u).shape
            assert s[0]>=s[1], f"Not enough spikes in unit {u}"

        self.model = {}
        for u in sorting.get_unit_ids():
            invcov = np.linalg.inv(np.cov(we.get_waveforms(u)[:,:,0],rowvar=False))
            w_template = we.get_template(u,mode='average')
            w_std = s[1] #after whitening each dimension has std=1
            self.model[u] = {'mean':w_template, 'sumvar': w_std, 'invcov': invcov}

        self.nknown_cells = len(self.model)
        self.clusters = sorting.get_unit_ids().copy()
        
    def get_model_assignation_rates(self, waveforms):
        assign = np.zeros(self.nknown_cells)
        distances = np.empty(self.nknown_cells)
        for i in range(waveforms.shape[0]):
            for m_i, m_unit_info in enumerate(self.model.values()):
                daux = m_unit_info['mean']-waveforms[i,:,:]
                d = daux[:,:,0].T @ m_unit_info['invcov'] @ daux[:,:,0]
                if d <= m_unit_info['sumvar']*self.var_mul:
                    distances[m_i] = d
                else:
                    distances[m_i] = np.inf
            min_m_i = np.argmin(distances)
            if distances[min_m_i] != np.inf:
               assign[min_m_i] = assign[min_m_i] + 1
        return assign/waveforms.shape[0]

