import numpy as np
import copy
import matplotlib.pyplot as plt

class Syndexies():

    def __init__(self, path) -> None:
        self.data = dict()
        self.data['DATA_0'] = self._read_data(path)

    def _read_data(self, path):
        ans = dict()

        if path.split('.')[-1] == 'csv':
            data = np.genfromtxt(path, dtype=float, delimiter=',', names=True)
            ans['X'], ans['Y']  = 0.0, 0.0
            ans['DEPT'] = data['DEPT']
            ans['CURVE'] = data['GR']
            ans['IDX'] = np.asarray([i for i in range(data['DEPT'].size)])
            
            return ans
        
    
    def add_data(self, shift_depth=None, sample_rate=None, interpolate_rate=None, XY=None, noise=None):

        new_data = copy.deepcopy(self.data['DATA_0'])

        #Create Lat, Long
        if XY:
            new_data['X'], new_data['Y'] = XY
        else:
            new_data['X'], new_data['Y'] = np.random.uniform(-10,10), np.random.uniform(-10,10)

        #Shift on depth
        if not shift_depth:
            shift_depth = np.random.uniform(-2000,2000)        
        new_data['DEPT'] = new_data['DEPT'] + shift_depth
        
        #Transformations on data
        if sample_rate:
            self._sample(new_data, sample_rate)

        if interpolate_rate:
            self._interpolate(new_data, interpolate_rate)
        
        if noise:
            self._noise(new_data, noise)

        self.data[f'DATA_{len(self.data)}'] = new_data

    def _sample(self, data, sample_rate):
        keys = ['DEPT', 'CURVE', 'IDX']

        for key in keys:
            data[key] = data[key][::sample_rate]

    def _interpolate(self, data, interpolate_rate):
        
        start, end = data['DEPT'][0], data['DEPT'][-1]
        step = abs((end-start)/data['DEPT'].size)
        new_dept = np.linspace(start-step/2, end+step/2, data['DEPT'].size * interpolate_rate)
        
        data['CURVE'] = np.interp(new_dept, data['DEPT'], data['CURVE'])
        data['DEPT'] = new_dept
        data['IDX'] = np.repeat(data['IDX'], interpolate_rate)

    def _noise(self, data, noise):

        if type(noise) == int:
            bias, std = 0, noise
        else:
            bias, std = noise
        data['CURVE'] += np.random.normal(bias, std, data['CURVE'].size)

    def view(self):
        _, ax = plt.subplots(figsize=(12,7))
        offset = 0

        for data in self.data:
            ax.plot(data['CURVE']+offset, data['DEPT'])
            offset = data['CURVE'].max