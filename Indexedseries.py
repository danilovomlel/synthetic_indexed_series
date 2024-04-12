import numpy as np
import copy
import matplotlib.pyplot as plt

class Syndexies():

    def __init__(self, path) -> None:
        self.data = dict()
        self.data['DATA_0'] = self._read_data(path)
        self.new_data = None

    def _read_data(self, path):
        ans = dict()

        if path.split('.')[-1] == 'csv':
            data = np.genfromtxt(path, dtype=float, delimiter=',', names=True)
            ans['X'], ans['Y']  = 0.0, 0.0
            ans['DEPT'] = data['DEPT']
            ans['CURVE'] = data['GR']
            ans['IDX'] = np.asarray([i for i in range(data['DEPT'].size)])
            
            return ans

    def view(self):
        _, ax = plt.subplots(figsize=(12,15))
        offset = 0

        for key, data in self.data.items():
            ax.plot(data['CURVE']+offset, data['DEPT'])
            ax.invert_yaxis()
            offset += data['CURVE'].max()
    
    def add_data(self, shift_depth=None, XY=None):

        self.new_data = copy.deepcopy(self.data['DATA_0'])

        #Create Lat, Long
        if XY:
            self.new_data['X'], self.new_data['Y'] = XY
        else:
            self.new_data['X'], self.new_data['Y'] = np.random.uniform(-10,10), np.random.uniform(-10,10)

        #Shift on depth
        if not shift_depth:
            shift_depth = np.random.uniform(-100,100)        
        self.new_data['DEPT'] = self.new_data['DEPT'] + shift_depth
    
    def add_transformation(self, transform, parameters):
            
        transformations = {
            "compress": self._compress,
            "compress_local": self._compress_local,
            "expand": self._expand,
            "expand_local": self._expand_local,
            "remove_segment": self._remove_segment,
            "gaussian_noise": self._noise,
            "sine_noise": self._add_sine            
        }
        func = transformations.get(transform)
        if func:
            func(self.new_data, **parameters)
        else:
            print("Invalid input")

    def end_transformations(self):
        self.data[f'DATA_{len(self.data)}'] = self.new_data    

    def _compress(self, data, compress_rate):
        keys = ['DEPT', 'CURVE', 'IDX']

        for key in keys:
            data[key] = data[key][::compress_rate]

        start, end = data['DEPT'][0], data['DEPT'][-1]
        dep_len = abs(end-start)
        delta_compress = (dep_len - dep_len/compress_rate)/2
        data['DEPT'] = np.linspace(start+delta_compress, end-delta_compress, data['DEPT'].size)

    def _compress_local(self, data, compress_rate, extension=0.2):
        start = np.random.uniform(0, 1 - extension)
        start = int(start*data['DEPT'].size)
        end = int(start + extension*data['DEPT'].size)
        for item in ['CURVE', 'IDX']:
            data[item] = np.concatenate((data[item][:start], 
                                         data[item][start:end:compress_rate],
                                         data[item][end:]))
        total_length = abs(data['DEPT'][0] - data['DEPT'][-1])
        compressed_length = total_length*extension*(1 - 1/compress_rate)
        data['DEPT'] = np.linspace(data['DEPT'][0], data['DEPT'][-1] - compressed_length, data['IDX'].size)

    def _expand(self, data, expand_rate):
        
        start, end = data['DEPT'][0], data['DEPT'][-1]
        step = abs((end-start)/data['DEPT'].size)
        new_dept = np.linspace(start-step/2, end+step/2, data['DEPT'].size * expand_rate)
        
        data['CURVE'] = np.interp(new_dept, data['DEPT'], data['CURVE'])
        half = abs(end-start)/2
        data['DEPT'] = np.linspace(start-half*expand_rate, end+half*expand_rate, new_dept.size)
        data['IDX'] = np.repeat(data['IDX'], expand_rate)
    
    def _expand_local(self, data, expand_rate, extension=0.2):
        start = np.random.uniform(0, 1 - extension)
        start = int(start*data['DEPT'].size)
        original_len = int(extension*data['DEPT'].size)
        end = start+original_len

        inteprolated_indices = np.linspace(0,original_len, int(original_len*expand_rate))
        expanded_region = np.interp(inteprolated_indices, np.arange(original_len), data['CURVE'][start:end])
        data['CURVE'] = np.concatenate((data['CURVE'][:start],
                                        expanded_region,
                                        data['CURVE'][end:]))
        data['IDX'] = np.concatenate((data['IDX'][:start],
                                      np.repeat(data['IDX'][start:end], expand_rate),
                                      data['IDX'][end:]))
        
        total_length = abs(data['DEPT'][0] - data['DEPT'][-1])
        expanded_length = total_length*extension*(expand_rate - 1)
        data['DEPT'] = np.linspace(data['DEPT'][0], data['DEPT'][-1] + expanded_length, data['IDX'].size)

    def _remove_segment(Self, data, segment_size):
        start = np.random.uniform(0,1-segment_size)
        start = int(start*data['DEPT'].size)
        end = start + int(segment_size*data['DEPT'].size)
        dept_diff = abs(data['DEPT'][start] - data['DEPT'][end])

        for item in ['IDX', 'CURVE', 'DEPT']:
            data[item] = np.concatenate((data[item][:start],
                                         data[item][end:]))
        data['DEPT'][start:] -= dept_diff   

    def _noise(self, data, noise):

        if type(noise) == int:
            bias, std = 0, noise
        else:
            bias, std = noise
        data['CURVE'] += np.random.normal(bias, std, data['CURVE'].size)

    def _add_sine(self, data, T=2, amp=None, modulate=False):
        if not amp:
            amp = 0.2*max(data['CURVE'])
        
        sine = np.sin(np.linspace(0,2*np.pi*T, data['CURVE'].size))

        if modulate:
            data['CURVE'] *= sine*amp
        else:
            data['CURVE'] += sine*amp
