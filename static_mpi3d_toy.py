import numpy as np
import random
import os

class Mpi3dToy:

    def __init__(self, data_root, return_factors = False):
        self.seed_is_set = False
        self.return_factors = return_factors
        self.data = np.load(os.path.join(data_root, "mpi3d_toy_images.npy"), "r")
        
        self.length = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
            random.seed(seed)

    def __len__(self):
        return self.length
    
    def get_data_index(self,color, shape, size, cam_height, backgroud_color, el, az):
        return color * 115200 + shape * 28800 + size * 14400 + cam_height * 4800 + backgroud_color * 1600 + el * 40 + az

    def __getitem__(self, index):
        self.set_seed(index)

        color_idx = np.random.randint(4)
        shape_idx = np.random.randint(4)
        size_idx = np.random.randint(2)
        cam_idx = np.random.randint(3)
        back_idx = np.random.randint(3)

        s_el = np.random.randint(40)
        s_az = np.random.randint(40)

        f_c = np.array([color_idx, shape_idx, size_idx, cam_idx, back_idx])
        f_p = np.array([s_el, s_az])
        
        x = self.data[self.get_data_index(color_idx, shape_idx, size_idx, cam_idx, back_idx, s_el, s_az)].astype(np.float32)/255.0

        if self.return_factors:
            return [f_c,f_p], x
        else:
            return x
                   
class Mpi3dReal(Mpi3dToy):
        def __init__(self, data_root, return_factors = False):
            self.seed_is_set = False
            self.return_factors = return_factors
            self.data = np.load(os.path.join(data_root, "mpi3d_real_images.npy"), "r")
            
            self.length = len(self.data)
