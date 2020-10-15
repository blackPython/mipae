import numpy as np
import h5py
import os
import random

class StaticDSprites:
    def __init__(self, data_root, num_sprites = 2, color = True, return_factors = True):
        zip_file = np.load(os.path.join(data_root,"dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"), "r")
        self.num_sprites = num_sprites
        self.color = color
        self.return_factors = return_factors
        self.seed_is_set = False

        self.data = zip_file["imgs"]
        self.length = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
            random.seed(seed)

    def __len__(self):
        return self.length
    
    def get_data_index(self, shape, scale, so, sx, sy):
        return sy + 32 * sx + 32 * 32 * so + 32*32*40 * scale + 32*32*40*6 * shape

    def __getitem__(self, index):
        self.set_seed(index)
        f_c = np.zeros((3*self.num_sprites+1 if self.color else 0), dtype = np.int64)
        f_p = np.zeros((2*self.num_sprites), dtype = np.int64)
        x = np.zeros((64,64,3), dtype = np.float32)
        for n in range(self.num_sprites):
            shape_idx = np.random.randint(3)
            scale_index = np.random.randint(6)
            so = np.random.randint(40)
            sx = np.random.randint(32)
            sy = np.random.randint(32)
            f_c[3*n:(3*(n+1))] = np.array([shape_idx, scale_index, so])
            f_p[2*n:(2*(n+1))] = np.array([sx,sy])
            x[:,:,n] = np.copy(self.data[self.get_data_index(shape_idx, scale_index, so, sx, sy)].astype(np.float32))
        
        if self.color:
            front = np.random.randint(self.num_sprites)
            f_c[-1] = front
            for cc in range(self.num_sprites):
                if cc != front:
                    x[:,:,cc][x[:,:,front]>0] = 0
        else:
            x = x.sum(axis = -1, keepdims=True)
            x[x>1] = 1
        
        if self.return_factors:
            return [f_c, f_p], x
        else:
            return x




    