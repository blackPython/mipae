import numpy as np
import random
import os

class Mpi3dToy:

    def __init__(self, train, data_root, seq_len = 20, deterministic = False):
        self.train = train
        self.seq_len = seq_len
        self.seed_is_set = False
        self.deterministic = deterministic

        self.data = np.load(os.path.join(data_root, "mpi3d_toy_images.npy"), "r")
        if train:
            self.data_split = np.load(os.path.join(data_root, "mpi3d_toy_train_split.npy"), "r")
        else:
            self.data_split = np.load(os.path.join(data_root, "mpi3d_toy_test_split.npy"), "r")
        
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

        x = np.zeros((self.seq_len, 64, 64, 3), dtype = np.float32)

        color, shape, size, cam_height, background_color = random.choice(self.data_split)

        s_el = np.random.randint(40)
        s_az = np.random.randint(40)

        d_el = np.random.randint(-4,5)
        d_az = np.random.randint(-4,5)
        
        for t in range(self.seq_len):
            if s_el < 0:
                s_el = 0
                if self.deterministic:
                    d_el = -d_el
                else:
                    d_el = np.random.randint(1,5)
                    d_az = np.random.randint(-4,5)
            elif s_el >= 40:
                s_el = 39
                if self.deterministic:
                    d_el = -d_el
                else:
                    d_el = np.random.randint(-4,0)
                    d_az = np.random.randint(-4,5)
            
            if s_az < 0:
                s_az = 40 + s_az
                s_el = 39 - s_el
                d_el = -d_el
            elif s_az >= 40:
                s_az = s_az - 40
                s_el = 39 - s_el
                d_el = -d_el


            s_az = s_az%40
            data_index = self.get_data_index(color, shape, size, cam_height, background_color, s_el, s_az)
            x[t,:,:,:] = np.copy(self.data[data_index].astype(np.float32)/255.0)
            s_el += d_el
            s_az += d_az

        return x            

class Mpi3dReal(Mpi3dToy):
    def __init__(self, train, data_root, seq_len = 20, deterministic = False):
        self.train = train
        self.seq_len = seq_len
        self.seed_is_set = False
        self.deterministic = deterministic

        self.data = np.load(os.path.join(data_root, "mpi3d_real_images.npy"), "r")
        if train:
            self.data_split = np.load(os.path.join(data_root, "mpi3d_toy_train_split.npy"), "r")
        else:
            self.data_split = np.load(os.path.join(data_root, "mpi3d_toy_test_split.npy"), "r")
        
        self.length = len(self.data)
