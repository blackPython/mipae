import numpy as np
import h5py
import os
import random

class MovingDSprites:

    def __init__(self, train, data_root, seq_len=20, num_sprites = 2, color = True, rotate = True, deterministic = True, return_factors = False):
        zip_file = np.load(os.path.join(data_root,"dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"), "r")
        self.train = train
        self.seq_len = seq_len
        self.num_sprites = num_sprites
        self.seed_is_set = False
        self.color = color
        self.rotate = rotate
        self.deterministic = deterministic
        self.return_factors = return_factors

        if not rotate:
            if train:
                self.data_split = np.load(os.path.join(data_root, "dsprites_train_split.npy"),"r")
            else:
                self.data_split = np.load(os.path.join(data_root, "dsprites_test_split.npy"), "r")
        else:
            if train:
                self.data_split = np.load(os.path.join(data_root, "dsprites_no_rot_train_split.npy"), "r")
            else:
                self.data_split = np.load(os.path.join(data_root, "dsprites_no_rot_test_split.npy"), "r")

        self.data = zip_file["imgs"]
        self.length = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
            random.seed(seed)
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.set_seed(index)
        x = np.zeros((self.seq_len, 64, 64, 3), dtype = np.float32)
        factors = np.zeros((self.seq_len, 5 * self.num_sprites))
        # factors = np.zeros((self.seq_len, 8))
        for n in range(self.num_sprites):
            if self.rotate:
                shape_idx, scale_idx = random.choice(self.data_split)
                so = np.random.randint(40)
            else:
                shape_idx, scale_idx, so = random.choice(self.data_split)

            sx = np.random.randint(32)
            sy = np.random.randint(32)

            dx = np.random.randint(-2,3)
            dy = np.random.randint(-2,3)
            if self.rotate:
                do = np.random.randint(-2,3)
            else:
                do = 0

            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1,3)
                        dx = np.random.randint(-2,3)
                elif sy >= 31:
                    sy = 31
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-2,0)
                        dx = np.random.randint(-2,3)

                if sx < 0:
                    sx = 0
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1,3)
                        dy = np.random.randint(-2,3)
                elif sx >= 31:
                    sx = 31
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-2,0)
                        dy = np.random.randint(-2,3)

                so = so%40
                data_index = sy + 32 * sx + 32 * 32 * so + 32 * 32 * 40 * scale_idx + 32 * 32 * 40 * 6 * shape_idx
                x[t,:,:,n] = np.copy(self.data[data_index].astype(np.float32))
                # factors[t,:] = np.stack([shape_idx, scale_idx, so, sx, sy, dx, dy, do])
                factors[t,n*5:(5*(n+1))] = np.stack([shape_idx, scale_idx, so, sx, sy])
                sy += dy
                sx += dx
                so += do

            front = np.random.randint(self.num_sprites)

        if self.color:
            for cc in range(self.num_sprites):
                if cc != front:
                    x[:,:,:,cc][x[:,:,:,front]>0] = 0
                
        else:
            x = x.sum(axis = -1, keepdims=True)
            x[x>1] = 1
        if not self.return_factors:
            return x
        else:
            return factors, x
