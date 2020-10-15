import numpy as np

data = np.load("mpi3d_real.npz")
data = data["images"]
np.save("mpi3d_real_images.npy", data)
