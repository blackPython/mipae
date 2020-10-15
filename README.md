#Mutual Information based method for Unsupervised Disentanglement of Video Representations
This is the offical implementation of "Mutual Information based method for Unsupervised Disentanglement of Video Represenations" accepted for publication in ICPR 2020. The paper will be uploaded to arXiv soon.
This code is developed using pytorch 1.4.0, make sure you use the same version for smooth execution. 

To train or test for Moving Dsprites or MPI3D-Real datasets you need to download the datasets fist. To download Dsprites run the following command:

`bash download_dsprites.sh`

Similarly for MPI3D-Toy dataset:

`bash download_mpi3d_real.sh`

### Training

Two train scripts are used one for traning the auto-encoder ans another to train LSTM. 

To train auto-encoder for Moving mnist run the following command

`python3 train_autoencoder.py --no_color --num_channels 1 --dataset mnist --niters 400`

To train LSTM for Moving mnist run the following command (< checkpoint > is the latest autoencoder checkpoint) :

`python3 train_lstm.py --encoder_checkpoint <checkpoint> --dataset mnist --no_color --num_channels 1 --niters 200`

Similarly to train for Moving Dsprites dataset:

`python3 train_autoencoder.py --dataset dsprites --niters 400`

`python3 train_lstm.py --encoder_checkpoint <checkpoint> --dataset dsprites --niters 200`

Similarly to train for Moving MPI3D_Real dataset:

`python3 train_autoencoder.py --dataset mpi3d_real --niters 200 --z_dims 10`

`python3 train_lstm.py --encoder_checkpoint <checkpoint> --dataset mpi3d_real --niters 200 --z_dims 10`

### Evaluation

To evaluate the auto-encoder run the following command:

`python3 test_ours.py --checkpoint <checkpoint> --dataset <dataset>`

Where < checkpoint > is the latest auto-encoder checkpoint. < dataset > is dataset to use, if dataset is mnist append `--no_color` and `--num_channels` arguments at the end and `--z_dims` if dataset is mpi3d_real.

To evaluate the LSTM run the following command:

`python3 test_lstm.py --ae_checkpoint <ae_checkpoint> --lstm_checkpoint <lstm_checkpoint> --dataset <dataset>`

Where < ae_checkpoint > is the latest auto-encoder checkpoint and < lstm_checkpoint > is latest LSTM checkpoint. < dataset > is dataset to use, if dataset is mnist append `--no_color` and `--num_channels` arguments at the end and `--z_dims` if dataset is mpi3d_real.

To compute the proposed disentanglement metric:

`python3 compute_disentanglement_metric.py --checkpoint <checkpoint> --dataset <dataset>`

Where < checkpoint > is the latest auto-encoder checkpoint. < dataset > is dataset to use, if dataset is mnist append `--no_color` and `--num_channels` arguments at the end and `--z_dims` if dataset is mpi3d_real.
