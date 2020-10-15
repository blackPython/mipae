import numpy as np
from sklearn.feature_selection.mutual_info_ import _compute_mi
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma
import tensorflow.compat.v1 as tf
import metrics.frechet_video_distance as fvd
import ndd

EPS = 1e-8

#Almost sklearn implementation (just some changes to allow for continous random variable)
def _compute_mi_cd(c, d, n_neighbors):
    """Compute mutual information between continuous and discrete variables.

    Parameters
    ----------
    c : ndarray, shape (n_samples,)
        Samples of a continuous random variable.

    d : ndarray, shape (n_samples,)
        Samples of a discrete random variable.

    n_neighbors : int
        Number of nearest neighbors to search for each point, see [1]_.

    Returns
    -------
    mi : float
        Estimated mutual information. If it turned out to be negative it is
        replace by 0.

    Notes
    -----
    True mutual information can't be negative. If its estimate by a numerical
    method is negative, it means (providing the method is adequate) that the
    mutual information is close to 0 and replacing it by 0 is a reasonable
    strategy.

    References
    ----------
    .. [1] B. C. Ross "Mutual Information between Discrete and Continuous
       Data Sets". PLoS ONE 9(2), 2014.
    """
    n_samples = c.shape[0]
    if len(c.shape) == 1:
        c = c.reshape([-1,1])
    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)
    nn = NearestNeighbors()

    for label in np.unique(d, axis = 0):
        mask = np.all(d == label, axis = -1)
        count = np.sum(mask)
        if count > 1:
            k = min(n_neighbors, count - 1)
            nn.set_params(n_neighbors=k)
            nn.fit(c[mask])
            r = nn.kneighbors()[0]
            radius[mask] = np.nextafter(r[:, -1], 0)
            k_all[mask] = k
        label_counts[mask] = count

    # Ignore points with unique labels.
    mask = label_counts > 1
    n_samples = np.sum(mask)
    label_counts = label_counts[mask]
    k_all = k_all[mask]
    c = c[mask]
    radius = radius[mask]

    nn.set_params(algorithm='kd_tree')
    nn.fit(c)
    ind = nn.radius_neighbors(radius=radius, return_distance=False)
    m_all = np.array([i.size for i in ind])

    mi = (digamma(n_samples) + np.mean(digamma(k_all)) -
          np.mean(digamma(label_counts)) -
          np.mean(digamma(m_all + 1)))
    return max(0, mi)

def discrete_entropy(x):
    _, count = np.unique(x, return_counts=True, axis = 0)
    return ndd.entropy(count, estimator = "Grassberger")

def mig_metric(representations, factors):
    z_c, z_p = representations
    f_c, f_p = factors
    e_f_c, e_f_p = discrete_entropy(f_c), discrete_entropy(f_p)
    score = 0
    for factor, entropy in zip([f_c,f_p],[e_f_c,e_f_p]):
        content_mutual = _compute_mi_cd(z_c, factor, 3)
        position_mutual = _compute_mi_cd(z_p, factor, 3)
        print(content_mutual, position_mutual)
        score += (content_mutual - position_mutual if content_mutual > position_mutual else position_mutual - content_mutual)/ entropy
    
    score /= 2

    return score

def compute_fvd_embedding(videos):
    batch_size = videos.get_shape().as_list()[0]
    videos = fvd.preprocess(videos,(224,224))
    videos = tf.split(videos,batch_size//16,axis = 0)
    embedings = []
    for video in videos:
        embedings = fvd.create_id3_embedding(video)
    embedings = tf.concat(embedings,axis = 0)
    return embedings

def compute_fvd(real_frames, fake_frames):
    real_frames, fake_frames = real_frames.transpose((1,0,3,4,2))*255.0, fake_frames.transpose((1,0,3,4,2))*255.0
    if real_frames.shape[-1] == 1:
        real_frames = np.repeat(real_frames,3,axis=-1)
        fake_frames = np.repeat(fake_frames,3,axis=-1)
    real_frames = tf.convert_to_tensor(real_frames)
    fake_frames = tf.convert_to_tensor(fake_frames)
    real_embed = compute_fvd_embedding(real_frames)
    fake_embed = compute_fvd_embedding(fake_frames)
    fvd_graph = fvd.calculate_fvd(real_embed,fake_embed)
    fvd_graph = tf.stop_gradient(fvd_graph)
    print("running")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        result = sess.run(fvd_graph)
    return result

class PersistanceFVD:
    def __init__(self,batch_size = 16, video_length = 30):
        self.video_length = video_length
        self.batch_size = batch_size
        self.real_placeholder = tf.placeholder(tf.float32,(batch_size,video_length,64,64,3))
        self.fake_placeholder = tf.placeholder(tf.float32,(batch_size,video_length,64,64,3))
        self.real_embed = compute_fvd_embedding(self.real_placeholder)
        self.fake_embed = compute_fvd_embedding(self.fake_placeholder)
        self.fvd_graph = fvd.calculate_fvd(self.real_embed,self.fake_embed)
        self.fvd_graph = tf.stop_gradient(self.fvd_graph)
        self.__sess = tf.Session()
        self.__sess.run(tf.global_variables_initializer())
        self.__sess.run(tf.tables_initializer())
    
    def __call__(self, real_frames, fake_frames):
        """
            real_frames, fake_frames : 5D numpy ndarrays [video_length,batch_size,num_channels,64,64]
            returns: fvd distance between the videos
        """
        real_frames, fake_frames = real_frames.transpose((1,0,3,4,2))*255.0, fake_frames.transpose((1,0,3,4,2))*255.0
        if real_frames.shape[-1] == 1:
            real_frames = np.repeat(real_frames,3,axis=-1)
            fake_frames = np.repeat(fake_frames,3,axis=-1)
        return self.__sess.run(self.fvd_graph, {
                    self.real_placeholder : real_frames,
                    self.fake_placeholder : fake_frames
                })


