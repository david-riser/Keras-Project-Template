from base.base_model import BaseModel
from utils.factory import create
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer

class ClusteringLayer(Layer):
    """ Clustering layer. """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.clusters = self.add_weight(
            name='clusters',
            shape=(self.n_clusters, input_dim),
            initializer='glorot_uniform',
            trainable=True)
        super(ClusteringLayer, self).build(input_shape)
        
    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        layer_config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(layer_config.items()))


class CifarDeepClusterModel(BaseModel):

    def __init__(self, config):
        super(CifarDeepClusterModel, self).__init__(config)
        self.model_builder = create("tensorflow.keras.applications.{}".format(
            self.config.model.backbone
        ))

        print("[DEBUG] Building model.")
        self.build_model()

    def build_model(self):

        self.backbone = self.model_builder(
            weights='imagenet',
            pooling=self.config.model.pooling,
            include_top=False
        )
        print("[DEBUG] Setup backbone.")
        outputs = ClusteringLayer(
            n_clusters=self.config.model.n_clusters,
            name='clustering_layer',
            input_shape=2048)(self.backbone.output)

        print("[DEBUG] Setup clustering layer.")
        # Setup the clustering model.
        self.model = Model(inputs=self.backbone.input, outputs=outputs)
        self.model.compile(
              loss='kld',
              optimizer=self.config.model.optimizer
        )

        # Enable training of all the layers.
        for layer in self.model.layers:
            layer.trainable = True
