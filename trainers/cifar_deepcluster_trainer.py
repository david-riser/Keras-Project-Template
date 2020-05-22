import numpy as np
import os

from base.base_trainer import BaseTrain
from sklearn.cluster import MiniBatchKMeans


class CifarDeepClusterTrainer(BaseTrain):

    def __init__(self, model, data, config):
        super(CifarDeepClusterTrainer, self).__init__(model, data, config)

    def init_callbacks(self):
        pass
        
    def clustering_target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T
        
    def train(self):

        # First initialize the model cluster centroids.  This is
        # done with kmeans.  Since the dataset it large, we use
        # mini-batch kmeans. 
        print("[INFO] Initializing cluster centroids w/ Kmeans (mini-batch).")
        if self.config.data_loader.batch_size < self.config.model.n_clusters:
            print("[FATAL] Mini-batch KMeans requires a batch size at least as large as the number of clusters.")
            exit()

        self.kmeans = MiniBatchKMeans(n_clusters=self.config.model.n_clusters)

        for epoch in range(self.config.trainer.kmeans_epochs):
            for batch in range(self.config.trainer.kmeans_batches_per_epoch):
                (x_batch, y_batch) = next(self.data.get_train_flow())
                self.kmeans.partial_fit(
                    self.model.backbone.predict(x_batch)
                )

        self.model.model.get_layer("clustering_layer").set_weights(
            [self.kmeans.cluster_centers_]
        )

        print("[INFO] Done!  Starting the training.")
        update_interval = int(np.ceil(self.config.trainer.batches / self.config.trainer.target_updates))
        if update_interval == 0:
            update_interval = 1
        
        print("[INFO] Using update interval {}".format(update_interval))
    
        kld_loss = []
        loss = np.inf
        for ite in range(self.config.trainer.batches):

            if ite % update_interval == 0:
            
                (x_batch, y_batch) = next(self.data.get_train_flow())
                while len(x_batch) != self.config.data_loader.batch_size:
                    (x_batch, y_batch) = next(self.data.get_train_flow())            

                q = self.model.model.predict(x_batch, verbose=0)
                p = self.clustering_target_distribution(q)


            (x_batch, y_batch) = next(self.data.get_train_flow())
            if len(x_batch) == self.config.data_loader.batch_size:
                loss = self.model.model.train_on_batch(x=x_batch, y=p)
                kld_loss.append(loss)

            print("[INFO] Epoch: {0}, Loss: {1:8.4f}".format(ite, loss))
            
        self.loss = kld_loss
        
