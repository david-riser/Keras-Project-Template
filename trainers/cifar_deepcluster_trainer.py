import os
from base.base_trainer import BaseTrainer
from sklearn.cluster import MiniBatchKMeans


class CifarDeepClusterTrainer(BaseTrainer):

    def __init__(self, model, data, config):
        super(CifarDeepClusterTrainer, self).__init__(model, data, config)

    def init_callbacks(self):
        pass
        
    def clustering_target_distribution(self):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T
        
    def train(self):

        # First initialize the model cluster centroids.  This is
        # done with kmeans.  Since the dataset it large, we use
        # mini-batch kmeans. 
        if self.config.data_loader.batch_size < self.config.model.n_clusters:
            print("[FATAL] Mini-batch KMeans requires a batch size at least as large as the number of clusters.")
            exit()

        self.kmeans = MiniBatchKMeans(n_clusters=self.config.model.n_clusters)

        for epoch in self.config.trainer.kmeans_epochs:
            for batch in self.config.trainer.kmeans_batches_per_epoch:
                self.kmeans.partial_fit(next(self.data.get_train_flow()))
        
        # Now start doing the model training
        update_interval = int(np.ceil(self.config.trainer.batches / self.config.trainer.target_updates))
        if update_interval == 0:
            update_interval = 1
        
        print("[INFO] Using update interval {}".format(update_interval))
    
        kld_loss = []
        loss = np.inf
        for ite in range(int(self.config.trainer.batches)):

            if ite % update_interval == 0:
            
                batch = next(self.data.get_train_flow())
                while len(batch) != batch_size:
                    batch = next(self.data.get_train_flow())            
                    q = self.model.predict(batch, verbose=0)
                    p = self.clustering_target_distribution(q)


            batch = next(self.data.get_train_flow())
            if len(batch) == batch_size:
                loss = self.model.train_on_batch(x=batch, y=p)
                kld_loss.append(loss)

            print("[INFO] Epoch: {0}, Loss: {1:8.4f}".format(ite, loss))
            
        self.loss = kld_loss
        
