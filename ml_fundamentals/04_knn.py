# import torch
import time
import numpy as np


n, d, m=500, 20, 4
k=5


class KNN:
    def __init__(self, X_train: np.ndarray, Y_train: np.ndarray, k: int=5):
        self.x = X_train
        self.y = Y_train
        self.k = k

    def predict(self, z: np.ndarray):
        x2 = np.tile(np.diag(np.matmul(self.x, self.x.T)), (m, 1)).T # N, M
        xz = np.matmul(self.x, z.T)  # N, M
        zz = np.tile(np.diag(np.matmul(z, z.T)), (n, 1)) # N, M
        dists = np.sqrt(np.clip(x2 - 2 * xz + zz, min=0.))

        nn = np.argsort(dists.T, axis=-1)[:, :self.k]
        assert nn.shape == (m, self.k)
        values, counts = np.unique(nn, axis=-1, return_counts=True)
        preds = values[np.argmax(counts, axis=-1)]
        return preds


if __name__ == "__main__":

    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    X=np.random.random((n,d))
    Z=np.random.random((m,d))
    Y=np.random.randint(3,size=n)

    s_time = time.time()
    knn = KNN(X, Y)
    preds = knn.predict(Z)
    time_taken = time.time() - s_time
    print(f"{time_taken=}")
    print(preds)
