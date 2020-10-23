import numpy as np
import random
from tensorflow.keras.utils import Sequence, to_categorical


class DataGenerator(Sequence):
    def __init__(self, data, batch_size, size=2048, nb_classes=40, train=True):
        self.data = data
        self.batch_size = batch_size
        self.size = size
        self.nb_classes = nb_classes
        self.train = train

        self.nb_sample = self.data["data"].shape[0]
        self.indexes = np.arange(self.nb_sample)

    def __len__(self):
        return int(np.floor(self.nb_sample / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        data_temp = [self.data["data"][k] for k in indexes]
        labels_temp = [self.data["label"][k] for k in indexes]

        # Generate augmented batch data
        X, y = self.__generator(data_temp, labels_temp)

        return X, y

    def on_epoch_end(self):
        if self.train:
            np.random.shuffle(self.indexes)

    def __generator(self, data, labels):
        X = []
        Y = []
        for d, l in zip(data, labels):
            item = d
            label = l
            if self.train:
                rotate = random.randint(0, 1)
                jitter = random.randint(0, 1)
                if rotate:
                    item = rotate_point_cloud(item)
                if jitter:
                    item = jitter_point_cloud(item)
            if self.size < item.shape[0]:
                item = item[
                    np.random.choice(item.shape[0], self.size, replace=False), :
                ]

            X.append(item)
            Y.append(label[0])
        Y = to_categorical(np.array(Y), self.nb_classes)
        return np.array(X), np.array(Y)


def rotate_point_cloud(data):
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
    rotated_data = np.dot(data.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(data, sigma=0.01, clip=0.05):
    N, C = data.shape
    assert clip > 0
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_data += data
    return jittered_data
