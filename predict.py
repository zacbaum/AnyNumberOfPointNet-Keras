from data_loader import DataGenerator
from model import PointNet_cls
import h5py
import numpy as np
from tensorflow.keras.optimizers import Adam
import sklearn.metrics


def predict(model_points, predict_points):

    print(
        "Predicting with {} points on a model trained with {} points.".format(
            predict_points, model_points
        )
    )

    test_file = "./ModelNet40/ply_data_test.h5"
    test_file = h5py.File(test_file, mode="r")

    nb_classes = 40

    val = DataGenerator(test_file, 32, predict_points, nb_classes, train=False)

    model = PointNet_cls(nb_classes, predict_points)
    model.load_weights("./results/pointnet-" + str(model_points) + ".h5")
    pred = np.argmax(model.predict(val), axis=1)

    labels = np.squeeze(
        [test_file["label"][x] for x in range(test_file["label"].shape[0])]
    )
    labels = np.array([int(x) for x in labels])

    print(
        "Accuracy: {:.5}%\n".format(
            100 * sklearn.metrics.accuracy_score(labels[: pred.shape[0]], pred)
        )
    )


if __name__ == "__main__":

    predict(2048, 2048)
    predict(2048, 1024)
    predict(2048, 512)
    predict(2048, 256)

    predict(1024, 2048)
    predict(1024, 1024)
    predict(1024, 512)
    predict(1024, 256)

    predict(512, 2048)
    predict(512, 1024)
    predict(512, 512)
    predict(512, 256)

    predict(256, 2048)
    predict(256, 1024)
    predict(256, 512)
    predict(256, 256)
