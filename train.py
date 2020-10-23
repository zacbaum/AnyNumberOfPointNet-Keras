from data_loader import DataGenerator
from model import PointNet_cls

import os
import h5py
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("AGG")


def plot_history(history, result_dir, points):
    plt.plot(history.history["acc"], marker=".")
    plt.plot(history.history["val_acc"], marker=".")
    plt.title("model accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid()
    plt.legend(["acc", "val_acc"], loc="lower right")
    plt.savefig(os.path.join(result_dir, "model_accuracy-" + str(points) + ".png"))
    plt.close()

    plt.plot(history.history["loss"], marker=".")
    plt.plot(history.history["val_loss"], marker=".")
    plt.title("model loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid()
    plt.legend(["loss", "val_loss"], loc="upper right")
    plt.savefig(os.path.join(result_dir, "model_loss-" + str(points) + ".png"))
    plt.close()


def train(num_points, epochs, batch_size, lr):

    if not os.path.exists("./results/"):
        os.mkdir("./results/")

    train_file = "./ModelNet40/ply_data_train.h5"
    train_file = h5py.File(train_file, mode="r")
    test_file = "./ModelNet40/ply_data_test.h5"
    test_file = h5py.File(test_file, mode="r")

    nb_classes = 40

    train = DataGenerator(train_file, batch_size, num_points, nb_classes, train=True)
    val = DataGenerator(test_file, batch_size, num_points, nb_classes, train=False)

    model = PointNet_cls(nb_classes, num_points)
    model.summary()
    model.compile(
        optimizer=Adam(lr=lr), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    checkpoint = ModelCheckpoint(
        "./results/pointnet-best-" + str(num_points) + ".h5",
        monitor="val_acc",
        save_weights_only=True,
        save_best_only=True,
        verbose=0,
    )

    history = model.fit(
        train,
        steps_per_epoch=9840 // batch_size,
        epochs=epochs,
        validation_data=val,
        validation_steps=2468 // batch_size,
        callbacks=[checkpoint],
        verbose=2,
    )

    plot_history(history, "./results/", num_points)
    model.save_weights("./results/pointnet-" + str(num_points) + ".h5")


if __name__ == "__main__":

    train(2048, 100, 32, 0.0001)
    train(1024, 100, 32, 0.0001)
    train(512, 100, 32, 0.0001)
    train(256, 100, 32, 0.0001)
