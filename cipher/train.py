import argparse
import pandas as pd
from tensorflow import keras
from cipher import load


def main(model_name, data_path, args):

    # Load data
    x_train, y_train, x_valid, y_valid, x_test, y_test = load.standard_data(
        data_path, reverse_comp=args.rc
    )

    # Import model from the zoo as singular animal
    # equivalent of `from model_zoo import model_name as animal` where model_name is
    # evaluated at runtime.

    # TODO: this code is difficult to understand. If there is buggy behavior related to
    # this line, it will be difficult to debug. I propose refactoring this to use a
    # method that grabs the model function from a dictionary of the known models.
    animal = __import__(
        "cipher.model_zoo." + model_name, globals(), locals(), [model_name], 0
    )

    # Build model
    num_labels = y_train.shape[1]
    N, L, A = x_train.shape
    model = animal.model(input_shape=(L, A), output_shape=num_labels)

    # set up optimizer and metrics
    auroc = keras.metrics.AUC(curve="ROC", name="auroc")
    aupr = keras.metrics.AUC(curve="PR", name="aupr")
    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)

    # compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=[auroc, aupr])

    # set up callbacks
    es_callback = keras.callbacks.EarlyStopping(
        monitor="val_aupr",
        patience=args.es_patience,
        verbose=1,
        mode="max",
        restore_best_weights=True,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_aupr",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=1e-7,
        mode="max",
        verbose=1,
    )
    # Fit model
    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        validation_data=(x_valid, y_valid),
        callbacks=[es_callback, reduce_lr],
    )

    # Save fit history
    history_df = pd.DataFrame(
        list(history.history.items()), columns=["column1", "column2"]
    )
    history_path = args.output_path_prefix + "_history.tsv"
    history_df.to_csv(history_path, sep="\t", index=False)

    # Save model weights
    weights_path = args.output_path_prefix + "_weights.hdf5"
    model.save_weights(weights_path)


if __name__ == "__main__":
    # ---------- Parse arguments ----------
    # EG.
    # python train_single_task.py -m deepbind -d ../data/test_dataset.h5 \
    #   -o ../results/test_deepbind_trial1" -e 200 -bs 64

    parser = argparse.ArgumentParser(
        description="Train model on single-task dataset and save outputs.",
        epilog=(
            "Example usage: python train_single_task.py -epochs 200 deepbind"
            " ../data/test_dataset.h5 ../results/test_deepbind_trial1"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--model_name",
        metavar="MODEL_NAME",
        type=str,
        help="Model name (from model zoo) to use",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        metavar="/PATH/TO/DATA/",
        type=str,
        help="Path to HDF5 data file",
    )
    parser.add_argument(
        "-o",
        "--output_path_prefix",
        metavar="/PATH/TO/OUTPUT/PLUS_PREFIX",
        type=str,
        help=(
            "Path (if not current working directory) plus filename prefix for two"
            " outputs: weights and history"
        ),
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        metavar="BATCH_SIZE",
        type=int,
        help="Number of epochs to train each model",
        default=100,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="EPOCHS",
        type=int,
        help="Number of epochs to train each model",
        default=200,
    )
    parser.add_argument(
        "-rc",
        metavar="TRAIN_REVERSE_COMP",
        type=bool,
        help="Augment dataset with reverse complement sequences",
        default=False,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        metavar="LEARNING_RATE",
        type=float,
        help="Learning rate for Adam optimizer",
        default=0.001,
    )

    parser.add_argument(
        "-f",
        "--lr_factor",
        metavar="REDUCELRONPLATEAU_FACTOR",
        type=float,
        help=(
            "Factor for learning rate reduction on plateau callback"
            " (see tensorflow.keras.callbacks.ReduceLROnPlateau())"
        ),
        default=0.2,
    )
    parser.add_argument(
        "-lrp",
        "--lr_patience",
        metavar="REDUCELRONPLATEAU_PATIENCE",
        type=float,
        help=(
            "Patience for learning rate reduction on plateau callback"
            " (see tensorflow.keras.callbacks.EarlyStopping())"
        ),
        default=3,
    )
    parser.add_argument(
        "-esp",
        "--es_patience",
        metavar="EARLYSTOPPING_PATIENCE",
        type=float,
        help=(
            "Patience for early stopping callback"
            " (see tensorflow.keras.callbacks.ReduceLROnPlateau())"
        ),
        default=10,
    )

    args = parser.parse_args()

    # run main
    main(args.model_name, args.data_path, args)
