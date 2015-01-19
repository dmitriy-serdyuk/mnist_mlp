usage: mnist_mlp.py [-h] [--filename FILENAME] [--num_feat NUM_FEAT]
                    [--num_layers NUM_LAYERS] [--out_size OUT_SIZE]
                    [--batch_size BATCH_SIZE] [--n_epochs N_EPOCHS]
                    [--learn_rate LEARN_RATE] [--n_hiddens N_HIDDENS]
                    [--model_file MODEL_FILE]

Run MLP for classification

optional arguments:

  -h, --help            show this help message and exit

  --filename FILENAME   File which contains pickled datasetIt should have the
                        following structure:a thee element tuple (train
                        dataset, validationdataset, testing dataset) and which
                        dataset isa pair of numpy arrays containing features
                        and labels. The numpy arrays have dimensionality
                        (number of data points, point dimensionality)

  --num_feat NUM_FEAT   Number of input features

  --num_layers NUM_LAYERS
                        Number of layers of MLP

  --out_size OUT_SIZE   Size of output layer

  --batch_size BATCH_SIZE
                        Size of batch

  --n_epochs N_EPOCHS   Number of epochs

  --learn_rate LEARN_RATE
                        Learning rate

  --n_hiddens N_HIDDENS
                        Number of hidden units

  --model_file MODEL_FILE
                        Model file
