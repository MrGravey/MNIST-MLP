import numpy as np
import csv
import math

NUM_HIDDEN = 100
NUM_OUTPUT = 10
LEARN_RATE = 0.1
MOMENTUM = 0.9
RAND_SEED = 21
MAX_EPOCH = 50


def generateWeights(num_vectors: int, num_inputs: int):
    global RAND_SEED
    weights = np.zeros((num_vectors, num_inputs))

    for i in range(0, num_vectors):
        np.random.seed(RAND_SEED)
        weight_vector = np.random.uniform(-0.5, 0.5, [num_inputs])
        weights[i] = weight_vector
        RAND_SEED = RAND_SEED + 2

    return weights


def generateTargets(label: int):
    if not (label < NUM_OUTPUT):
        raise ValueError(
            "label("
            + str(label)
            + ") must not be greater than or equal to outputs("
            + str(NUM_OUTPUT)
            + ")"
        )
    targets = np.full((1, NUM_OUTPUT), 0.1)
    targets[0][label] = MOMENTUM

    return targets


def getData(file_name: str):
    with open(file_name, "r") as mnist_csv:
        reader = csv.reader(mnist_csv)
        header = list(next(reader))
        matrix = np.array(list(reader)).astype(int)

        # Create target vector
        targets = np.atleast_2d(matrix[:, 0])

        # Create matrix of input vectors
        inputs = matrix[:, 1:] / 255

    return [targets, inputs]


def sigmoidFunction(x):
    return 1 / (1 + math.exp(-x))


def outputErrorFunction(o, t):
    if o.shape != t.shape:
        raise ValueError(
            "shapes "
            + str(o.shape)
            + " and "
            + str(t.shape)
            + " not of the same dimensions"
        )

    a = np.zeros((o.shape))

    # Calculate error term
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            a[r][c] = o[r][c] * (1 - o[r][c]) * (t[r][c] - o[r][c])

    return a


def hiddenErrorFunction(h, e):
    if h.shape != e.shape:
        raise ValueError(
            "shapes "
            + str(h.shape)
            + " and "
            + str(e.shape)
            + " not of the same dimensions"
        )

    a = np.zeros((h.shape))

    # Calculate error term
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            a[r][c] = h[r][c] * (1 - h[r][c]) * e[r][c]

    return a


def accuracyFunction(labels, input_values):
    total = 0
    correct = 0
    for ex in range(len(input_values)):
        label = labels[0][ex]

        # Generate hidden values
        hidden_values = sigmoid(
            np.atleast_2d(
                np.dot(input_weights, input_values[ex]) + input_bias_weights.T
            )
        )

        # Generate output values
        output_values = sigmoid(
            np.atleast_2d(np.dot(hidden_weights, hidden_values.T) + hidden_bias_weights)
        )

        guess = np.argmax(output_values)

        if guess == label:
            correct += 1

        total += 1

    return correct / total


def confusionFunction(labels, input_values):
    confusion = np.zeros((10, 10))

    for ex in range(0, len(input_values)):
        label = labels[0][ex]

        # Generate hidden values
        hidden_values = sigmoid(
            np.atleast_2d(
                np.dot(input_weights, input_values[ex]) + input_bias_weights.T
            )
        )

        # Generate output values
        output_values = sigmoid(
            np.atleast_2d(np.dot(hidden_weights, hidden_values.T) + hidden_bias_weights)
        )

        guess = np.argmax(output_values)
        confusion[label][guess] += 1

    return confusion


# Convert sigmoid function to NumPy ufunc.
sigmoid = np.frompyfunc(sigmoidFunction, 1, 1)

# Pre-process test data
test_data = getData("mnist_test.csv")
test_labels = test_data[0]
test_values = test_data[1]

# Pre-process train data
train_data = getData("mnist_train.csv")
input_labels = train_data[0]
input_values = train_data[1]

# Generate weights
input_bias_weights = generateWeights(NUM_HIDDEN, 1)
input_weights = generateWeights(NUM_HIDDEN, len(input_values[0]))
hidden_bias_weights = generateWeights(NUM_OUTPUT, 1)
hidden_weights = generateWeights(NUM_OUTPUT, NUM_HIDDEN)

print("pre-processing done!")

acc_file = open("accuracy.csv", "w")

print("epoch, test_data, train_data")
acc_file.write("epoch,test_data,train_data\n")

# MLP Training
for epoch in range(MAX_EPOCH):

    # Get data-set accuracy
    test_acc = accuracyFunction(test_labels, test_values)
    train_acc = accuracyFunction(input_labels, input_values)

    # Write data-set accuracy
    print(str(epoch) + ", " + str(test_acc) + ", " + str(train_acc))
    acc_file.write(str(epoch) + "," + str(test_acc) + "," + str(train_acc) + "\n")

    # Loop through each example
    for ex in range(len(input_values)):
        # Generate targets
        targets = generateTargets(input_labels[0][ex])

        # Generate hidden values
        hidden_values = sigmoid(
            np.atleast_2d(
                np.dot(input_weights, input_values[ex]) + input_bias_weights.T
            )
        )

        # Generate output values
        output_values = sigmoid(
            np.atleast_2d(np.dot(hidden_weights, hidden_values.T) + hidden_bias_weights)
        )

        # Calculate error terms
        errorOutputTerm = outputErrorFunction(output_values.T, targets)

        errorHiddenTerm = hiddenErrorFunction(
            hidden_values, np.dot(errorOutputTerm, hidden_weights)
        )

        # Update weights
        hidden_weights = hidden_weights + LEARN_RATE * np.dot(
            errorOutputTerm.T, hidden_values
        )
        hidden_bias_weights = hidden_bias_weights + LEARN_RATE * errorOutputTerm.T

        input_weights = input_weights + LEARN_RATE * np.dot(
            errorHiddenTerm.T, np.atleast_2d(input_values[ex].T)
        )
        input_bias_weights = input_bias_weights + LEARN_RATE * errorHiddenTerm.T

# Get data-set accuracy
test_acc = accuracyFunction(test_labels, test_values)
train_acc = accuracyFunction(input_labels, input_values)

# Write data-set accuracy
print(str(MAX_EPOCH) + ", " + str(test_acc) + ", " + str(train_acc))
acc_file.write(str(MAX_EPOCH) + "," + str(test_acc) + "," + str(train_acc) + "\n")

confusion = confusionFunction(test_labels, test_values)
np.savetxt("confusion.csv", confusion, fmt="%d", delimiter=",")
