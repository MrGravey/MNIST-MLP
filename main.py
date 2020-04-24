import numpy as np
import csv
import math

NUM_HIDDEN = 25
NUM_OUTPUT = 10
LEARN_RATE = 0.1
MOMENTUM = 0.9
RAND_SEED = 21


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

        # Create bias vector
        biases = np.ones((1, len(inputs))).T

        # Add bias vector to matrix of input vectors
        all_inputs = np.c_[biases, inputs]

    return [targets, all_inputs]


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


def accuracyFunction(lables, input_values, input_weights, hidden_weights):
    total = 0
    correct = 0
    for ex in range(0, len(input_values)):
        target = lables[0][ex]

        # Generate hidden values
        hidden_values = sigmoid(np.atleast_2d(np.dot(input_weights, input_values[ex])))

        # Add hidden bias
        hidden_values = np.c_[1, hidden_values]

        # Generate output values
        output_values = sigmoid(np.atleast_2d(np.dot(hidden_weights, hidden_values.T)))

        guess = np.argmax(output_values)

        if guess == target:
            correct += 1

        total += 1

    return correct / total


# Convert sigmoid function to NumPy ufunc.
sigmoid = np.frompyfunc(sigmoidFunction, 1, 1)

test_data = getData("mnist_test.csv")
l = test_data[0]
v = test_data[1]

train_data = getData("mnist_train.csv")
labels = train_data[0]
input_values = train_data[1]

input_weights = generateWeights(NUM_HIDDEN, len(input_values[0]))
hidden_weights = generateWeights(NUM_OUTPUT, NUM_HIDDEN + 1)

print("pre-processing done!")

print(accuracyFunction(l, v, input_weights, hidden_weights))

for epoch in range(0, 60):
    # Loop through each example
    for ex in range(0, len(input_values)):

        # Generate targets
        targets = generateTargets(labels[0][ex])

        # Generate hidden values
        hidden_values = sigmoid(np.atleast_2d(np.dot(input_weights, input_values[ex])))

        # Add hidden bias
        hidden_values = np.c_[1, hidden_values]

        # Generate output values
        output_values = sigmoid(np.atleast_2d(np.dot(hidden_weights, hidden_values.T)))

        # Calculate error terms
        errorOutputTerm = outputErrorFunction(output_values.T, targets)
        errorHiddenTerm = hiddenErrorFunction(
            np.atleast_2d(hidden_values[:, 1:].T),
            np.dot(hidden_weights.T[1:], errorOutputTerm.T),
        )

        # Update weights
        hidden_weights = hidden_weights + LEARN_RATE * np.dot(
            errorOutputTerm.T, hidden_values
        )
        input_weights = input_weights + LEARN_RATE * np.dot(
            errorHiddenTerm, np.atleast_2d(input_values[ex])
        )

    print(accuracyFunction(l, v, input_weights, hidden_weights))
