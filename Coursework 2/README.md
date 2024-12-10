# Report on Optical Recognition System for Handwritten Digits

## 1. Introduction
The objective of this project is to develop a machine learning system capable of categorizing handwritten digits using the Optical Recognition of Handwritten Digits dataset from the UCI Machine Learning Repository. The dataset is divided into two subsets: `cw2DataSet1` and `cw2DataSet2`.

A Multi-Layer Perceptron (MLP) model was implemented from scratch to train and evaluate the system. The primary goal was to achieve high accuracy in digit recognition while providing insights into the system's performance through a well-defined training and testing strategy. This project serves as a demonstration of fundamental principles in neural network design and application.

---

## 2. Algorithm and Design

### Algorithm: Multi-Layer Perceptron (MLP)
The MLP is a feedforward neural network comprising an input layer, one or more hidden layers, and an output layer. It is effective for tasks requiring non-linear decision boundaries, such as image recognition.

### Key Features of MLP Implementation:
- **Input Layer**: Encodes pixel intensity features from the dataset.
- **Hidden Layer**: Captures complex patterns in the data; the number of neurons is set to 64, matching the dimensionality of the input.
- **Output Layer**: Outputs probabilities for each digit class (0–9).
- **Activation Function**: The sigmoid activation function introduces non-linearity, enabling the network to model complex relationships. And the step activation function for creating binary results for the output neuron
- **Optimization**: Gradient descent with backpropagation ensures efficient weight updates.
- **Evaluation Metrics**: Accuracy quantifies the model's performance.

### Weight Initialization:
- **Weights Between Input and Hidden Layers (WH)**: Randomly initialized with values ranging between -1 and 1, structured as a (64, 64) matrix.
- **Weights Between Hidden and Output Layers (WO)**: Similarly initialized, with a (10, 64) shape to map 64 hidden neurons to 10 output classes.

### Rationale for Choosing MLP:
The MLP's ability to model non-linear relationships makes it well-suited for recognizing handwritten digits with diverse styles and orientations. While more advanced architectures like Convolutional Neural Networks (CNNs) are preferred for image-based tasks, implementing an MLP highlights the foundational aspects of neural network design and operation.

---

## 3. Data Usage
The project utilizes two subsets of the Optical Recognition of Handwritten Digits dataset:
- **cw2DataSet1**: Contains 2,809 samples for training.
- **cw2DataSet2**: Contains 2,809 samples for testing.

### Structure of the Dataset:
- **Features**: Each sample contains 64 features representing pixel intensity values of an 8×8 image of a handwritten digit.
- **Target Class**: A single column indicates the digit label (0–9).

### Testing Strategy:
The model was trained exclusively on `cw2DataSet1` and tested on `cw2DataSet2`, ensuring a clear separation between training and testing phases.

---

## 4. Results

### Training Accuracy:
The training process demonstrated steady accuracy improvement over 550 cycles:
- **Cycle 1**: 32.93%
- **Cycle 50**: 95.34%
- **Cycle 100**: 97.54%
- **Cycle 250**: 99.07%
- **Cycle 500**: 99.86%
- **Cycle 550**: 100.00%

These results indicate the model successfully learned the training data relationships, achieving perfect accuracy.

### Testing Accuracy:
Testing the trained model on `cw2DataSet2` yielded the following result:
- **Accuracy**: 100.00%

### Observations:
The MLP achieved perfect accuracy on the test dataset, demonstrating strong generalization to unseen data. This result underscores the model's robustness for handwritten digit recognition.

---

## 5. Conclusion
This project successfully implemented a Multi-Layer Perceptron to classify handwritten digits, achieving notable accuracy. The training and testing accuracies reached 100%, validating the model's robustness and generalization capability.
