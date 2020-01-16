import numpy as np
from data_reader import news_datareader
import pickle

input_dimension = 1000
epoch = 1
learning_rate = 0.1

def predict(weights, inputs):
        summation = np.dot(weights, inputs.T)
        activation = np.sign(summation)
        return [activation, summation]

def train(training_inputs, labels):
        weights = np.zeros((1, 1000))
        for _ in range(epoch):
            for inputs, label in zip(training_inputs, labels):
                prediction = predict(weights, inputs)
                if prediction [0] != label:
                    inputs = np.reshape(inputs, (1,-1))
                    weights += learning_rate * label * inputs
        return weights

def label_one_to_rest(label, class_number):
        new_label = []
        for item in list(label):
            item = int(item)
            if item == class_number:
                item = 1
            else:
                item = -1
            new_label.append(item)
        return new_label

if __name__ == '__main__':
    vect_train_dense, Y_train, vect_test_dense, Y_test = news_datareader().TF_IDF_datareader()
    all_classifier = []
    for classifier_num in range(0, 20):
        new_train_label = label_one_to_rest(Y_train, classifier_num)
        update_weight = train(vect_train_dense, new_train_label)
        print(update_weight)
        with open('model/perceptron'+str(classifier_num)+'.pickle', 'wb') as f:
            pickle.dump(update_weight,f)
    for classifier_num in range(0, 20):
        with open('model/perceptron'+str(classifier_num)+'.pickle', 'rb') as f:
            all_classifier.append(pickle.load(f))
    correct = 0
    for inputs, label in zip(vect_test_dense, Y_test):
        predication_to_compare =[]
        for each_classifier in all_classifier:
            final_prediction = predict(each_classifier, inputs)[1]
            predication_to_compare.append(float(final_prediction))
        if predication_to_compare.index(max(predication_to_compare)) == label:
            correct +=1
    print(correct/len(Y_test))