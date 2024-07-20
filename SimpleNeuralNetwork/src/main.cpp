#include"net.h"
#include<iostream>
#include<vector>
#include"mnist.h"
#include<assert.h>
#include<algorithm>

using namespace std;

net::layer* makeReluLayer(int neurons, int outputs) {
	return new net::relu(neurons, outputs);
}

net::layer* makeSigmoidLayer(int neurons, int outputs) {
	return new net::sigmoid(neurons, outputs);
}

net::layer* makeSoftmaxLayer(int neurons, int outputs) {
	return new net::softmax(neurons, outputs);
}

net::layer* makeArcTanLayer(int neurons, int outputs) {
	return new net::arctan(neurons, outputs);
}

double countCorrectPredictions(const vector<vector<double>>& original, const vector<vector<double>>& predicted) {
	assert(original.size() == predicted.size());
	auto findMaxIndex = [](const vector<double>& x) {
		int maxI = 0;
		auto max = x[0];
		for (int i = 1; i < x.size(); ++i) {
			if (x[i] > max) {
				maxI = i;
				max = x[i];
			}
		}
		return maxI;
	};
	int correctPredictions = 0;
	int i;
	for (i = 0; i < original.size(); ++i) {
		int j = findMaxIndex(original[i]);
		int k = findMaxIndex(predicted[i]);
		if (j == k) ++correctPredictions;
	}
	return (double)correctPredictions / i;
}

void mnist_test() {
	vector<vector<double>> train_input_set;
	vector<vector<double>> train_output_set;
	vector<vector<double>> test_input_set;
	vector<vector<double>> test_output_set;
	vector<vector<double>> test_result;

	auto ret = mnist::parseCSVFile("./mnist/mnist_train.csv", train_input_set, train_output_set, 40000);
	ret = mnist::parseCSVFile("./mnist/mnist_train.csv", test_input_set, test_output_set, 20000, 40000);
	vector<int> layers{ 784,15,10 };
	net::network nnet(layers, 3, makeArcTanLayer, makeSigmoidLayer);
	nnet.train(train_input_set, train_output_set, 3);
	nnet.test(test_input_set, test_result);
	auto rate = countCorrectPredictions(test_output_set, test_result);
	cout << "Prediction rate = " << rate << endl;
	return;
}

void xor_test() {
	const vector<vector<net::netfloat_t>> xor_in = {
		{1, 1}, { 0,0 }, { 1,0 }, { 0,1 }
	};

	const vector<vector<net::netfloat_t>> xor_out = {
		{0}, { 0 }, { 1}, { 1 }
	};
	vector<vector<net::netfloat_t>> xor_result;

	vector<int> layers{ 2,2,1 };
	net::network nnet(layers, 1000, makeReluLayer, makeSigmoidLayer);
	nnet.train(xor_in, xor_out);
	nnet.test(xor_in, xor_result);
	return;
}

int main() {
	mnist_test();
	//xor_test();	
	return 0;
}