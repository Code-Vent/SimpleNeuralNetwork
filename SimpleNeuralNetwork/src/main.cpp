#include"net.h"
#include<iostream>
#include<vector>
#include"mnist.h"


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

void mnist_test() {
	vector<vector<double>> train_input_set;
	vector<vector<double>> train_output_set;
	vector<vector<double>> test_input_set;
	vector<vector<double>> test_output_set;
	vector<vector<double>> test_result;

	auto ret = mnist::parseCSVFile("./mnist/mnist_train.csv", train_input_set, train_output_set, 40000);
	ret = mnist::parseCSVFile("./mnist/mnist_train.csv", test_input_set, test_output_set, 20000, 40000);
	vector<int> layers{ 784,15,10 };
	net::network nnet(layers, 1, makeReluLayer, makeSoftmaxLayer);
	nnet.train(train_input_set, train_output_set, 1);
	nnet.test(test_input_set, test_result);
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
	//mnist_test();
	xor_test();	
	return 0;
}