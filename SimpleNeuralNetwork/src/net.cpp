#include"net.h"
#include<assert.h>
#include<math.h>
#include<time.h>
#include<iostream>
#include<limits.h>

using namespace std;
using namespace net;

net::layer::layer(int neurons, int outputs)
{
	assert(neurons > 0);
	ww.resize(neurons);
	wdw.resize(neurons);
	out.resize(outputs);
	in.resize(neurons);
	bias.resize(neurons);
	dz.resize(neurons);
	if (outputs == 0) {
		cost.resize(neurons);
	}
	for (int i = 0; i < neurons; ++i) {
		ww[i].resize(outputs);
		wdw[i].resize(outputs);
	}
	initialize();
}

const vector<netfloat_t>& net::layer::feedforward(const vector<netfloat_t>& in_)
{
	// TODO: insert return statement here
	assert(in.size() == in_.size());
	activation(in_, in);
	if (out.size() > 0)
		forward();
	return out;
}

const vector<netfloat_t>& net::layer::updateWeights(layer& prev, netfloat_t alpha)
{
	// TODO: insert return statement here
	for (int i = 0; i < dz.size(); ++i) {
		for (int j = 0; j < prev.ww.size(); ++j) {
			prev.ww[j][i] -= alpha * dz[i] * prev.in[j];
		}
	}
	for (int j = 0; j < dz.size(); ++j) {
		bias[j] -= alpha * dz[j];
	}
	return out;
}

void net::layer::outputLayerDerivative(const vector<netfloat_t>& desiredOut)
{
	static vector<netfloat_t> dout;
	static vector<netfloat_t> delta;
	derivative(in, dout);
	computeDelta(in, desiredOut, delta);
	computeCost(delta);
	for (int i = 0; i < dz.size(); ++i) {
		dz[i] = dout[i] * delta[i];
	}
	dout.resize(0);
	delta.resize(0);
}

void net::layer::hiddenLayerDerivative(const layer& next)
{
	vector<netfloat_t> dhidden;
	for (int j = 0; j < dz.size(); ++j) {
		netfloat_t sum = 0;
		for (int k = 0; k < next.dz.size(); ++k) {
			sum += next.dz[k] * ww[j][k];
		}
		dz[j] = sum;
	}
	derivative(in, dhidden);
	for (int i = 0; i < dz.size(); ++i) {
		dz[i] *= dhidden[i];
	}
	dhidden.resize(0);
}

void net::layer::computeDelta(const vector<netfloat_t>& calcOut, const vector<netfloat_t>& desiredvalue, vector<netfloat_t>& delta)
{
	for (int i = 0; i < calcOut.size(); ++i) {
		delta.push_back(calcOut[i] - desiredvalue[i]);
	}
}

void net::layer::computeCost(const vector<netfloat_t>& delta)
{
	static netfloat_t full_cost = 0.0;
	static int n = 1;
	int j;
	float tmpcost = 0;
	float tcost = 0;
	for (j = 0; j < delta.size(); j++)
	{
		tmpcost = delta[j];
		cost[j] = (tmpcost * tmpcost) / 2;
		tcost = tcost + cost[j];
	}
	full_cost = (full_cost + tcost) / n;
	n++;
	cout << "Full Cost: " << full_cost << endl;
}

void net::layer::computeDeltaWeights()
{
	for (int i = 0; i < wdw.size(); ++i) {
		for (int k = 0; k < wdw[0].size(); ++k) {
			wdw[i][k] = (dz[i] * ww[i][k]) + (.5 * wdw[i][k]);
		}
	}
}

void net::layer::initialize()
{
	for (int k = 0; k < ww.size(); ++k) {
		for (int i = 0; i < ww[0].size(); ++i) {
			auto r = ((netfloat_t)rand() / RAND_MAX);
			ww[k][i] = r;
			wdw[k][i] = 0;
		}
	}
	for (int i = 0; i < bias.size(); ++i) {
		auto r = ((netfloat_t)rand() / RAND_MAX);
		bias[i] = r;
	}
}

void net::layer::forward()
{
	for (int i = 0; i < out.size(); ++i) {
		netfloat_t sum = 0.0;
		for (int j = 0; j < in.size(); ++j) {
			sum += ww[j][i] * in[j];
		}
		out[i] = sum;
	}
}

void net::relu::activation(const vector<netfloat_t>& in_, vector<netfloat_t>& out_)
{
	for (int i = 0; i < out_.size(); ++i) {
		out_[i] = in_[i] + bias[i];
		if (out_[i] < 0.0) {
			out_[i] *= 0.01;
		}
	}
}

void net::relu::derivative(const vector<netfloat_t>& x, vector<netfloat_t>& dx)
{
	//assert(x.size() == dx.)
	for (int i = 0; i < x.size(); ++i) {
		if (x[i] > 0.0) {
			dx.push_back(1.0);
		}
		else {
			dx.push_back(0.01);
		}
	}
}

void net::sigmoid::activation(const vector<netfloat_t>& in_, vector<netfloat_t>& out_)
{
	for (int i = 0; i < out_.size(); ++i) {
		out_[i] = 1.0 / (1.0 + exp(-(in_[i] + bias[i])));
	}
}

void net::sigmoid::derivative(const vector<netfloat_t>& x, vector<netfloat_t>& dx)
{
	for (int i = 0; i < x.size(); ++i) {
		auto d = x[i] * (1.0 - x[i]);
		dx.push_back(d);
	}
}

net::network::network(vector<int>& lay, int num_iter, layerFactory hidden, layerFactory output)
{
	assert(lay.size() > 2);
	layers.resize(lay.size());
	int i = 0;
	layers[i] = new input(lay[i], lay[i + 1]);
	for (i = 1; i < lay.size() - 1; ++i) {
		layers[i] = hidden(lay[i], lay[i + 1]);
	}
	layers[i] = output(lay[i], 0);
	num_iteration = num_iter;
}

net::network::~network()
{
	for (int i = 0; i < layers.size() - 1; ++i) {
		delete layers[i];
	}
}

void net::network::train(const vector<vector<netfloat_t>>& in, const vector<vector<netfloat_t>>& out, int epoch, netfloat_t alpha)
{
	for (int m = 0; m < num_iteration; ++m) {
		for (int i = 0; i < in.size(); ++i) {
			trainHelper(&in[i], &out[i], epoch, alpha);
		}
	}
}

void net::network::test(const vector<vector<netfloat_t>>& in, vector<vector<netfloat_t>>& out)
{
	for (int i = 0; i < in.size(); ++i) {
		int k = 0;
		auto curr_in = &in[i];
		layer* curr_lay = nullptr;
		while (k < layers.size()) {
			curr_lay = layers[k++];
			curr_in = &curr_lay->feedforward(*curr_in);
		}
		out.push_back(curr_lay->in);
	}
}

void net::network::trainHelper(const vector<netfloat_t>* in, const vector<netfloat_t>* out, int epoch, netfloat_t alpha)
{
	for (int i = 0; i < epoch; ++i) {
		int k = 0;
		auto curr_out = out;
		auto curr_in = in;
		layer* curr_lay = nullptr;
		while (k < layers.size()) {
			curr_lay = layers[k++];
			curr_in = &curr_lay->feedforward(*curr_in);
		}
		curr_lay->outputLayerDerivative(*curr_out);
		for (k = layers.size() - 1; k > 0; --k) {
			layers[k - 1]->hiddenLayerDerivative(*layers[k]);
		}

		for (k = layers.size() - 1; k > 0; --k) {
			layers[k]->updateWeights(*layers[k - 1], alpha);
		}
	}
}

void net::arctan::activation(const vector<netfloat_t>& in_, vector<netfloat_t>& out_)
{
	for (int i = 0; i < out_.size(); ++i) {
		out_[i] = std::atan(in_[i] + bias[i]);
	}
}

void net::arctan::derivative(const vector<netfloat_t>& x, vector<netfloat_t>& dx)
{
	for (int i = 0; i < x.size(); ++i) {
		auto d = 1.0 / (1.0 + x[i] * x[i]);
		dx.push_back(d);
	}
}

void net::input::activation(const vector<netfloat_t>& in_, vector<netfloat_t>& out_)
{
	for (int i = 0; i < out_.size(); ++i) {
		out_[i] = in_[i];
	}
}

void net::input::derivative(const vector<netfloat_t>& x, vector<netfloat_t>& dx)
{
	for (int i = 0; i < x.size(); ++i) {
		dx.push_back(1.0);
	}
}

void net::softmax::activation(const vector<netfloat_t>& in_, vector<netfloat_t>& out_)
{
	netfloat_t sum = 0;
	netfloat_t max = in_[0];
	for (int i = 1; i < in_.size(); ++i) {
		if (in_[i] > max)
			max = in_[i];
	}
	for (int i = 0; i < out_.size(); ++i) {
		sum += exp(in_[i]);
	}
	for (int i = 0; i < out_.size(); ++i) {
		out_[i] = exp(in_[i] - max) / sum;
	}
}

void net::softmax::derivative(const vector<netfloat_t>& x, vector<netfloat_t>& dx)
{
	for (int i = 0; i < x.size(); ++i) {
		for (int j = 0; j < x.size(); ++j) {
			if (i != j) {
				//dx.push_back(-x[i] * x[j]);
			}
			else {
				dx.push_back(x[i] * (1.0 - x[i]));
			}
		}
	}
}
