#pragma once
#include<vector>


using namespace std;

namespace net {
	typedef double netfloat_t;
	using weights = vector<netfloat_t>;
	class network;

	class layer {
		friend class network;
	public:
		layer(int neurons, int outputs);
		const vector<netfloat_t>& feedforward(const vector<netfloat_t>& in);
		const vector<netfloat_t>& updateWeights(layer& prev, netfloat_t alpha);
		void outputLayerDerivative(const vector<netfloat_t>& desiredOut);
		void hiddenLayerDerivative(const layer& next);
		virtual void activation(const vector<netfloat_t>& in, vector<netfloat_t>& out) = 0;
		virtual void derivative(const vector<netfloat_t>& x, vector<netfloat_t>& dx) = 0;
	protected:
		vector<weights> ww;
		vector<weights> wdw;
		vector<netfloat_t> out;
		vector<netfloat_t> in;
		vector<netfloat_t> bias;
		vector<netfloat_t> dz;
		vector<netfloat_t> cost;
	private:
		void initialize();
		void forward();
		void computeDelta(const vector<netfloat_t>& calcOut, const vector<netfloat_t>& value, vector<netfloat_t>& delta);
		void computeCost(const vector<netfloat_t>& delta);
		void computeDeltaWeights();
	};

	class input : public layer {
	public:
		input(int neurons, int outputs) : layer(neurons, outputs) {}
		void activation(const vector<netfloat_t>& in, vector<netfloat_t>& out) override;
		void derivative(const vector<netfloat_t>& x, vector<netfloat_t>& dx) override;
	};

	class relu : public layer {
	public:
		relu(int neurons, int outputs) : layer(neurons, outputs) {}
		void activation(const vector<netfloat_t>& in, vector<netfloat_t>& out) override;
		void derivative(const vector<netfloat_t>& x, vector<netfloat_t>& dx) override;
	};

	class sigmoid : public layer {
	public:
		sigmoid(int neurons, int outputs) : layer(neurons, outputs) {}
		void activation(const vector<netfloat_t>& in, vector<netfloat_t>& out) override;
		void derivative(const vector<netfloat_t>& x, vector<netfloat_t>& dx) override;
	};

	class softmax : public layer {
	public:
		softmax(int neurons, int outputs) : layer(neurons, outputs) {}
		void activation(const vector<netfloat_t>& in, vector<netfloat_t>& out) override;
		void derivative(const vector<netfloat_t>& x, vector<netfloat_t>& dx) override;
	};

	class arctan : public layer {
	public:
		arctan(int neurons, int outputs) : layer(neurons, outputs) {}
		void activation(const vector<netfloat_t>& in, vector<netfloat_t>& out) override;
		void derivative(const vector<netfloat_t>& x, vector<netfloat_t>& dx) override;
	};

	class network {
	public:
		typedef layer* (*layerFactory)(int neurons, int outputs);
		network(vector<int>& layers, int num_iteration, layerFactory hidden, layerFactory output);
		~network();
		void train(const vector< vector<netfloat_t>>& in, const vector< vector<netfloat_t>>& out, int epoch = 10, netfloat_t alpha = 0.15);
		void test(const vector< vector<netfloat_t>>& in, vector< vector<netfloat_t>>& out);
	private:
		void trainHelper(const vector<netfloat_t>* curr_in, const vector<netfloat_t>* curr_out, int epoch, netfloat_t alpha);
		vector<layer*> layers;
		int num_iteration;
	};
}