#include"mnist.h"
#include<iostream>
#include<fstream>
#include<vector>

using namespace std;


int mnist::parseCSVFile(const string& filename, vector<vector<double>>& inSet,
	vector<vector<double>>& outSet, int samplesToRead, int offset, bool includeLabels)
{
	ifstream fin(filename);

	if (!fin.is_open()) {
		cerr << "Error opening file!" << endl;
		return 0;
	}
	inSet.resize(0);
	outSet.resize(0);
	string line = "";
	string token = "";
	string delimiter = ",";
	int sample_index = 0;
	getline(fin, line); //Reads Header
	//cout << line << endl;
	while (getline(fin, line) && samplesToRead) {
		uint32_t pos = 0;
		uint32_t line_len = line.size();
		vector<double> label;
		bool label_read = false;
		vector<double> in(784, 0);
		vector<double> out;
		int k = 0;
		bool process_sample = sample_index >= offset;
		while (pos < line_len && process_sample && k < in.size()) {
			token = line.substr(pos, line.substr(pos).find(delimiter));
			pos += token.size() + 1;
			int value = atoi(token.c_str());
			if (includeLabels && !label_read) {
				label.resize(10);
				label.at(value) = 1.0;
				label_read = true;
			}
			else if (!includeLabels || label_read) {
				in[k++] = (double)value / 255.0;
			}
		}

		for (auto e : label) {
			out.push_back(e);
		}
		++sample_index;
		if (process_sample) {
			--samplesToRead;
			inSet.push_back(in);
			outSet.push_back(out);
		}
	}
	fin.close();
	return samplesToRead;
}