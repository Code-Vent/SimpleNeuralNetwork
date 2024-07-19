#pragma once
#include<string>
#include<vector>

using namespace std;

namespace mnist {
	int parseCSVFile(const string& file,
		vector<vector<double>>& inputSet,
		vector<vector<double>>& outputSet,
		int samplesToRead,
		int offset = 0,
		bool includeLabels = true);
}