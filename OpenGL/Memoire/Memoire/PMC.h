#ifndef _PMC_H
#define _PMC_H

#include <vector>

using namespace std;

typedef struct {
    vector<double> weight;
    double hx;
	double in;
	double loss;
} Node;

typedef struct {
    vector<double> data;
    vector<double> result;
} TrainingStruct;

class PMC
{
private:
	// This vector contains the number of nodes by layer
	vector<vector<Node>> perceptron;
	double RandomDouble(double Low, double High);
	double FunctionSigmoid(double wx);
	double FunctionThreshold(double wx);
	void InitializeWeight();
public:
	PMC(vector<int> sizePMC);
	~PMC(void);
	bool LaunchLearning(int maxIteration, double learningRate, bool useSigmoid, vector<TrainingStruct> listTraining);
};

#endif