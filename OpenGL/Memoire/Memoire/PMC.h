#ifndef _PMC_H
#define _PMC_H

#include <vector>

using namespace std;

typedef struct {
	double loss;
    double out;
    vector<double> weight;
    vector<double> weightPrevious;
    vector<double> weightSave;
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
	int maxIterations;
	double learningRate;
	double inertiaRate;
	bool useSigmoid;

	void InitializeRandoms();
	double RandomDouble(double Low, double High);
	void InitializeWeight();
	void InitializeStruct(const char* fileName, vector<TrainingStruct> & listTraining);
	void SaveWeights();
	void RestoreWeights();
	double FunctionSigmoid(double wx);
	double FunctionThreshold(double wx);
public:
	PMC(vector<int> sizePMC, int maxIterations, double learningRate, double inertiaRate, bool useSigmoid);
	~PMC(void);
	void LaunchLearning(const char* fileName);
	void Evaluate(const char* fileName);
};

#endif