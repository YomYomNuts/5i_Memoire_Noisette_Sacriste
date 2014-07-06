#include "PMC.h"
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>


PMC::PMC(vector<int> sizePMC, int maxIterations, double learningRate, double inertiaRate, bool useSigmoid)
{
	this->maxIterations = maxIterations;
	this->learningRate = learningRate;
	this->inertiaRate = inertiaRate;
	this->useSigmoid = useSigmoid;
	for (unsigned int i = 0; i < sizePMC.size(); i++)
	{
		vector<Node> layer;

		int size = sizePMC.at(i);
		int sizePrev = 0;
		if (i > 0)
			sizePrev = sizePMC.at(i-1);

		for (int j = 0; j < size; ++j)
		{
			Node node;
			
			node.out = 1;
			node.loss = 0;
			for (int m = 0; m < sizePrev; ++m)
			{
				node.weight.push_back(0);
				node.weightPrevious.push_back(0);
				node.weightSave.push_back(0);
			}

			layer.push_back(node);
		}

		this->perceptron.push_back(layer);
	}
}

PMC::~PMC(void)
{
}

void PMC::InitializeRandoms()
{
	//  srand( (unsigned)time( NULL ) );
	srand(4711);
}

double PMC::RandomDouble(double Low, double High)
{
    double f = (double)rand() / RAND_MAX;
    return Low + f * (High - Low);
}

void PMC::InitializeWeight()
{
	for (unsigned int i = 1; i < this->perceptron.size(); i++)
	{
		vector<Node> * layer = &this->perceptron.at(i);
		int sizePrev = this->perceptron.at(i-1).size();

		for (unsigned int j = 0; j < layer->size(); ++j)
		{
			Node * node = &layer->at(j);
			for (int m = 0; m < sizePrev; ++m)
			{
				node->weight.at(m) = RandomDouble(-0.5, 0.5);
			}
		}
	}
}

void PMC::InitializeStruct(const char* fileName, vector<TrainingStruct> & listTraining)
{
	ifstream file;
	char line[1024];

	file.open (fileName, ifstream::in);

	while(file.getline(line, 1024)) 
	{
		string lineString(line);
		istringstream iss(lineString);
		vector<string> values;
		copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter<vector<string>>(values));
		TrainingStruct trainingStruct;
		for (int i = 0; i < values.size(); ++i)
		{
			if (i < this->perceptron.at(0).size())
				trainingStruct.data.push_back(atof(values.at(i).c_str()));
			else
				trainingStruct.result.push_back(atof(values.at(i).c_str()));
		}
		listTraining.push_back(trainingStruct);
	}

	file.close();
}

void PMC::SaveWeights()
{
	for (unsigned int i = 1; i < this->perceptron.size(); i++)
	{
		vector<Node> * layer = &this->perceptron.at(i);
		vector<Node> * prevLayer = &this->perceptron.at(i-1);
		for (unsigned int j = 0; j < layer->size(); j++)
		{
			Node * node = &layer->at(j);
			for (unsigned int k = 0; k < prevLayer->size(); k++) 
			{
				node->weightSave.at(k) = node->weight.at(k);
			}
		}
	}
}

void PMC::RestoreWeights()
{
	for (unsigned int i = 1; i < this->perceptron.size(); i++)
	{
		vector<Node> * layer = &this->perceptron.at(i);
		vector<Node> * prevLayer = &this->perceptron.at(i-1);
		for (unsigned int j = 0; j < layer->size(); j++)
		{
			Node * node = &layer->at(j);
			for (unsigned int k = 0; k < prevLayer->size(); k++) 
			{
				node->weight.at(k) = node->weightSave.at(k);
			}
		}
	}
}

double PMC::FunctionSigmoid(double wx)
{
    return 1.0 / (1.0 + exp(-wx));
}

double PMC::FunctionThreshold(double wx)
{
    return wx < 0.5 ? 0 : 1;
}

void PMC::LaunchLearning(const char* fileName)
{
	bool continueLearning;
	int numberIterationsTest = 0;
	bool firstIter = true;
	double dMinTestError = 0;
	vector<TrainingStruct> listTraining;
	
	InitializeStruct(fileName, listTraining);

	InitializeRandoms();
	InitializeWeight();

	do
	{
		continueLearning = true;
		++numberIterationsTest;

		// Learning
		for (unsigned int i = 0; i < listTraining.size(); ++i)
		{
			TrainingStruct training = listTraining.at(i);
			vector<Node> * layerInput = &this->perceptron.at(0);
			vector<Node> * layerOutput = &this->perceptron.at(this->perceptron.size() - 1);

			/* Propagate the inputs */
			for (unsigned int k = 0; k < layerInput->size(); ++k)
			{
				layerInput->at(k).out = training.data.at(k);
			}
			for(unsigned int j = 1; j < this->perceptron.size(); ++j)
			{
				vector<Node> * layer = &this->perceptron.at(j);
				vector<Node> * prevLayer = &this->perceptron.at(j-1);
				for(unsigned int k = 0; k < layer->size(); ++k)
				{
					Node * node = &layer->at(k);
					double sum = 0;
					for (unsigned int l = 0; l < prevLayer->size(); ++l)
						sum += node->weight.at(l) * prevLayer->at(l).out;
					if (this->useSigmoid)
						node->out = FunctionSigmoid(sum);
					else
						node->out = FunctionThreshold(sum);
				}
			}

			/* Propagate the loss */
			for (unsigned int k = 0; k < layerOutput->size(); ++k)
			{
				Node * node = &layerOutput->at(k);
				if (this->useSigmoid)
					node->loss = node->out * (1 - node->out) * (training.result.at(k) - node->out);
				else
					node->loss = node->out * (1 - node->out) * (training.result.at(k) - node->out);
			}
			for(int j = this->perceptron.size() - 2; j >= 0; --j)
			{
				vector<Node> * layer = &this->perceptron.at(j);
				vector<Node> * nextLayer = &this->perceptron.at(j+1);
				for(unsigned int k = 0; k < layer->size(); ++k)
				{
					Node * node = &layer->at(k);
					double sum = 0;
					for (unsigned int l = 0; l < nextLayer->size(); ++l)
                    {
                        Node * nodeNextLayer = &nextLayer->at(l);
						sum += nodeNextLayer->weight.at(k) * nodeNextLayer->loss;
                    }
					if (this->useSigmoid)
						node->loss = node->out * (1 - node->out) * sum;
					else
						node->loss = node->out * (1 - node->out) * sum;
				}
			}

			/* Update weight */
			for(unsigned int j = 1; j < this->perceptron.size(); ++j)
			{
				vector<Node> * layer = &this->perceptron.at(j);
				vector<Node> * prevLayer = &this->perceptron.at(j-1);
				for(unsigned int k = 0; k < layer->size(); ++k)
				{
					Node * node = &layer->at(k);
					for (unsigned int l = 0; l < prevLayer->size(); ++l)
					{
						Node * nodePrevLayer = &prevLayer->at(l);
						node->weight.at(l) += this->learningRate * nodePrevLayer->out * node->loss + this->inertiaRate * node->weightPrevious.at(l);
						node->weightPrevious.at(l) = this->learningRate * nodePrevLayer->out * node->loss;
					}
				}
			}
		}

		// Test
		double dAvgTestError = 0;
		for (unsigned int i = 0; i < listTraining.size(); ++i)
		{
			TrainingStruct training = listTraining.at(i);
			vector<Node> * layerInput = &this->perceptron.at(0);
			vector<Node> * layerOutput = &this->perceptron.at(this->perceptron.size() - 1);

			/* Propagate the inputs */
			for (unsigned int k = 0; k < layerInput->size(); ++k)
			{
				Node * node = &layerInput->at(k);
				node->out = training.data.at(k);
			}
			for(unsigned int j = 1; j < this->perceptron.size(); ++j)
			{
				vector<Node> * layer = &this->perceptron.at(j);
				vector<Node> * prevLayer = &this->perceptron.at(j-1);
				for(unsigned int k = 0; k < layer->size(); ++k)
				{
					Node * node = &layer->at(k);
					double sum = 0;
					for (unsigned int l = 0; l < prevLayer->size(); ++l)
						sum += node->weight.at(l) * prevLayer->at(l).out;
					if (this->useSigmoid)
						node->out = FunctionSigmoid(sum);
					else
						node->out = FunctionThreshold(sum);
				}
			}

			/* The loss */
			double dMAE = 0;
			for (unsigned int k = 0; k < layerOutput->size(); ++k)
			{
				Node * node = &layerOutput->at(k);
				dMAE += fabs(training.result.at(k) - node->out);
			}
			dMAE /= layerOutput->size();
			dAvgTestError += dMAE;
		}
		dAvgTestError /= listTraining.size();

		if(firstIter)
		{
			dMinTestError = dAvgTestError;
			firstIter = false;
		}
		if ( dAvgTestError < dMinTestError) 
		{
			dMinTestError = dAvgTestError;
			SaveWeights();
		}
		else if (dAvgTestError > 1.2 * dMinTestError) 
		{
			continueLearning = false;
			RestoreWeights();
		}
	} while(continueLearning && numberIterationsTest < this->maxIterations);
}

void PMC::Evaluate(const char* fileName)
{
	vector<TrainingStruct> listEvaluate;

	InitializeStruct(fileName, listEvaluate);

	for (unsigned int i = 0; i < listEvaluate.size(); ++i)
	{
		TrainingStruct training = listEvaluate.at(i);
		vector<Node> * layerInput = &this->perceptron.at(0);

		/* Propagate the inputs */
		for (unsigned int k = 0; k < layerInput->size(); ++k)
		{
			Node * node = &layerInput->at(k);
			node->out = training.data.at(k);
		}
		for(unsigned int j = 1; j < this->perceptron.size(); ++j)
		{
			vector<Node> * layer = &this->perceptron.at(j);
			vector<Node> * prevLayer = &this->perceptron.at(j-1);
			for(unsigned int k = 0; k < layer->size(); ++k)
			{
				Node * node = &layer->at(k);
				double sum = 0;
				for (unsigned int l = 0; l < prevLayer->size(); ++l)
					sum += node->weight.at(l) * prevLayer->at(l).out;
				if (this->useSigmoid)
					node->out = FunctionSigmoid(sum);
				else
					node->out = FunctionThreshold(sum);
			}
		}
	}
	
	vector<Node> * layerOutput = &this->perceptron.at(this->perceptron.size() - 1);
	printf("Carre    Cercle   Ligne    Z        U        8        S        Triangle Etoile   Escargot\n");
	for (int i = 0; i < layerOutput->size(); i++)
	{
		printf("%.2f     ", layerOutput->at(i).out);
	}
	printf("\n");
}