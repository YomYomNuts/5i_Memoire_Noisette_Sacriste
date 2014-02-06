#include "PMC.h"
#include <random>
#include <iostream>


PMC::PMC(vector<int> sizePMC)
{
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
			
			node.hx = 0;
			node.in = 0;
			node.loss = 0;
			for (int m = 0; m < sizePrev; ++m)
			{
				node.weight.push_back(0);
			}

			layer.push_back(node);
		}

		this->perceptron.push_back(layer);
	}
}

PMC::~PMC(void)
{
}

double PMC::RandomDouble(double Low, double High)
{
    double f = (double)rand() / RAND_MAX;
    return Low + f * (High - Low);
}

double PMC::FunctionSigmoid(double wx)
{
    return 1.0 / (1.0 + exp(-wx));
}

double PMC::FunctionThreshold(double wx)
{
    return wx < 0.5 ? 0 : 1;
}

bool PMC::LaunchLearning(int maxIteration, double learningRate, bool useSigmoid, vector<TrainingStruct> listTraining)
{
	bool continueLearning;
	int numberFramesTest = 0;

	InitializeWeight();

	// Learning
	do
	{
		continueLearning = false;
		++numberFramesTest;
		for (unsigned int i = 0; i < listTraining.size(); ++i)
		{
			TrainingStruct training = listTraining.at(i);
			vector<Node> * layerInput = &this->perceptron.at(0);
			vector<Node> * layerOutput = &this->perceptron.at(this->perceptron.size() - 1);

			/* Propagate the inputs */
			for (unsigned int k = 0; k < layerInput->size(); ++k)
			{
				Node * node = &layerInput->at(k);
				node->hx = training.data.at(k);
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
						sum += node->weight.at(l) * prevLayer->at(l).hx;
					node->in = sum;
					if (useSigmoid)
						node->hx = FunctionSigmoid(node->in);
					else
						node->hx = FunctionThreshold(node->in);
				}
			}

			/* Propagate the loss */
			for (unsigned int k = 0; k < layerOutput->size(); ++k)
			{
				Node * node = &layerOutput->at(k);
				if (useSigmoid)
					node->loss = node->hx * (1 - node->hx) * (training.result.at(k) - node->in);
				else
					node->loss = (-node->hx) * (1 - node->hx) * (training.result.at(k) - node->in);
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
					if (useSigmoid)
						node->loss = node->hx * (1 - node->hx) * sum;
					else
						node->loss = (-node->hx) * (1 - node->hx) * sum;
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
						node->weight.at(l) -= learningRate * node->in * prevLayer->at(l).loss;
				}
			}

            /* Verify if it's finish */
			for (unsigned int k = 0; k < layerOutput->size(); ++k)
			{
				Node * node = &layerOutput->at(k);
                if (node->loss != 0)
                    continueLearning = true;
			}
		}
	} while(continueLearning && numberFramesTest < maxIteration);


    for(unsigned int j = 0; j < this->perceptron.size() - 1; ++j)
    {
        vector<Node> * layer = &this->perceptron.at(j);
        vector<Node> * nextLayer = &this->perceptron.at(j+1);
        cout << "Layer " << j << " nb Nodes " << layer->size() << endl;
        for(unsigned int k = 0; k < layer->size(); ++k)
        {
            cout << "  Weights Node " << k << endl;
			double sum = 0;
            for (unsigned int l = 0; l < nextLayer->size(); ++l)
            {
                Node * node = &nextLayer->at(l);
				sum += node->loss;
                cout << "    Weight " << l << " " << node->weight.at(k) << endl;
            }
			cout << "    Loss " << sum << endl;
        }
    }

    cout << "Finish Learning " << !continueLearning << endl;
    cout << " --> With " << numberFramesTest << " iterations" << endl;

	return continueLearning;
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
			
			node->hx = 0;
			node->in = 0;
			node->loss = 0;
			for (int m = 0; m < sizePrev; ++m)
			{
				node->weight.at(m) = RandomDouble(-0.5, 0.5);
			}
		}
	}
}