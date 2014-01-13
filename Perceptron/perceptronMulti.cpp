#include <iostream>
#include <math.h>
#include <stdlib.h>

#define USE_MULTI
#ifdef USE_MULTI

using namespace std;

#define NUMBER_PARAMETERS    2
#define NUMBER_SIZE_VECTOR	(NUMBER_PARAMETERS+1) // number parameters and the through
#define NUMBER_INPUTS		4
#define NUMBER_OUTPUTS		1
#define LEARNING_RATE		0.1
#define NUMBER_LAYERS		3 // 1 input, 1 output and 1 hide
#define NUMBER_NODES		{ NUMBER_SIZE_VECTOR, NUMBER_PARAMETERS, NUMBER_OUTPUTS }
#define MAX_FRAMES_TEST	    10000
#define USE_SIGMOID

// Types
typedef struct {
    int data[NUMBER_SIZE_VECTOR];
    int result[NUMBER_OUTPUTS];
} TrainingStruct;

typedef struct {
    double * weight;
    double hx;
	double in;
	double loss;
} Node;

typedef struct {
    Node * nodes;
    int numberNodes;
} Layer;

typedef struct {
    Layer * layers;
    int numberLayers;
} Perceptron;


// Internal functions
void initializePerceptron(Perceptron * perceptron);
void initializeWeight(Perceptron * perceptron);



double RandomDouble(double Low, double High)
{
  return ((double) rand() / RAND_MAX) * (High-Low) + Low;
}

double FunctionSigmoid(double wx)
{
    return 1.0 / (1.0 + exp(-wx));
}

double FunctionThreshold(double wx)
{
    return wx < 0.5 ? 0 : 1;
}

int main (int argc, char *argv[])
{
	Perceptron * perceptron = new Perceptron();
    TrainingStruct trainingData[NUMBER_INPUTS] =
    {
        {
            {0, 0, 1},
            {0}
        },
        {
            {0, 1, 1},
            {1}
        },
        {
            {1, 0, 1},
            {1}
        },
        {
            {1, 1, 1},
            {0}
        }
    };
	bool continueLearning;
	int numberFramesTest = 0;

	initializePerceptron(perceptron);
	initializeWeight(perceptron);

	// Learning
	do
	{
		continueLearning = false;
		++numberFramesTest;
		for (int i = 0; i < NUMBER_INPUTS; ++i)
		{
			TrainingStruct * training = &trainingData[i];
			Layer * layerInput = &perceptron->layers[0];
			Layer * layerOutput = &perceptron->layers[perceptron->numberLayers - 1];

			/* Propagate the inputs */
			for (int k = 0; k < layerInput->numberNodes; ++k)
			{
				Node * node = &layerInput->nodes[k];
				node->hx = training->data[k];
			}
			for(int j = 1; j < perceptron->numberLayers; ++j)
			{
				Layer * layer = &perceptron->layers[j];
				Layer * prevLayer = &perceptron->layers[j-1];
				for(int k = 0; k < layer->numberNodes; ++k)
				{
					Node * node = &layer->nodes[k];
					double sum = 0;
					for (int l = 0; l < prevLayer->numberNodes; ++l)
						sum += node->weight[l] * prevLayer->nodes[l].hx;
					node->in = sum;
#ifdef USE_SIGMOID
					node->hx = FunctionSigmoid(node->in);
#else
                    node->hx = FunctionThreshold(node->in);
#endif
				}
			}

			/* Propagate the loss */
			for (int k = 0; k < layerOutput->numberNodes; ++k)
			{
				Node * node = &layerOutput->nodes[k];
                node->loss = node->hx * (1 - node->hx) * (training->result[k] - node->in);
			}
			for(int j = perceptron->numberLayers - 2; j >= 0; --j)
			{
				Layer * layer = &perceptron->layers[j];
				Layer * nextLayer = &perceptron->layers[j+1];
				for(int k = 0; k < layer->numberNodes; ++k)
				{
					Node * node = &layer->nodes[k];
					double sum = 0;
					for (int l = 0; l < nextLayer->numberNodes; ++l)
                    {
                        Node * nodeNextLayer = &nextLayer->nodes[l];
						sum += nodeNextLayer->weight[k] * nodeNextLayer->loss;
                    }
#ifdef USE_SIGMOID
                    node->loss = node->hx * (1 - node->hx) * sum;
#else
                    node->loss = node->hx * sum;
#endif
				}
			}

			/* Update weight */
			for(int j = 1; j < perceptron->numberLayers; ++j)
			{
				Layer * layer = &perceptron->layers[j];
				Layer * prevLayer = &perceptron->layers[j-1];
				for(int k = 0; k < layer->numberNodes; ++k)
				{
					Node * node = &layer->nodes[k];
					for (int l = 0; l < prevLayer->numberNodes; ++l)
						node->weight[l] -= LEARNING_RATE * node->in * prevLayer->nodes[l].loss;
				}
			}

            /* Verify if it's finish */
			for (int k = 0; k < layerOutput->numberNodes; ++k)
			{
				Node * node = &layerOutput->nodes[k];
                if (node->loss != 0)
                    continueLearning = true;
			}
		}
	} while(continueLearning && numberFramesTest < MAX_FRAMES_TEST);


    for(int j = 0; j < perceptron->numberLayers - 1; ++j)
    {
        Layer * layer = &perceptron->layers[j];
        Layer * nextLayer = &perceptron->layers[j+1];
        cout << "Layer " << j << " nb Nodes " << layer->numberNodes << endl;
        for(int k = 0; k < layer->numberNodes; ++k)
        {
            cout << "  Weights Node " << k << endl;
            for (int l = 0; l < nextLayer->numberNodes; ++l)
            {
                Node * node = &nextLayer->nodes[l];
                cout << "    Weight " << l << " " << node->weight[k] << endl;
            }
        }
    }

    cout << "Finish Learning " << !continueLearning << endl;
    cout << " --> With " << numberFramesTest << " iterations" << endl;

	return 0;
}

void initializePerceptron(Perceptron * perceptron)
{
	const int numberNodes[NUMBER_LAYERS] = NUMBER_NODES;

	// Creation of the structure
	perceptron->numberLayers = NUMBER_LAYERS;
	perceptron->layers = new Layer[perceptron->numberLayers];
	for(int j = 0; j < perceptron->numberLayers; ++j)
	{
		Layer * layer = &perceptron->layers[j];
		layer->numberNodes = numberNodes[j];
		layer->nodes = new Node[layer->numberNodes];
		for(int k = 0; k < layer->numberNodes; ++k)
		{
			Node * node = &layer->nodes[k];
			if(j > 0)
			{
				node->weight = new double[numberNodes[j - 1]];
			}
			else
			{
				node->weight = NULL;
			}
		}
	}
}

void initializeWeight(Perceptron * perceptron)
{
	for(int j = 1; j < perceptron->numberLayers; ++j)
	{
		Layer * layer = &perceptron->layers[j];
		Layer * prevLayer = &perceptron->layers[j-1];
		for(int k = 0; k < layer->numberNodes; ++k)
		{
			Node * node = &layer->nodes[k];
			for(int l = 0; l < prevLayer->numberNodes; ++l)
			{
				node->weight[l] = RandomDouble(-0.5, 0.5);
			}
		}
	}
}

#endif
