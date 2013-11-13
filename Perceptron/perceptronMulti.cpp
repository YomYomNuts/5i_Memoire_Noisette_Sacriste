#include <iostream>
#include <math.h>
#include <stdlib.h>

using namespace std;

#define NUMBER_SIZE_VECTOR	3 // 2 parameters of vector and the through
#define NUMBER_INPUTS		4
#define NUMBER_OUTPUTS		1
#define LEARNING_RATE		0.1
#define NUMBER_LAYERS		3
#define NUMBER_NODES		{ NUMBER_SIZE_VECTOR, 2, NUMBER_OUTPUTS }
#define MAX_FRAMES_TEST		500

// Types
typedef struct {
    int data[NUMBER_SIZE_VECTOR];
    int result[NUMBER_OUTPUTS];
} TrainingStruct;

typedef struct {
    double * weight;
    double a;
	double in;
	double delta;
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
				node->a = training->data[k];
			}
			for(int j = 1; j < perceptron->numberLayers; ++j)
			{
				Layer * layer = &perceptron->layers[j];
				Layer * prevLayer = &perceptron->layers[j-1];
				for(int k = 0; k < layer->numberNodes; ++k)
				{
					Node * node = &layer->nodes[k];
					double sum = 0;
					for (int l = 0; l < prevLayer->numberNodes; ++l )
						sum += node->weight[l] * prevLayer->nodes[l].a;
					node->in = sum;
					node->a = 1.0 / (1.0 + exp(-node->in));
				}
			}

			/* Propagate the delta */
			for (int k = 0; k < layerOutput->numberNodes; ++k)
			{
				Node * node = &layerOutput->nodes[k];
				node->delta = training->result[k] - node->a;
			}
			for(int j = perceptron->numberLayers - 2; j >= 0; --j)
			{
				Layer * layer = &perceptron->layers[j];
				Layer * nextLayer = &perceptron->layers[j+1];
				for(int k = 0; k < layer->numberNodes; ++k)
				{
					Node * node = &layer->nodes[k];
					double sum ;
					for (int l = 0; l < nextLayer->numberNodes; ++l)
						sum += nextLayer->nodes[l].weight[k] * nextLayer->nodes[l].delta;
					node->delta = node->in * (1 - node->in) * sum;
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
					for (int l = 0; l < prevLayer->numberNodes; ++l )
						node->weight[l] += LEARNING_RATE * node->a * prevLayer->nodes[l].delta;
				}
			}
		}
	} while(continueLearning && numberFramesTest < MAX_FRAMES_TEST);

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
				node->weight = new double[ numberNodes[j - 1] ];
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
