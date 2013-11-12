#include <iostream>

using namespace std;

#define NUMBER_SIZE_VECTOR	3 // 2 parameters of vector and the through
#define NUMBER_ENTRIES		4
#define LEARNING_RATE		0.1

typedef struct {
    int data[NUMBER_SIZE_VECTOR];
    int result;
} TrainingStruct;

bool threshold(double value)
{
	return value > 0.5 ? 1 : 0;
}

int main (int argc, char *argv[])
{
    TrainingStruct trainingData[NUMBER_ENTRIES] =
    {
        {
            {0, 0, 1},
            0
        },
        {
            {0, 1, 1},
            0
        },
        {
            {1, 0, 1},
            0
        },
        {
            {1, 1, 1},
            1
        }
    };
	double weight[NUMBER_SIZE_VECTOR] = { 0, 0, 0 };
	bool continueLearning, Hw;
	double sumWX;

	do
	{
	    continueLearning = false;
		for (int i = 0; i < NUMBER_ENTRIES; ++i)
		{
		    TrainingStruct * training = &trainingData[i];

			// Calculate w * x
			sumWX = 0;
			for(int j = 0; j < NUMBER_SIZE_VECTOR; ++j)
				sumWX += training->data[j] * weight[j];

			// Apply the function threshold
			Hw = threshold(sumWX);

			// Apply the modification of the weight
			for (int j = 0; j < NUMBER_SIZE_VECTOR; ++j)
				weight[j] = weight[j] + LEARNING_RATE * (training->result - Hw) * training->data[j];

			// Verify the end of the learning
			continueLearning |= ((training->result - Hw) != 0);
		}
	} while (continueLearning);

	// Display the weight
	for (int i = 0; i < NUMBER_SIZE_VECTOR; ++i)
		cout << "weight[" << i << "] = " << weight[i] << endl;

	return 0;
}
