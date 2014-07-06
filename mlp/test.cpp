/*********************************************************************
 * File  : test.cpp
 * Author: Sylvain BARTHELEMY
 *         mailto:sylvain@sylbarth.com
 *         http://www.sylbarth.com
 * Date  : 2000-08
 *********************************************************************/

#include "mlp.h"
#include <iostream>

int main(int argc, char* argv[])
{
	std::cout << "Traitement en cours..." << std::endl;

	/*
	int layers1[] = {2,2,1};
	MultiLayerPerceptron mlp1(3,layers1);
	mlp1.Run("xor.dat",40000);
	*/
	/*
	int layers2[] = {1,5,1};
	MultiLayerPerceptron mlp2(3,layers2);
	mlp2.Run("sin.dat",500);
	*/
	
	
	int layers3[] = {50,75,25,10};
	MultiLayerPerceptron mlp3(4,layers3);
	mlp3.Run("./../OpenGL/Memoire/Memoire/inputs10_essais_50angles.txt",5000);
	mlp3.Evaluate("./../OpenGL/Memoire/Memoire/inputsEscargot_50angles.txt");
	int layers4[] = {50,75,50,10};
	MultiLayerPerceptron mlp4(4,layers4);
	mlp4.Run("./../OpenGL/Memoire/Memoire/inputs10_essais_50angles.txt",5000);
	mlp4.Evaluate("./../OpenGL/Memoire/Memoire/inputsEscargot_50angles.txt");
	int layers5[] = {50,75,75,10};
	MultiLayerPerceptron mlp5(4,layers5);
	mlp5.Run("./../OpenGL/Memoire/Memoire/inputs10_essais_50angles.txt",5000);
	mlp5.Evaluate("./../OpenGL/Memoire/Memoire/inputsEscargot_50angles.txt");

	int layers6[] = {50,75,25,10};
	MultiLayerPerceptron mlp6(4,layers6);
	mlp6.Run("./../OpenGL/Memoire/Memoire/inputs20_essais_50angles.txt",5000);
	mlp6.Evaluate("./../OpenGL/Memoire/Memoire/inputsEscargot_50angles.txt");
	int layers7[] = {50,75,50,10};
	MultiLayerPerceptron mlp7(4,layers7);
	mlp7.Run("./../OpenGL/Memoire/Memoire/inputs20_essais_50angles.txt",5000);
	mlp7.Evaluate("./../OpenGL/Memoire/Memoire/inputsEscargot_50angles.txt");
	int layers8[] = {50,75,75,10};
	MultiLayerPerceptron mlp8(4,layers8);
	mlp8.Run("./../OpenGL/Memoire/Memoire/inputs20_essais_50angles.txt",5000);
	mlp8.Evaluate("./../OpenGL/Memoire/Memoire/inputsEscargot_50angles.txt");
	
	/*
	int layers3[] = {50,25,10};
	MultiLayerPerceptron mlp3(3,layers3);
	mlp3.Run("./../OpenGL/Memoire/Memoire/inputs10_essais_50angles.txt",5000);
	mlp3.Evaluate("./../OpenGL/Memoire/Memoire/inputsEscargot_50angles.txt");
	int layers4[] = {50,50,10};
	MultiLayerPerceptron mlp4(3,layers4);
	mlp4.Run("./../OpenGL/Memoire/Memoire/inputs10_essais_50angles.txt",5000);
	mlp4.Evaluate("./../OpenGL/Memoire/Memoire/inputsEscargot_50angles.txt");
	int layers5[] = {50,75,10};
	MultiLayerPerceptron mlp5(3,layers5);
	mlp5.Run("./../OpenGL/Memoire/Memoire/inputs10_essais_50angles.txt",5000);
	mlp5.Evaluate("./../OpenGL/Memoire/Memoire/inputsEscargot_50angles.txt");

	int layers6[] = {50,25,10};
	MultiLayerPerceptron mlp6(3,layers6);
	mlp6.Run("./../OpenGL/Memoire/Memoire/inputs20_essais_50angles.txt",5000);
	mlp6.Evaluate("./../OpenGL/Memoire/Memoire/inputsEscargot_50angles.txt");
	int layers7[] = {50,50,10};
	MultiLayerPerceptron mlp7(3,layers7);
	mlp7.Run("./../OpenGL/Memoire/Memoire/inputs20_essais_50angles.txt",5000);
	mlp7.Evaluate("./../OpenGL/Memoire/Memoire/inputsEscargot_50angles.txt");
	int layers8[] = {50,75,10};
	MultiLayerPerceptron mlp8(3,layers8);
	mlp8.Run("./../OpenGL/Memoire/Memoire/inputs20_essais_50angles.txt",5000);
	mlp8.Evaluate("./../OpenGL/Memoire/Memoire/inputsEscargot_50angles.txt");
	*/

	char test;
	std::cin >> test;

	return 0;
}

