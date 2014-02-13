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
	/*
	int layers3[] = {10,100,50,1};
	MultiLayerPerceptron mlp3(4,layers3);
	mlp3.Run("./../OpenGL/Memoire/Memoire/inputs20_essais_10vecteurs.txt",2000);
	mlp3.Evaluate("./../OpenGL/Memoire/Memoire/inputs.txt");
	*/
	/*
	int layers4[] = {20,40,30,1};
	MultiLayerPerceptron mlp4(4,layers4);
	mlp4.Run("./../OpenGL/Memoire/Memoire/inputs20_essais_20vecteurs.txt",5000);
	*/
	/*
	int layers5[] = {50,75,50,1};
	MultiLayerPerceptron mlp5(4,layers5);
	mlp5.Run("./../OpenGL/Memoire/Memoire/inputs20_essais_50angles.txt",3000);
	mlp5.Evaluate("./../OpenGL/Memoire/Memoire/inputs.txt");
	*/
	
	int layers6[] = {50,75,10};
	MultiLayerPerceptron mlp6(3,layers6);
	mlp6.Run("./../OpenGL/Memoire/Memoire/inputs20_essais_50angles.txt",3000);
	mlp6.Evaluate("./../OpenGL/Memoire/Memoire/inputs.txt");
	

	char test;
	std::cin >> test;

	return 0;
}

