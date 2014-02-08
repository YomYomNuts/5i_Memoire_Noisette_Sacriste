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
	/*
	int layers1[] = {2,2,1};
	MultiLayerPerceptron mlp1(3,layers1);
	mlp1.Run("xor.dat",10000);
	*/
	/*
	int layers2[] = {1,5,1};
	MultiLayerPerceptron mlp2(3,layers2);
	mlp2.Run("sin.dat",500);
	*/
	int layers3[] = {10,20,1};
	MultiLayerPerceptron mlp2(3,layers3);
	mlp2.Run("./../OpenGL/Memoire/Memoire/inputs10_essais_10vecteurs.txt",5000);
	
	char test;
	std::cin >> test;

	return 0;
}

