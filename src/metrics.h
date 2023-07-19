#ifndef metrics
#define metrics

#include<iostream>
#include<vector>

namespace ms {

	/**	Calculate the confusion Matrix of the algorithm 
	* 
	* @param center - vector of string containing the predicted value of the image with real class center
	* @param left - vector of string containing the predicted value of the image with real class left
	* @param right - vector of string containing the predicted value of the image with real class right
	* 
	*/
	void ConfusionMatrix(std::vector<std::string> center, std::vector<std::string> left, std::vector<std::string> right);
	
	/**	Print the confusion Matrix of the algorithm 
	* 
	* @param center - vector of string containing the predicted value of the image with real class center
	* @param left - vector of string containing the predicted value of the image with real class left
	* @param right - vector of string containing the predicted value of the image with real class right
	* 
	*/
	void printConfusionMatrix(std::vector<std::string> center, std::vector<std::string> left, std::vector<std::string> right);
	
		/**	Calculate the accuracy of the algorithm
	* 
	* @param center - vector of string containing the predicted value of the image with real class center
	* @param left - vector of string containing the predicted value of the image with real class left
	* @param right - vector of string containing the predicted value of the image with real class right
	* @return- float containing the value of the accuracy parameter
	* 
	*/
	float CalAccuracy(std::vector<std::string> center, std::vector<std::string> left, std::vector<std::string> right);

}


#endif
