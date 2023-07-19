#ifndef detector
#define detector

#include<iostream>
#include<vector>

namespace dt {

	/**	Comparing the area of rect a and b
	* 
	* @param a- Rect a
	* @param b - Rect b
	* @return- bool if area of a is greater than area of b
	* 
	*/
	bool compareByArea(cv::Rect &a, cv::Rect  &b);
	
	/**	Get the string containing the value of where a person is watching 
	* 
	* @param Frame - image to test
	* @return- string containing the value of the prediction
	* 
	*/
	std::string GazeEstimation(cv::Mat frame);
	
		/**	Calculate the caccuracy of the algorithm
	* 
	* @param faceClassifier - path of the face classifier
	* @param eyeClassifier- path of the eye classifier
	* @return- true if classifier are loaded right
	* 
	*/
	bool loadClassifiers(cv::String faceClassifier, cv::String eyeClassifier);
	
		/**	Find the face from a image
	* 
	* @param frame- the image where we want to extract the face
	* @return- rect containing the position of the face
	* 
	*/
	cv::Rect faceDetect(cv::Mat frame);

    		/**	Find the eyes on a image of a face
	* 
	* @param faceROI- the full image where we want to extract the eyes
	* @param face- the rect containing the position of the face
	* @return- vector containing the rect where the eyes are
	* 
	*/
    std::vector<cv::Rect> eyeDetect(cv::Mat frame,cv::Rect face);

}


#endif
