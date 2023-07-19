#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include "daugman.h"
#include "metrics.h"
#include "detector.h"

using namespace std;
using namespace cv;

//declare path to classifier
const std::string faceClassifier = "../haarcascade_frontalface_default.xml";
const std::string eyeClassifier = "../haarcascade_eye_tree_eyeglasses.xml"; 
const std::string imgPath ="../eyes_direction_cv/";

//classes of interest
const vector<std::string> classes={"center","left","right"};

int main(int argc, const char** argv)
{
    Mat frame;
    std::string actualClass,predictedClass;

    //try to load classifiers
    if(dt::loadClassifiers(faceClassifier,eyeClassifier))
    {
        //paths of images
        vector<cv::String> paths;
        vector<cv::String> paths_tmp;
        vector<std::string> center,left,right;
        
        //find all image's paths
        for(std::string pos : classes)
        {
           glob(imgPath+pos+"/*.*", paths_tmp, false);
           paths.insert( paths.end(), paths_tmp.begin(), paths_tmp.end() );
        }
        
        //for each path
        for(int i=0;i<paths.size();i++)
        {
            //get the image
            frame=imread(paths[i]);
            // Apply the face detection with the haar cascade classifier
            predictedClass=dt::GazeEstimation(frame);
           
            //insert the prediction into the array of original class
            for(int j=0;j<classes.size();j++)
                if (paths[i].find(classes[j]) != string::npos) 
                {
                    switch(j)
                    {
                        case 0:
                            center.push_back(predictedClass);
                            break;
                            
                        case 1:
                            left.push_back(predictedClass);
                            break;
                            
                        case 2:
                            right.push_back(predictedClass);
                            break;
                    }
                }
               
                 
           
           if (waitKey(3000) == 'q')
           {
               cout<<""; // next img if q is pressed
           }
           
        }
        //calculate conf matrix
        ms::ConfusionMatrix(center,left,right);
        //calculate accuracy
        float acc=ms::CalAccuracy(center,left,right);
        //print Confusion Matrix
        ms::printConfusionMatrix(center,left,right);
        //print accuracy
        cout<<"ACCURACY = "<<acc<<endl;
    }else 
        return -1;
    return 0;
}

