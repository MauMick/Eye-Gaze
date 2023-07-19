#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include "detector.h"
#include "daugman.h"

using namespace std;
using namespace cv;

//declare classifier
CascadeClassifier face_cascade;
CascadeClassifier eye_cascade;

//area comparison to sort
bool dt::compareByArea(Rect &a, Rect  &b)
{
    return a.area() > b.area();
}

bool dt::loadClassifiers(String faceClassifier, String eyeClassifier)
{
    // Load the pre trained haar cascade classifier

    if (!face_cascade.load(faceClassifier))
    {
        cout << "Could not load the classifier";
        return false;
    };
    if (!eye_cascade.load(eyeClassifier))
    {
        cout << "Could not load the classifier";
        return false;
    };
    return true;
}


//detect the Rect with the face
Rect dt::faceDetect(Mat frame)
{
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    vector<Rect> faces;
    
    //first try to detect
    face_cascade.detectMultiScale(frame_gray, faces,1.1,7,0,Size(100,100));

    //find biggest face
    int face=-1, face_area=0;
    for (int val = 0; val < faces.size(); val++)
    {
        if(faces[val].area()>face_area)
        {
            face=val;
            face_area=faces[val].area();
        }
    }
    //if the face is not found, try again
    if(faces.size()<1)
    {
        //clearing old 
        faces.clear();
        
        //equalize image 
        Mat equal_img;
        equalizeHist(frame_gray, equal_img);
        //detect with better accuracy ( more time consuming)
        face_cascade.detectMultiScale(equal_img, faces,1.0021,8,0,Size(100,100));

        //search best face aka the face with biggest area
        for (int val = 0; val < faces.size(); val++)
        {
            if(faces[val].area()>face_area)
            {
                face=val;
                face_area=faces[val].area();
            }
        }
    }
    //if face is found return it
    if(faces.size()>0)
        return faces[face];
    return Rect();
}

//function to detect eyes
vector<Rect> dt::eyeDetect(Mat frame,Rect face)
{
    //variable for eyes
    int first=-1,second=-1;
    int biggest=0;
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    
    Mat faceROI = frame_gray(face);
    
    vector<Rect> eyes;
    //detect eye with different params
    eye_cascade.detectMultiScale(faceROI, eyes,1.06,7);
    if(eyes.size()<2)
    {
        //stronger parameters
        eye_cascade.detectMultiScale(faceROI, eyes,1.03,6);
        if(eyes.size()<2)
        {
            //if not found try to equalize
            Mat faceROI_eq;
            equalizeHist(faceROI, faceROI_eq);
            eye_cascade.detectMultiScale(faceROI_eq , eyes,1.1,5);
            if(eyes.size()<2)
            {
                //if not found try to increment contrast and brightness
                Mat faceROI_cont;
                faceROI.convertTo(faceROI_cont, -1, 1.3, 100);
                
                eye_cascade.detectMultiScale(faceROI_cont, eyes,1.025,2);
                if(eyes.size()<2)
                {
                    //if not found, probably only 1 eye in the image so try normal eye detector
                    eye_cascade.detectMultiScale(faceROI, eyes,1.01,6);
                }
                
                 
            }
        }
    }
    //sort the vector of the eyes based on the area
    sort(eyes.begin(), eyes.end(), dt::compareByArea);
    vector<Rect> reyes;
    
    //select best 2 eyes if they exist
    if(eyes.size()>0)
    {
        Rect r1 = Rect(face.x + eyes[0].x, face.y + eyes[0].y, eyes[0].width, eyes[0].height);
        reyes.push_back(r1);
    }
    if(eyes.size()>1)
    {
        Rect r1 = Rect(face.x + eyes[1].x, face.y + eyes[1].y, eyes[1].width, eyes[1].height);
        reyes.push_back(r1);
    }
    //return the vector with the box of the eyes
    return reyes;
}

//function to estimate the direction 
std::string dt::GazeEstimation(Mat frame)
{
    //declare helping variables
    Mat frame_gray;
    Mat h_frame = frame.clone();
    string result;
    
    Rect face;
    vector<Rect> eyes;
    //converting to greyscale
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

    //find face box
    face=dt::faceDetect(frame);
    
    if(face!=Rect())
    {
        //detect eyes
        eyes=dt::eyeDetect(frame,face);
        //variable to threshold the direction
        float dist1=0,dist2=0;
        int vL=-10, vl=-1, vc=0, vr=1, vR=10, v=0;
        int thresholdl=15, thresholdr=22;
        //if there is only 1 eye, the thresholds are lowest
        if(eyes.size()==1)
        {
            thresholdr=15;
            thresholdl=15;
        }
        //for each eye
        Mat upscale;
        for (Rect r : eyes) {
            //find the iris and print it
            vector<int> coordVal = dm::printIris(h_frame, &h_frame, r, &upscale);
            //get x,y and radius of the iris
            int x=coordVal[0];
            int y=coordVal[1];
            int radius=coordVal[2];
            dist1= x-radius;
            dist2= upscale.cols-(x+radius);
            
            //check if an eye is more on left or right (bigger impact)
            if(dist1-dist2>thresholdl*2) v+=vL;
            else
            if(dist1-dist2<-thresholdr*2) v+=vR;
            else
            {
                //check if eye is a bit on left or right (smaller impact)
                if(dist1-dist2>=-thresholdr && dist1-dist2<=thresholdl) v+=0;
                if(dist1-dist2>thresholdl) v+=vl;
                if(dist1-dist2<-thresholdr) v+=vr;   
            }
        }
        
        //Printing eyes
        for(Rect e : eyes)
        {
            Point centereye(e.x + e.width / 2, e.y + e.height / 2);
            ellipse(h_frame, centereye, Size(e.width / 2, e.height / 2), 0, 0, 360, Scalar(0, 255, 0), 2);
        }
        //printing on image where the subject is watching and save the result
        if(v<=-1) 
        {
            putText(h_frame, " looking left", Point(5, 20), HersheyFonts::FONT_HERSHEY_PLAIN, 2, 255, 2);
            result="left";
        }
        else
        {
            if(v>=1) 
            {
                putText(h_frame, "looking right", Point(5, 20), HersheyFonts::FONT_HERSHEY_PLAIN, 2, 255, 2);
                result="right";
            }
            else
            {
                putText(h_frame, "looking straight", Point(5, 20), HersheyFonts::FONT_HERSHEY_PLAIN, 2, 255, 2);
                result="center";
            }
        }
    }
    
    //show img        
    imshow("Live Face Detection", h_frame);
    
    //return the result;
    return result;
}


