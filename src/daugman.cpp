#include "daugman.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>

using namespace std;
using namespace cv;

vector<double> dm::daugman(Mat img, Point p, int start_r, int end_r, int step) {

    vector<double> values;                          
    Mat mask = Mat::zeros(img.size(), CV_8U);

    //find sum of pixel on the circumference for each radius value
    for (int r = start_r;
        //keep the circle inside the image
        r <= end_r && p.x + r < img.cols && p.x - r >= 0 && p.y + r < img.rows && p.y - r >= 0;
        r += step) {

        //sum values according to the mask
        circle(mask, p, r, 255, 1);
        Mat diff;
        bitwise_and(img, mask, diff);
        double val = cv::sum(diff)[0];

        //normalize the sum
        values.push_back(val / (2 * M_PI * r));

        //reset the mask
        mask.setTo(0);
    }
    //if not enough values retrun an empty vector
    if (values.size() < 2) return vector<double>{ 0, 0 };

    //gaussian blurred vector of differences
    vector<double> delta;
    for (int i = 1; i < values.size(); i++) delta.push_back(values[i - 1] - values[i]);
    GaussianBlur(delta, delta, Size(3, 5), 0);
    for (int i = 0; i < delta.size(); i++) delta[i] = abs(delta[i]);

    //find argmax and radius
    auto it = max_element(delta.begin(), delta.end());
    int index = it - delta.begin();
    double radius = start_r + index * step;

    //find average of pixel values inside detected iris
    Mat a_diff, a_mask = Mat::zeros(img.size(), CV_8U);
    circle(a_mask, p, radius, 255, FILLED);
    //Mat kernel = getGaussianKernel(2*radius, 0, CV_32F);
    bitwise_and(img, a_mask, a_diff);
    //a_diff.mul(kernel);
    double sum = cv::sum(a_diff)[0];
    //parameter inversely proportional to pixel intensities to weight delta
    double alpha = 1 - sum / (pow(radius, 2) * M_PI * 255);

    // the following code calculated angular variance of the iris detected

    vector<double> data;
    //angle degree increment
    int deg = 30;
    //find total iris' sector area 
    double area = pow(radius, 2) * M_PI * deg / 360;
    //for each bin of 30°
    for (int i = deg; i <= 360; i += deg) {
        //calculate mask
        ellipse(mask, p, Size(radius, radius), i, 0, deg, 255, FILLED, LINE_AA);
        bitwise_and(img, mask, mask);
        //save value of the area mask
        double val = cv::sum(mask)[0];
        data.push_back(val / area);
        mask.setTo(0);
    }
    //variables
    double mean = 0;
    double var = 0;
    //calculate mean
    for (double d : data) mean += d;
    mean = mean / data.size();
    //calculate variance
    for (double d : data) var += pow((mean - d),2);
    var = var / data.size();

    double beta = min(1.0, 1 / log10(var));
    return vector<double>{ delta[index] * pow(alpha, 2) * pow(beta,1/3), radius };

}


vector<int> dm::findIris(Mat img, int d_start, int d_end, int d_step, int p_step) {

    if (img.cols != img.rows) cout << "Image should be a square for better performance";

    //create a grid of points coordinate 
    vector<Point> points;
    for (int i = img.cols * 1 / 5; i <= img.cols * 4 / 5; i += p_step) {
        for (int j = img.rows * 1 / 3; j <= img.rows * 2 / 3; j += p_step) {
            points.push_back(Point(i, j));
        }
    }

    //apply daugman to each point and find best result (max delta)
    vector<int> coordVal;
    int max = 0;
    int max_index = 0;
    for (int i = 0; i < points.size(); i++) {
        vector<double> val = dm::daugman(img, points[i], d_start, d_end, d_step);
        if (val[0] > max) {
            max = val[0];
            max_index = i;

            //return coordinates of center and radius
            coordVal = vector<int>{ points[i].x, points[i].y, (int)val[1] };
        }
    }
    return coordVal;
}


vector<int> dm::printIris(Mat src, Mat* dst, Rect r, Mat* upscaled) {

    //extract ROI of the eye
    Mat eye = src(r);
    //pre-initialize destination image
    *dst = src.clone();
    Mat eye_gray, eye_up, eye_gauss;
    //convert to grayscale
    cvtColor(eye, eye_gray, COLOR_BGR2GRAY);
    //resize image to obtained omogeneous results
    double scale = eye_gray.rows / 150.f;
    resize(eye_gray, eye_up, Size(150, 150), 0, 0, INTER_CUBIC);
    //equalize and filter the image
    equalizeHist(eye_up, eye_up);
    GaussianBlur(eye_up, eye_up, Size(5, 5), 0);
    //save image before apply daugman
    *upscaled=eye_up.clone();
    //apply daugman
    vector<int> coordVal = dm::findIris(eye_up, 20, 70, 3, 5);
    //find iris center on the destination image 
    Point center(coordVal[0] * scale + r.x, coordVal[1] * scale + r.y);
    int radius = coordVal[2] * scale;
    //print iris center and radius
    circle(*dst, center, 1, Scalar(100, 100, 100), 3, LINE_AA);
    circle(*dst, center, radius, Scalar(255, 255, 0), 1, LINE_AA);

    return coordVal;
}


