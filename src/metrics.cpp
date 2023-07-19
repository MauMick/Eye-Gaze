#include <iostream>
#include<vector>
#include "metrics.h"
#include <algorithm>
#include <stdio.h>
#include <string.h>


using namespace std;

//variables containing the classes
const vector<std::string> classes={"center","left","right"};
const vector<string> initial={"C","L","R"};
//variable for accuracy and matrix
float accuracy=0;
int matrix[3][3];

void ms::ConfusionMatrix(vector<std::string> center, vector<std::string> left, vector<std::string> right)
{
    //setting all matrix to 0
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            matrix[i][j]=0;
            
    //calculate first column, aka the center column
    for(std::string x : center)
    {
        for(int j=0;j<classes.size();j++)
        {
            if(classes[j].compare(x)==0)
            {
                matrix[0][j]++;
            }
        }
    }
    
    //calculate second column, aka the left column
    for(std::string x : left)
    {
        for(int j=0;j<classes.size();j++)
        {
            if(classes[j].compare(x)==0)
            {
                matrix[1][j]++;
            }
        }
    }
    
    //calculate third column, aka the right column
    for(std::string x : right)
    {
        for(int j=0;j<classes.size();j++)
        {
            if(classes[j].compare(x)==0)
            {
                matrix[2][j]++;
            }
        }
    }
    

}


//function to print
void ms::printConfusionMatrix(std::vector<std::string> center, std::vector<std::string> left, std::vector<std::string> right)
{
    //if the matrix is not created we create it
    if(!matrix)
            ms::ConfusionMatrix(center, left, right);
    //Printing confusion matrix
    cout<<endl;
    //predicted values on the columns
    cout<<"        PREDICTED"<<endl<<"        \\ ";
    
    //print the name of the columns
    for(int i=0;i<3;i++) cout<<initial[i]<<" ";
    cout<<endl;
    
    for(int i=0;i<3;i++)
    {
        //actual values on the rows
        if(i==1)
            cout<<"ACTUAL  ";
        else
            cout<<"        ";
        //print the name of the rows
        cout<<initial[i]<<" ";
        
        //print the values
        for(int j=0;j<3;j++)
            cout<<matrix[i][j]<<" ";
        cout<<endl;
    }
        
}

//function to calculate the accuracy
float ms::CalAccuracy(std::vector<std::string> center, std::vector<std::string> left, std::vector<std::string> right)
{
    //if the accuracy is not calculated we do it
    if(accuracy==0)
    {
        //if the matrix is not created we create it
        if(!matrix)
            ms::ConfusionMatrix(center, left, right);
        
        //find the total number of images
        int total=center.size()+left.size()+right.size();
        //calculate accuracy as sum of TP divided by the total 
        for(int i=0;i<3;i++)
            accuracy+=matrix[i][i];
        accuracy=accuracy/total;
    }
    //return accuracy
    return accuracy;
}


