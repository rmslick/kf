#pragma once
#include <armadillo>
using Mat = arma::mat;
enum AdaptiveRFiltering
{
    CONSTANT,
    INNOVATION,
    RESIDUAL
};

class KalmanFilter
{
    protected:

    public:
        virtual void Predict() = 0;
        virtual void Update(Mat z) = 0;
        virtual Mat GetState(Mat z) = 0;
        virtual void operator ()(){};
};