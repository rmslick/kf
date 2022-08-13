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
        Mat _A, _Q, _B, _H, _R;
        // State and covariance
        Mat _x, _P;
        Mat _zeros;
    public:
        virtual void Predict() = 0;
        virtual void Update(Mat z) = 0;
        virtual Mat GetState(Mat z) = 0;
        virtual void operator ()(){};
};