#pragma once
#include "KalmanFilter.h"
#include <armadillo>

using Mat = arma::mat;

class EKF : KalmanFilter
{
    private:
        Mat Jacobian(); 
};