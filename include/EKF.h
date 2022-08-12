#pragma once

#include <armadillo>

using Mat = arma::mat;

class EKF 
{
    private:
        Mat Jacobian(); 
};