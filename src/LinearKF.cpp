#include "LinearKF.h"

LinearKF::LinearKF(Mat A, Mat Q, Mat H, Mat R, Mat x, Mat P)
{
    _A = A;
    _Q = Q;
    _H = H;
    _R = R;
    _x = x;
    _P = P;
}
LinearKF::LinearKF(Mat A, Mat Q, Mat B, Mat H, Mat R, Mat x, Mat P)

{
    _A = A;
    _Q = Q;
    _B = B;
    _H = H;
    _R = R;
    _x = x;
    _P = P;
}
/*
Mat LinearKF::WhiteNoise(Mat mean, Mat covariance)
{
    return arma::mvnrnd(mean,covariance);
}*/
void LinearKF::Predict()
{
    //Need to add control vector
    _x = _A * _x; //+_B*u;
    //std::cout  << "\n\n _A*_P \n\n" << _A*_P * _A.transpose() +_Q << std::endl;

    _P = _A*_P* _A.transpose() + _Q;
    //std::cout << _P << std::endl;
}
void LinearKF::Predict(Mat u)
{
    //Need to add control vector
    _x = _A * _x +_B*u;
    _P = _A*_P* _A.transpose() + _Q;
    
}

void LinearKF::Update(Mat z)
{
    // Computer innovation
    Mat y = z - _H*_x;
    // Innovation covariance
    Mat S = _H*_P*_H.transpose() + _R;
    // Kalman gain
    Mat K = _P*_H.transpose() *S.inverse();
    // State update
    _x = _x + K*y;
    // Covariance update
    _P = _P - _P*K*_H;

    
}