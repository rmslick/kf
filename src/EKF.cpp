#include "EKF.h"

EKF::EKF(Mat A, Mat Q, Mat H, Mat R, Mat x, Mat P)
{
    _A = A;
    _Q = Q;
    _H = H;
    _R = R;
    _x = x;
    _P = P;
}
EKF::EKF(Mat A, Mat Q, Mat B, Mat H, Mat R, Mat x, Mat P)

{
    _A = A;
    _Q = Q;
    _B = B;
    _H = H;
    _R = R;
    _x = x;
    _P = P;
}
Mat EKF::WhiteNoise(Mat mean, Mat covariance)
{
    return arma::mvnrnd(mean,covariance);
}
void EKF::Predict()
{
    //Need to add control vector
    _x = _A * _x; //+_B*u;
    _P = _A*_P* trans(_A) + _Q;
}
void EKF::Predict(Mat u)
{
    //Need to add control vector
    _x = _A * _x +_B*u;
    _P = _A*_P* trans(_A) + _Q;
}

void EKF::Update(Mat z)
{
    // Computer innovation
    Mat y = z - _H*_x;
    // Innovation covariance
    Mat S = _H*_P*trans(_H) + _R;
    // Kalman gain
    Mat K = _P*trans(_H)*inv(S);
    // State update
    _x = _x + K*y;
    // Covariance update
    _P = _P - _P*K*_H;

}