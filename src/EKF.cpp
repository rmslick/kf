#include "EKF.h"
dual A1(dual x, dual y,dual x_v, dual y_v)
{
    return x+y;
}
dual A2(dual x, dual y,dual x_v, dual y_v)
{
    return y;
}

dual A3(dual x, dual y,dual x_v, dual y_v)
{
    return x_v +y_v;
}
dual A4(dual x, dual y,dual x_v, dual y_v)
{
    return y_v;
}

dual H1(dual x, dual y,dual x_v, dual y_v)
{
    return x;
}
dual H2(dual x, dual y,dual x_v, dual y_v)
{
    return x_v;
}
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
EKF::EKF(Mat A, Mat Q, Mat B, Mat R, Mat x, Mat P,VectorXreal(*_processModel)(VectorXreal),VectorXreal(*_observationModel)(VectorXreal))
{
    _A = A;
    _Q = Q;
    _B = B;
    //_H = H;
    _R = R;
    _x = x;
    _P = P;
    SetProcessModel(_processModel);
    SetObservationModel(_observationModel);
}
/*
Mat EKF::WhiteNoise(Mat mean, Mat covariance)
{
    return arma::mvnrnd(mean,covariance);
}*/
void EKF::Predict()
{
    //Need to add control vector
    Mat _x_k = _processModel->f(_x); //+_B*u;
    Mat J_x = _processModel->ComputeJacobian(_x);
    Mat J_x_T = J_x.transpose();
    _P =  J_x *_P * J_x_T + _Q;
    _x = _x_k;


}
void EKF::Predict(Mat u)
{
    /*
    auto _x_k = _processModel->f(_x) +_B*u;
    _P = _processModel->ComputeJacobian(_x) *_P* _processModel->ComputeJacobian(_x).transpose() + _Q;
    _x = _x_k;
    */
}

void EKF::Update(Mat z)
{

    
    Eigen::MatrixXd h_k = _observationModel->f(_x);
    Eigen::MatrixXd J_h = _observationModel->ComputeJacobian(_x);
    Eigen::MatrixXd J_h_T = J_h.transpose();
    // Compute Gain
    Eigen::MatrixXd S = J_h*_P*J_h_T+_R;
    Eigen::MatrixXd K = _P*J_h_T*( S ).inverse();
    // compute state
     
    //Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<double, autodiff::detail::Real<1, double> >, const Eigen::Matrix<double, -1, -1>, const Eigen::Product<Eigen::Product<Eigen::Product<Eigen::Matrix<double, -1, -1>, Eigen::Transpose<Eigen::Matrix<double, -1, -1> >, 0>, Eigen::Inverse<Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<double, double>, const Eigen::Product<Eigen::Product<Eigen::Matrix<double, -1, -1>, Eigen::Matrix<double, -1, -1>, 0>, Eigen::Transpose<Eigen::Matrix<double, -1, -1> >, 0>, const Eigen::Matrix<double, -1, -1> > >, 0>, Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<double, autodiff::detail::Real<1, double> >, const Eigen::Matrix<double, -1, -1>, const Eigen::Matrix<autodiff::detail::Real<1, double>, -1, 1, 0, -1, 1> >, 0> > wert = _x+K*(z-h(t));
    Eigen::VectorXd innovation = z - h_k;
    // Update state and covariance
    _x =_x + K*innovation;
    //_P = (_P - _P* K*J_h);
    _P = _P - K* S *K.transpose();

}