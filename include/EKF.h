#pragma once
#include <stdio.h>
#include <armadillo>
#include "KF.h"
#include "AutoDiffWrapper.h"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

dual A1(dual x, dual y,dual x_v, dual y_v);

dual A2(dual x, dual y,dual x_v, dual y_v);

dual A3(dual x, dual y,dual x_v, dual y_v);

dual A4(dual x, dual y,dual x_v, dual y_v);

template<typename T> 
class ProcessModel
{
    
    public:
        std::vector< T > F;
        virtual Mat f(Eigen::VectorXd x_dot) = 0;
        std::vector< T > GetVectorFunction()
        {
            return F;
        }
        virtual Mat ComputeJacobian(Eigen::VectorXd x_dot) = 0;
};
dual H1(dual x, dual y,dual x_v, dual y_v);

dual H2(dual x, dual y,dual x_v, dual y_v);

class processModelCV : public ProcessModel< dual (*)(dual , dual, dual , dual) >
{
    private:
        AutoDiffWrapper ad;
    public:
        processModelCV()
        {
            F.push_back(A1);
            F.push_back(A2);
            F.push_back(A3);
            F.push_back(A4);
        }
        Mat f(Eigen::VectorXd x_dot)
        {
            Mat y( (int) F.size(),1 );
            y << A1(x_dot(0), x_dot(1), x_dot(2),x_dot(3) ).val
            ,A2(x_dot(0), x_dot(1), x_dot(2),x_dot(3) ).val
            ,A3(x_dot(0), x_dot(1), x_dot(2),x_dot(3) ).val
            ,A4(x_dot(0), x_dot(1), x_dot(2),x_dot(3) ).val;
            return y;
        }
        // Can we factor this out with a template?
        /*
        std::vector<dual (*)(dual , dual, dual , dual)>  GetVectorFunction()
        {
            return F;
        }*/
        Mat ComputeJacobian(Eigen::VectorXd x_dot)
        {
            return ad.Jacobian ( GetVectorFunction() ,x_dot(0), x_dot(1), x_dot(2),x_dot(3)) ;
        }


};

class observationModelCV : public ProcessModel< dual (*)(dual , dual, dual , dual) >
{
    private:
        AutoDiffWrapper ad;
    public:
        observationModelCV()
        {
            F.push_back(H1);
            F.push_back(H2);
        }
        Mat f(Eigen::VectorXd x_dot)
        {
            Mat y( (int) F.size(),1 );
            y << H1(x_dot(0), x_dot(1), x_dot(2),x_dot(3) ).val
            ,H2(x_dot(0), x_dot(1), x_dot(2),x_dot(3) ).val;
            return y;
        }
        // Can we factor this out with a template?
        /*
        std::vector<dual (*)(dual , dual, dual , dual)>  GetVectorFunction()
        {
            return F;
        }*/
        Mat ComputeJacobian(Eigen::VectorXd x_dot)
        {
            return ad.Jacobian ( GetVectorFunction() ,x_dot(0), x_dot(1), x_dot(2),x_dot(3)) ;
        }


};

class EKF : public KalmanFilter
{
    private:

    protected:
        VectorXreal (*f)(VectorXreal);
        VectorXreal (*h)(VectorXreal);
        AutoDiffWrapper ad;
        ProcessModel<dual (*)(dual , dual, dual , dual)> * _processModel;
        ProcessModel<dual (*)(dual , dual, dual , dual)> * _observationModel;
    public:
    /*
        Parameters
            - A : State transition matrix
            - Q : Process error covariance
            - B : Control matrix
            - H : Observation matrix
            - R : Measurement error covariance
    */
        EKF(Mat A, Mat Q, Mat H, Mat R, Mat x, Mat P);
        EKF(Mat A, Mat Q, Mat B, Mat H, Mat R, Mat x, Mat P);
        EKF(Mat A, Mat Q, Mat B, Mat R, Mat x, Mat P, VectorXreal(*_processModel)(VectorXreal) ,VectorXreal(*_observationModel)(VectorXreal));

        EKF(Mat A, Mat Q, Mat B, Mat R, Mat x,Mat P, ProcessModel<dual (*)(dual , dual, dual , dual)> * pM, ProcessModel<dual (*)(dual , dual, dual , dual)> * oM )
        {
            _A = A;
            _Q = Q;
            _B = B;
            //_H = H;
            _R = R;
            _x = x;
            _processModel = pM;
            _observationModel = oM;
            _P = P;
        }
        // The single-variable function for which derivatives are needed
        // method must take in vector of functions, measuremnt, H
    /*
     * Gaussian white noise selection
     */
        //Mat WhiteNoise(Mat mean, Mat covariance);
    
    /*
        Predict - Process model to predict current state from state transition
                  vector, prior state and optional control vector
            Paramters
                - u : Control vector (optional)
            Output:
                - Private members _x and _P are updated with result of prediction
    */
        void Predict();
        // Overloaded for control vector
        void Predict(Mat u);
        // vector valued function
        template<typename function>
        Mat ComputeJacobian(function f, dual x, dual y)
        {
            return ad.Jacobian (f,x,y) ;
        }
        Mat F(Mat x)
        {
            return ad.VectorXRealToVectorxd( f(_x) );
        }
        Mat H(Mat x)
        {
            return ad.VectorXRealToVectorxd( h(_x) ) ;
        }
    /*
     *
     */
        void Update(Mat z);
    /*
     *  GetCovariance() - Access method
     */
        Mat GetCovariance()
        {
            return _P;
        }
        Mat GetState(Mat z)
        {
            // Predict state and covariance
            //std::cout << "[INFO] Predicting!\n";
            Predict();
            //std::cout << "[INFO] Predicted!\n";
            // Update state and covariance 
            Update(z);
            // Return the predicted state
            return _x;
        }
        Mat operator () (Mat z)
        {
            return GetState(z);
        }
        void SetProcessModel(VectorXreal(*_pM)(VectorXreal))
        {
            f = _pM;
        }
        void SetObservationModel(VectorXreal(*_oM)(VectorXreal))
        {
            h = _oM;
        }

};
