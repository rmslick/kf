#pragma once
#include <stdio.h>
#include <armadillo>
#include "KF.h"
#include "AutoDiffWrapper.h"
using Mat = Eigen::MatrixXd;




class EKF : public KalmanFilter
{
    private:
    protected:
        VectorXreal (*processModel)(VectorXreal);
        AutoDiffWrapper ad;
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
        // The single-variable function for which derivatives are needed
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
            Predict();
            // Update state and covariance 
            Update(z);
            // Return the predicted state
            return _x;
        }
        Mat operator () (Mat z)
        {
            return GetState(z);
        }
        void SetProcessModel(VectorXreal(*f)(VectorXreal))
        {
            processModel = f;
        }

};
