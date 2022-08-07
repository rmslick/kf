#pragma once
#include <stdio.h>
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
    private:
        Mat _A, _Q, _B, _H, _R;
        // State and covariance
        Mat _x, _P;
        Mat _zeros;
    public:
    /*
        Parameters
            - A : State transition matrix
            - Q : Process error covariance
            - B : Control matrix
            - H : Observation matrix
            - R : Measurement error covariance
    */
        KalmanFilter(Mat A, Mat Q, Mat H, Mat R, Mat x, Mat P);
        KalmanFilter(Mat A, Mat Q, Mat B, Mat H, Mat R, Mat x, Mat P);
    /*
     * Gaussian white noise selection
     */
        Mat WhiteNoise(Mat mean, Mat covariance);
    
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
        Mat operator () (Mat z)
        {
            // Predict state and covariance
            
            Predict();
            
            // Update state and covariance 
            Update(z);
            // Return the predicted state
            return _x;
        }
        

};
