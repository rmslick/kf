#include<stdio.h>
#include <armadillo>
//https://stackoverflow.com/questions/36861355/fatal-error-with-jsoncpp-while-compiling
#include <jsoncpp/json/json.h>
#include "KalmanFilter.h"
#include <fstream>

using Mat = arma::mat;
Mat jsoncppArrayToMat1D(Json::Value Amat)
{

    int j = 0;
    Mat m( Amat.size() , 1);
    
    for (auto itr2:Amat)
    {
        m.at(j,0) = itr2.asFloat();
        j++;
    }
    return m;
}
Mat jsoncppArrayToMat2D(Json::Value Amat)
{
    Mat A;
    int i =0;
    int j = 0;

    for (auto itr : Amat) {
        Mat m(1 , itr.size() );
        
        for (auto itr2:itr)
        {            
            m.at(1,j-1) = itr2.asFloat();
            j++;
        }
        j=0;
        A.insert_rows(i,m);
        
        i++;
    }
    return A;
}

Mat jsoncppArrayToMat3D(Json::Value Amat)
{
    
    int i =0;
    int j = 0;
    int k = 0;
    Mat A3d;
    for (auto itr : Amat) {
        Mat A;
        for (auto itr2:itr)
        {
            Mat m(1 , itr2.size() );            
            for (auto itr3:itr2)
            {            
                m.at(1,j-1) = itr3.asFloat();
                //std::cout << j << " " << i << " " << k << std::endl;
                j++;
            }
            j=0;    
            A.insert_rows(i,m);
            i++;
        }
        
        i=0;
        //std::cout << A << std::endl;
        A3d.insert_rows(k,A);
        
        k++;
    }
    return A3d;
}

int main()
{
    Json::Value simulation;
    std::ifstream simulation_file("./KalmanPy/simulation_data.json", std::ifstream::binary);
    simulation_file >> simulation;
    
    auto Amat = simulation["A"];
    auto Hmat = simulation["H"];
    auto Qmat = simulation["Q"];
    auto Rmat = simulation["R"];
    auto Bmat = simulation["B"];
    
    auto xmat = simulation["x_init"];
    auto Pmat = simulation["cov_init"];

    auto simulation_states = simulation["simulated_states"];
    auto simulation_measurements = simulation["simulated_measurements"];

    auto predicted_states = simulation["predicted_states"];
    auto predicated_covariances = simulation["predicted_covariances"];


    Mat A = jsoncppArrayToMat2D(Amat);
    Mat H = jsoncppArrayToMat2D(Hmat);
    Mat Q = jsoncppArrayToMat2D(Qmat);
    Mat R = jsoncppArrayToMat2D(Rmat);
    std::cout << "Measurement covariance: " << R << std::endl;
    Mat B;

    Mat x_init = jsoncppArrayToMat1D(xmat); 
    Mat P_init = jsoncppArrayToMat2D(Pmat); 

    // rows correspond to state vectors
    Mat sim_states = jsoncppArrayToMat2D(simulation_states);
    Mat sim_measurements = jsoncppArrayToMat2D(simulation_measurements);

    Mat pred_states = jsoncppArrayToMat2D(predicted_states);
    Mat pred_covariances = jsoncppArrayToMat3D(predicated_covariances);

    KalmanFilter kf(A,Q,B,H,R,x_init, P_init);

    for(int i = 0; i < sim_measurements.n_rows ; i++)
    {
        Mat meas(2,1);
        meas(0,0) = sim_measurements.row(i).at(0,0);
        meas(1,0) = sim_measurements.row(i).at(0,1);

        auto pred_meas = kf(meas);
        auto pred_cov = kf.GetCovariance();

        std::cout <<"Pred meas: " << pred_meas << std::endl;
        std::cout << "Actual pred meas: "<< pred_states.row(i) << std::endl;


    }

    return 0;
}