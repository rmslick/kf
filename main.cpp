#include<stdio.h>
#include <armadillo>
//https://stackoverflow.com/questions/36861355/fatal-error-with-jsoncpp-while-compiling
#include <jsoncpp/json/json.h>
#include "LinearKF.h"
#include "AutoDiffWrapper.h"
#include <fstream>

using ArmaMat = arma::mat;
ArmaMat jsoncppArrayToMat1D(Json::Value Amat)
{

    int j = 0;
    ArmaMat m( Amat.size() , 1);
    
    for (auto itr2:Amat)
    {
        m.at(j,0) = itr2.asFloat();
        j++;
    }
    return m;
}
ArmaMat jsoncppArrayToMat2D(Json::Value Amat)
{
    ArmaMat A;
    int i =0;
    int j = 0;

    for (auto itr : Amat) {
        ArmaMat m(1 , itr.size() );
        
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

ArmaMat jsoncppArrayToMat3D(Json::Value Amat)
{
    
    int i =0;
    int j = 0;
    int k = 0;
    ArmaMat A3d;
    for (auto itr : Amat) {
        ArmaMat A;
        for (auto itr2:itr)
        {
            ArmaMat m(1 , itr2.size() );            
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
Eigen::MatrixXd example_cast_eigen(arma::mat arma_A) {

  Eigen::MatrixXd eigen_B = Eigen::Map<Eigen::MatrixXd>(arma_A.memptr(),
                                                        arma_A.n_rows,
                                                        arma_A.n_cols);

  return eigen_B;
}
int main()
{
    Json::Value simulation;
    std::ifstream simulation_file("/home/rmslick/KalmanFilter/KalmanFilterC/KalmanPy/simulation_data.json", std::ifstream::binary);
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


    auto A = example_cast_eigen( jsoncppArrayToMat2D(Amat) );
    auto H = example_cast_eigen( jsoncppArrayToMat2D(Hmat) );
    auto Q = example_cast_eigen( jsoncppArrayToMat2D(Qmat) );
    auto R = example_cast_eigen( jsoncppArrayToMat2D(Rmat) );
    auto x_init = example_cast_eigen( jsoncppArrayToMat1D(xmat) ); 
    auto P_init = example_cast_eigen( jsoncppArrayToMat2D(Pmat) ); 
    auto sim_states = example_cast_eigen( jsoncppArrayToMat2D(simulation_states) );
    auto sim_measurements = example_cast_eigen( jsoncppArrayToMat2D(simulation_measurements) );
    auto pred_states = example_cast_eigen( jsoncppArrayToMat2D(predicted_states) );
    auto pred_covariances = example_cast_eigen( jsoncppArrayToMat3D(predicated_covariances) );

    LinearKF kf(A,Q,H,R,x_init, P_init);

    for(int i = 0; i < sim_measurements.rows() ; i++)
    {
        ArmaMat meas(2,1);
        meas(0,0) = (sim_measurements.row(i))(0,0);
        meas(1,0) = (sim_measurements.row(i))(0,1);
        auto x_dot = example_cast_eigen(meas);
        auto pred_meas = kf(x_dot);
        auto pred_cov = kf.GetCovariance();

        std::cout <<"Pred meas: " << pred_meas << std::endl;
        std::cout << "Actual pred meas: "<< pred_states.row(i) << std::endl;


    }


    return 0;
}