#include<stdio.h>
#include <armadillo>
//https://stackoverflow.com/questions/36861355/fatal-error-with-jsoncpp-while-compiling
#include <jsoncpp/json/json.h>
#include "LinearKF.h"
#include "AutoDiffWrapper.h"
#include <fstream>
#include "EKF.h"
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
    processModelCV * pmcv = new processModelCV();
    observationModelCV * omcv = new observationModelCV();
    EKF ekf(A,Q,H,R,x_init,P_init,pmcv,omcv );
    std::cout << A << std::endl;
    for(int i = 0; i < sim_measurements.rows() ; i++)
    {
        ArmaMat meas(2,1);
        meas(0,0) = (sim_measurements.row(i))(0,0);
        meas(1,0) = (sim_measurements.row(i))(0,1);

        auto x_dot = example_cast_eigen(meas);
        
        auto pred_meas = kf(x_dot);
        auto pred_cov = kf.GetCovariance();

        auto pred_meas_ekf = ekf(x_dot);
        auto pred_cov_ekf = ekf.GetCovariance();
        std::cout << "\n\n";
        std::cout <<"Pred meas: " << pred_meas(0,0) << " " << pred_meas(1,0)<<" " << pred_meas(2,0)<<" " << pred_meas(3,0) << std::endl;
        std::cout <<"Pred meas ekf: " << pred_meas_ekf(0,0) <<" " << pred_meas_ekf(1,0)<<" " << pred_meas_ekf(2,0)<<" " << pred_meas_ekf(3,0) << std::endl;
        std::cout << "Actual pred meas: "<< pred_states.row(i) << std::endl;
        //std::cout << "\n\n";
        //std::cout <<"Pred cov: \n" << pred_cov <<  std::endl;
        //std::cout <<"Pred cov ekf: \n" << pred_cov_ekf << std::endl;
        //std::cout << "Actual cov: "<< pred_states.row(i) << std::endl;
        //std::cout << "\n\n";
        if(pred_meas != pred_meas_ekf)
        {

            //std::cout << "Exiting gracefully\n";
            //break;
        }
    }


    return 0;
}