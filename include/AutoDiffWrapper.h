#pragma once

#include <autodiff/forward/dual.hpp>
// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

using namespace autodiff;
using Eigen::MatrixXd;
class AutoDiffWrapper
{
    public:
        // The single-variable function for which derivatives are needed
        dual f(dual x)
        {
            return 1 + x + x*x + 1/x + log(x);
        }
        template <typename F>
        dual ComputeJacobian1D(F f, dual x)
        {
            return derivative(f, wrt(x), at(x));  // evaluate the derivative du/dx
        }
        template <typename F>
        std::vector <dual> ComputeJacobianND(F f, std::vector< dual > x_dot)
        {
            std::vector <dual> jacobian;   
            for(int i = 0; i < x_dot.size(); i++)
            {
                //jacobian.push_back( derivative(f, wrt( x_dot.at(i) ), at( x_dot.at(i) )) );  // evaluate the derivative du/dx
                jacobian.push_back(ComputeJacobian1D( f,jacobian.at(i)) );
            }
            return jacobian;
        }
        //Jacobian matrix of a vector function
        template <typename jf>
        Eigen::MatrixXd JacobianMatrix(jf f, VectorXreal x, VectorXreal F = VectorXreal())
        {

            return jacobian(f, wrt(x), at(x), F);
        }
        /*
         * Input: Vector of vector functions all of the same variables (x,y,z, etc)
         *          (Up to 4 variables)
         *  Output: 
         */
        Eigen::MatrixXd Jacobian(std::vector<dual (*)(dual , dual)> f, dual _x, dual _y) 
        {
            dual x = _x;
            dual y = _y;

            std::vector<dual> F;
            // 2nd arg - dimensions
            // 1st arg - 
            Eigen::MatrixXd J( f.size() ,2);
            for(int i = 0; i < f.size() ; i++)
            {
                F.push_back( ( f.at(i) )(x,y) ) ;
                auto f1 = f.at(i);
                //std::cout << "Here: " << derivative(f1, wrt(x), at(x, y)) << std::endl;
                J(i, 0) = derivative(f1, wrt(x), at(x, y)) ;
                J(i, 1) = derivative(f1, wrt(y), at(x, y));
                
            }
            return J;
        }
        Eigen::MatrixXd Jacobian(std::vector<dual (*)(dual , dual,dual)> f, dual _x, dual _y, dual _z) 
        {
            dual x = _x;
            dual y = _y;
            dual z = _z;

            std::vector<dual> F;
            // 2nd arg - dimensions
            // 1st arg - 
            Eigen::MatrixXd J( f.size() ,3);
            for(int i = 0; i < f.size() ; i++)
            {
                // Compute the function
                F.push_back( ( f.at(i) )(x,y,z) ) ;
                auto f1 = f.at(i);
                //std::cout << "Here: " << derivative(f1, wrt(x), at(x, y)) << std::endl;
                J(i, 0) = derivative(f1, wrt(x), at(x, y, z)) ;
                J(i, 1) = derivative(f1, wrt(y), at(x, y, z));
                J(i, 2) = derivative(f1, wrt(z), at(x, y, z));
            }
            return J;
        }
        Eigen::MatrixXd Jacobian(std::vector<dual (*)(dual , dual,dual,dual)> f, dual _x, dual _y, dual _z, dual _q) 
        {
            dual x = _x;
            dual y = _y;
            dual z = _z;
            dual q = _q;

            std::vector<dual> F;
            // 2nd arg - dimensions
            // 1st arg - 
            Eigen::MatrixXd J( f.size() ,4);
            for(int i = 0; i < f.size() ; i++)
            {
                // Compute the function
                F.push_back( ( f.at(i) )(x,y,z,q) ) ;
                auto f1 = f.at(i);
                //std::cout << "Here: " << derivative(f1, wrt(x), at(x, y)) << std::endl;
                J(i, 0) = derivative(f1, wrt(x), at(x, y, z,q)) ;
                J(i, 1) = derivative(f1, wrt(y), at(x, y, z,q));
                J(i, 2) = derivative(f1, wrt(z), at(x, y, z,q));
                J(i, 3) = derivative(f1, wrt(q), at(x, y, z,q));
            }
            return J;
        }
        Eigen::VectorXd VectorXRealToVectorxd(VectorXreal vr)
        {
            Eigen::VectorXd vec(vr.size(),1);
            for(int i=0; i<vr.size(); ++i)
            {
                vec[i] = vr[i].val();
            }
            return vec;
        }


};