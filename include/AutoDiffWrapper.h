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
        template <typename function>
        VectorXreal JacobianMatrix(function f, VectorXreal x, VectorXreal F)
        {

            return jacobian(f, wrt(x), at(x), F);
        }

};