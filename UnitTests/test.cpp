// whattotest.cpp

#include <math.h>
#include <gtest/gtest.h> 
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

#include "AutoDiffWrapper.h"
using namespace autodiff;

double squareRoot(const double a) {
    double b = sqrt(a);
    if(b != b) { // nan check
        return -1.0;
    }else{
        return sqrt(a);
    }
}
 
TEST(AutoDiffTest, VectorTesting) {
    AutoDiffWrapper ad;

    VectorXreal vxr(10,1);
    Eigen::VectorXd test_vec(10,1);

    VectorXreal test_vxr(10,1);
    Eigen::VectorXd ev(10,1);

    for(int i = 0; i < 10; i++)
    {
        vxr[i] = i*10;
        test_vec[i] = i*10;

        ev[i] = i*10;
        test_vxr[i] = i*10;

    }
    Eigen::VectorXd pred_vec = ad.VectorXRealToVectorxd( vxr );
    ASSERT_EQ(pred_vec, test_vec );

}
// The vector function for which the Jacobian is needed
ArrayXreal f(const ArrayXreal& x)
{
    return x*x  ;
}


TEST(AutoDiffTest, JacobianMatrixTesting) {
    AutoDiffWrapper ad;
    ArrayXreal x(5);                           // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;                         // x = [1, 2, 3, 4, 5]
    //ArrayXreal x(5);                           // the input vector x with 5 variables
    //x << 1, 2, 3, 4, 5;                         // y = [1, 2, 3, 4, 5]
    Eigen::MatrixXd J = ad.JacobianMatrix(f,x);
    std::cout << "Jacobian: " << J << std::endl;
}

TEST(AutoDiffTest, JacobianTest2D) {
    AutoDiffWrapper ad;
    ArrayXreal x(5);                           // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;                         // x = [1, 2, 3, 4, 5]
    //ArrayXreal x(5);                           // the input vector x with 5 variables
    //x << 1, 2, 3, 4, 5;                         // y = [1, 2, 3, 4, 5]
    Eigen::MatrixXd J = ad.JacobianMatrix(f,x);
    std::cout << "Jacobian: " << J << std::endl;
}

// The multi-variable function for which higher-order derivatives are needed (up to 4th order)
// The multi-variable function for which derivatives are needed
dual f1(dual x, dual y)
{
    return pow(x,4) + 3*pow(y,2)*x;
}
dual f2(dual x, dual y)
{
    return 5*pow(y,2)-2*x*y+1;
}
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    dual x = 1.0;
    dual y = 2.0;
    dual z = 3.0;

    dual u = f1(x, y);
    Eigen::MatrixXd J(2,2);
    J(0,0) = derivative(f1, wrt(x), at(x, y));
    J(0,1) = derivative(f1, wrt(y), at(x, y));

    J(1,0) = derivative(f2, wrt(x), at(x, y));
    J(1,1) = derivative(f2, wrt(y), at(x, y));
    
    std::cout << J << std::endl;
    
    //std::cout << "du/dz = " << dudz << std::endl;  // print the evaluated derivative du/dz
    return RUN_ALL_TESTS();
}