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
dual f1_3dtest(dual x, dual y,dual z)
{
    return z* tan(x*x-y*y);
}
dual f2_3dtest(dual x, dual y,dual z)
{
    return x*y*log(z*0.5);
}

TEST(AutoDiffTest, JacobianTestVector3D)
{   // The multi-variable function for which higher-order derivatives are needed (up to 4th order)
    // The multi-variable function for which derivatives are needed

    dual x = 2.0;
    dual y = -2.0;
    dual z = 2.0;

    Eigen::MatrixXd J(2,3);
    J(0,0) = derivative(f1_3dtest, wrt(x), at(x, y,z)) ;
    J(0,1) = derivative(f1_3dtest, wrt(y), at(x, y,z)) ;
    J(0,2) = derivative(f1_3dtest, wrt(z), at(x, y,z)) ;
    J(1,0) = derivative(f2_3dtest, wrt(x), at(x, y,z)) ;
    J(1,1) = derivative(f2_3dtest, wrt(y), at(x, y,z)) ;
    J(1,2) = derivative(f2_3dtest, wrt(z), at(x, y,z)) ;
    
    std::cout <<"From 3d!\n" << J << std::endl;
    AutoDiffWrapper ad;
    std::vector<dual (*)(dual , dual,dual) > f;
    f.push_back(f1_3dtest);
    f.push_back(f2_3dtest);
    ASSERT_EQ ( ad.Jacobian(f,x,y,z), J );
}

    dual f1(dual x, dual y)
    {
        return pow(x,4) + 3*pow(y,2)*x;
    }
    dual f2(dual x, dual y)
    {
        return 5*pow(y,2)-2*x*y+1;
    }
    dual f1c(dual x, dual y)
    {
        return pow(x,4) + 3*pow(y,2)*x;
    }
    dual f2c(dual x, dual y)
    {
        return 5*pow(y,2)-2*x*y+1;
    }
TEST(AutoDiffTest, JacobianTestVector2D)
{   // The multi-variable function for which higher-order derivatives are needed (up to 4th order)
    // The multi-variable function for which derivatives are needed

    dual x = 1.0;
    dual y = 2.0;
    dual z = 3.0;

    dual u = f1(x, y);
    Eigen::MatrixXd J(2,2);
    J(0,0) = derivative(f1, wrt(x), at(x, y)) ;
    J(0,1) = derivative(f1, wrt(y), at(x, y));

    J(1,0) = derivative(f2, wrt(x), at(x, y));
    J(1,1) = derivative(f2, wrt(y), at(x, y));
    
    std::cout << J << std::endl;
    AutoDiffWrapper ad;
    std::vector<dual (*)(dual , dual) >f;
    f.push_back(f1);
    f.push_back(f2);
    ASSERT_EQ ( ad.Jacobian(f,x,y), J );
}

TEST (AutoDiffTest, JacobianNDTest2D)
{
    dual x = 2.0;
    dual y = -2.0;
    dual z = 2.0;

    Eigen::MatrixXd J(2,3);
    J(0,0) = derivative(f1_3dtest, wrt(x), at(x, y,z)) ;
    J(0,1) = derivative(f1_3dtest, wrt(y), at(x, y,z)) ;
    J(0,2) = derivative(f1_3dtest, wrt(z), at(x, y,z)) ;
    J(1,0) = derivative(f2_3dtest, wrt(x), at(x, y,z)) ;
    J(1,1) = derivative(f2_3dtest, wrt(y), at(x, y,z)) ;
    J(1,2) = derivative(f2_3dtest, wrt(z), at(x, y,z)) ;
    
    std::cout <<"From 3d!\n" << J << std::endl;
    
    //std::cout << J << std::endl;
    AutoDiffWrapper ad;
    std::vector<dual (*)(dual , dual,dual) >f;
    //f.push_back(f1c);
    //f.push_back(f2c);
    f.push_back(f1_3dtest);
    f.push_back(f2_3dtest);

    ASSERT_EQ ( ad.Jacobian(f, x, y , z) ,J );
}
/*
TEST (AutoDiffTest, JacobianNDTest3D)
{
    dual x = 2.0;
    dual y = -2.0;
    dual z = 2.0;

    Eigen::MatrixXd J(2,3);
    J(0,0) = derivative(f1_3dtest, wrt(x), at(x, y,z)) ;
    J(0,1) = derivative(f1_3dtest, wrt(y), at(x, y,z)) ;
    J(0,2) = derivative(f1_3dtest, wrt(z), at(x, y,z)) ;
    J(1,0) = derivative(f2_3dtest, wrt(x), at(x, y,z)) ;
    J(1,1) = derivative(f2_3dtest, wrt(y), at(x, y,z)) ;
    J(1,2) = derivative(f2_3dtest, wrt(z), at(x, y,z)) ;
    
    std::cout <<"From 3d!\n" << J << std::endl;
    AutoDiffWrapper ad;
    std::vector<dual (*)(dual , dual,dual) >f;
    f.push_back(f1_3dtest);
    f.push_back(f2_3dtest);
    ASSERT_EQ ( ad.Jacobian(f,std::vector<dual>{x,y,z} ),J );

}*/
// The vector function for which the Jacobian is needed
VectorXreal functorama (const VectorXreal& x)
{
    return x * x.sum();
}
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);

    //std::cout << "du/dz = " << dudz << std::endl;  // print the evaluated derivative du/dz
    return RUN_ALL_TESTS();
}