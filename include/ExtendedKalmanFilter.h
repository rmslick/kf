#pragma once
// Good quat lib: https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
// Multiplicative: https://ntrs.nasa.gov/api/citations/20040037784/downloads/20040037784.pdf
template<typename T>
class KalmanFilterBase
{
    public:
        // in the case of lkf, optional control vector
        // in the case of ekf input to function
        virtual void predict();
        virtual void predict(T x);
        virtual T update(T x) = 0;
        virtual T measure(T x) = 0;
};
// strategy: EKF should remain open for future use on its own
//          therefore Mekf should modify and inherit rather than have a
template<typename T>
class ExtendedKalmanFilter : public KalmanFilterBase<T>
{
    public:
        //virtual f(x T) = 0;
        T (*f)();
        T Q;
        // general state
        T x_k;
        T p_k;

        ExtendedKalmanFilter( T (*_f)(T x) ,  T (*_F)(T x), T _Q )
        {
            // set state estimation model
            f = _f;
            F = _F;
            Q = _Q;
        }
        ExtendedKalmanFilter()
        {}
        // Should also be virtual here
        void predict(T x) {};
        T update(T x){return x;}
        T measure(T x) = 0;
        T setProcessModel( T (*_f)(T x) )
        {
            f = _f;
        }
        T setJacobianModel( T (*_F)(T x) )
        {
            F = _F;
        }
        void predict(T x)
        {
            x_k = f(x);
            P_k = F(x) * P_k * F(x).transpose() + Q; 
        }
        T update(T x)
        {
            return x;
        }
};
template<typename T>
class Quaternion
{
    public:
        std::array<T,4> q;
        Quaternion(){}
        Quaternion(T q0, T q1, T q2, T q3){}
        Quaternion(T r, T p, T y){}
        T operator[] (const int & i) const
        {
            return q[i];
        }
        std::array<T,3> getRadians()
        {
            return std::array<T,3>();
        }

};
/*
// This assumes a matrix class and should remove template
// going to have to be of type matrix NOT quaternion
template<typename T>
class MultiplicativeExtendedKalmanFilter : public ExtendedKalmanFilter<T>
{
    T x_k; // state composed of drift and gyro
    Quaternion<T> q_k_1;
    Quaternion<T> q_k; // rotation
    T d_g_k_1;
    T d_g_k; // Drift from gyro
    // x - state matrix composed of 7x1 dimensions 
    T Q_b_g_k; // covariance associated with gyro
    T Q_g_k; // covariance assosciated with drift
    T w_m_k; //gyro measurement
    T dt; // time interval or rate of gyro
    T I; // Identity
    T zeroes;

    // bias
    T w_b_k;
    T w_g_k;

    void T_dot()
    {

    }
    T processModel( T x )
    {
        // build a quaternion here
        
        q_k_1[0] = x[0];
        q_k_1[1] = x[1];
        q_k_1[2] = x[2];
        q_k_1[3] = x[3];
        // build gyro noise mat here
         
        d_g_k_1[0] = x[4];
        d_g_k_1[1] = x[5];
        d_g_k_1[2] = x[6];
        // quaternion kinematic equation update
        q_k = q_k * q_k_1;
        // walking gyro bias update
        d_g_k = d_g_k_1 + w_b_k;
        // update state
        T x_k_1; 
        x_k_1[0] = q_k[0];
        x_k_1[1] = q_k[1];
        x_k_1[2] = q_k[2];
        x_k_1[3] = q_k[3];
        x_k_1[4] = d_g_k[0];
        x_k_1[5] = d_g_k[1];
        x_k_1[6] = d_g_k[2];
        return x_k_1;
    }
    MultiplicativeExtendedKalmanFilter(T x_k_init, T Q_b_k_init, T Q_g_k_init, T I_init, T zeroes_init):ExtendedKalmanFilter<T>()
    {
        x_k = x_k_init;
        Q_b_g_k =  Q_b_k_init; // See (11) walking bias covariance
        Q_g_k = Q_g_k_init;
        I = I_init;
        zeroes = zeroes_init;
        setProcessModel(processModel);
    }
    // Input: measurement from gyro, prior drift bias
    T F(T x)
    {
        auto r = x.getRadians();
        w_m_k[0] = r[0];
        w_m_k[1] = r[1];
        w_m_k[2] = r[2];
        // extract drift prrediction from prior state
        T d_g_k_prior;
        d_g_k_prior[0] = x_k[4];
        d_g_k_prior[0] = x_k[5];
        d_g_k_prior[0] = x_k[6];

        auto tl = T_dot( w_m_k - d_g_k_prior );
        auto tr = -1*dt*I;
        auto bl  = zeroes;
        auto br = I;
        
        return ;
    }
    // MEKF is a modification of an EKF, so override
    // INPUT: 7d state vector consisting of quaternion and gyro bias model
    // OUTPUT: Updated 7d state vector
    // NOTE: It is assumed that x the rotation is a quaternion
    void predict(T x)
    {

        x_k = f(x);   

    }
};*/

// This assumes a matrix class and should remove template
// going to have to be of type matrix NOT quaternion

// Not templated because assumes functionality from both a quaternion class
// and a matrix class beyond operators
template<typename T>
class MEKF : public ExtendedKalmanFilter<mat<T>>
{
    public:
        // State composed of quaternion for rotation and drift bias
        mat<T> x_k;
        mat<T> x_k_err;
        mat<T> P_k;
        // 6D mat consisting of process noise and drift bias noise
        mat<T> w_k; // [w_g_k, w_b_g_k_1]
        // Covariance process for white noise, function of time differential
        T (*Q_g_k)(mat<T> t); // Yields w_g_k for dt
        T (*Q_b_g_k)(mat<T> t); // Yields w_b_g_k for dt
        T dt;
        mat<T> ones(3,3);
        mat<T> zeroes(3,3);
        mat<T> getM()
        {
            auto tl = -1*_dt*ones;
            auto tr = -1*_dt*ones;
            auto bl = zeroes;
            auto br = ones;
            mat<T> diff(6,6);
            // top left
            diff[0][0] = tl[0][0] ;
            diff[0][1] = tl[0][1] ;
            diff[0][2] = tl[0][2] ;
            
            diff[1][0] = tl[1][0] ;
            diff[1][1] = tl[1][1] ;
            diff[1][2] = tl[1][2] ;
            
            diff[2][0] = tl[2][0] ;
            diff[2][1] = tl[2][1];
            diff[2][2] = tl[2][2];
            
            // top right
            diff[0][3] = tr[0][0] ;
            diff[0][4] = tr[0][1] ;
            diff[0][5] = tr[0][2] ;
            
            diff[1][3] = tr[1][0] ;
            diff[1][4] = tr[1][1] ;
            diff[1][5] = tr[1][2] ;
            
            diff[2][3] = tr[2][0] ;
            diff[2][4] = tr[2][1] ;
            diff[2][5] = tr[2][2] ;

            // bottom left

            diff[3][0] = bl[0][0] ;
            diff[3][1] = bl[0][1] ;
            diff[3][2] = bl[0][2] ;
            
            diff[4][0] = bl[1][0] ;
            diff[4][1] = bl[1][1] ;
            diff[4][2] = bl[1][2] ;
            
            diff[5][0] = bl[2][0] ;
            diff[5][1] = bl[2][1] ;
            diff[5][2] = bl[2][2] ;

            // bottom right
            diff[3][3] = br[0][0] ;
            diff[3][4] = br[0][1] ;
            diff[3][5] = br[0][2] ;
            
            diff[4][3] = br[1][0] ;
            diff[4][4] = br[1][1] ;
            diff[4][5] = br[1][2] ;
            
            diff[5][3] = br[2][0] ;
            diff[5][4] = br[2][1] ;
            diff[5][5] = br[2][2] ;
            return diff
        } 
        mat<T> getQ()
        {
            auto tl = Q_g_k(dt)
            auto tr = zeroes;
            auto bl = zeroes;
            auto br = Q_b_g_k(dt);
            mat<T> diff(6,6);
            // top left
            diff[0][0] = tl[0][0] ;
            diff[0][1] = tl[0][1] ;
            diff[0][2] = tl[0][2] ;
            
            diff[1][0] = tl[1][0] ;
            diff[1][1] = tl[1][1] ;
            diff[1][2] = tl[1][2] ;
            
            diff[2][0] = tl[2][0] ;
            diff[2][1] = tl[2][1];
            diff[2][2] = tl[2][2];
            
            // top right
            diff[0][3] = tr[0][0] ;
            diff[0][4] = tr[0][1] ;
            diff[0][5] = tr[0][2] ;
            
            diff[1][3] = tr[1][0] ;
            diff[1][4] = tr[1][1] ;
            diff[1][5] = tr[1][2] ;
            
            diff[2][3] = tr[2][0] ;
            diff[2][4] = tr[2][1] ;
            diff[2][5] = tr[2][2] ;

            // bottom left

            diff[3][0] = bl[0][0] ;
            diff[3][1] = bl[0][1] ;
            diff[3][2] = bl[0][2] ;
            
            diff[4][0] = bl[1][0] ;
            diff[4][1] = bl[1][1] ;
            diff[4][2] = bl[1][2] ;
            
            diff[5][0] = bl[2][0] ;
            diff[5][1] = bl[2][1] ;
            diff[5][2] = bl[2][2] ;

            // bottom right
            diff[3][3] = br[0][0] ;
            diff[3][4] = br[0][1] ;
            diff[3][5] = br[0][2] ;
            
            diff[4][3] = br[1][0] ;
            diff[4][4] = br[1][1] ;
            diff[4][5] = br[1][2] ;
            
            diff[5][3] = br[2][0] ;
            diff[5][4] = br[2][1] ;
            diff[5][5] = br[2][2] ;
            return diff
        }
        void setDT()
        {
            // calculate time differential between calls
            // and set dt
            
        }
        mat<T> statePrediction(mat<T> w_m_k)
        {
            // get Q_k_1
            auto Q_k_1 = getQ();
            // update time differential
            setDT();
            // extract the drift
            auto d_g_k_1 = x_k[4:7];
            // extract bias drift noise from prior timestep
            auto w_b_g_k_1 = w_k[3:6];
            // extract prior rotation
            auto q_k_1 = Quaternion( x_k[0],x_k[1],x_k[2],x_k[3] );
            // Update w_g_k_1 to w_g_k for this timestep
            w_k[0:3] = Q_g_k(dt);
            auto w_g_k = w_k[0:3];
            // See equation 12
            auto w_k_1 = w_m_k - d_g_k_1 - w_b_g_k_1 - w_g_k;
            // See equation 15 - compute true measurement
            auto w_k_dt = w_k_1*dt;
            auto q_meas = Quaternion(w_k_dt[0],w_k_dt[1],w_k_dt[2])
            auto q_pred = q_meas * q_k_1;
            auto d_g_k = d_g_k_1+ w_b_g_k_1; 

            // Update the error
            M_k = getM();
            auto F_k_1 = F( T_ang( w_m_k - d_g_k_1 ) );
            x_k_err = F_k_1 * x_k_err + M_k*w_k;
            // Update state
            x_k = Mat<T>(q_pred[0],q_pred[1],q_pred[2],q_pred[3],
                        d_g_k[0],d_g_k[1],d_g_k[2]);
            // update covariance
            P_k = F_k_1* P_k * F_k_1.transpose() + M_k *  * M_k.transpose();
            // update to w_b_g_k
        }
        // INPUT 3D angle
        mat<T> T_ang(mat<T> angle)
        {
            // is this correct? 
            auto magnitude = sqrt( anglo[0] + angle[1] + angle[2] );
            auto normalized = angle*(1/magnitude);
            auto skewd_angled = normalized.skew();
            return ones - sin(magnitude)*skewd_angled + ((1-cos(magnitude))*(skewd_angled*skewd_angled));
        }
        mat<T> processModel( mat<T> w_m_k )
        {
            // propogate state
            statePrediction(w_m_k);   
            // propogate state error
            w_k[4:6] = Q_b_g_k(dt);

        }
        mat<T> jacobian(mat<T> m)
        {
            auto tl = T_ang(m);
            auto tr = -1*dt*ones;
            auto bl = zeroes;
            auto br = ones;
            mat<T> diff(6,6);
            // top left
            diff[0][0] = tl[0][0] ;
            diff[0][1] = tl[0][1] ;
            diff[0][2] = tl[0][2] ;
            
            diff[1][0] = tl[1][0] ;
            diff[1][1] = tl[1][1] ;
            diff[1][2] = tl[1][2] ;
            
            diff[2][0] = tl[2][0] ;
            diff[2][1] = tl[2][1];
            diff[2][2] = tl[2][2];
            
            // top right
            diff[0][3] = tr[0][0] ;
            diff[0][4] = tr[0][1] ;
            diff[0][5] = tr[0][2] ;
            
            diff[1][3] = tr[1][0] ;
            diff[1][4] = tr[1][1] ;
            diff[1][5] = tr[1][2] ;
            
            diff[2][3] = tr[2][0] ;
            diff[2][4] = tr[2][1] ;
            diff[2][5] = tr[2][2] ;

            // bottom left

            diff[3][0] = bl[0][0] ;
            diff[3][1] = bl[0][1] ;
            diff[3][2] = bl[0][2] ;
            
            diff[4][0] = bl[1][0] ;
            diff[4][1] = bl[1][1] ;
            diff[4][2] = bl[1][2] ;
            
            diff[5][0] = bl[2][0] ;
            diff[5][1] = bl[2][1] ;
            diff[5][2] = bl[2][2] ;

            // bottom right
            diff[3][3] = br[0][0] ;
            diff[3][4] = br[0][1] ;
            diff[3][5] = br[0][2] ;
            
            diff[4][3] = br[1][0] ;
            diff[4][4] = br[1][1] ;
            diff[4][5] = br[1][2] ;
            
            diff[5][3] = br[2][0] ;
            diff[5][4] = br[2][1] ;
            diff[5][5] = br[2][2] ;
            return diff;
        }
        
        MEKF( T(*_Q_g_k)(T x), T(*_Q_b_g_k)(T x), T(*F)(T x) )
        {
            Q_g_k = _Q_g_k;
            Q_b_g_k = _Q_b_g_k;
            setProcessModel(processModel);
            setJacobianModel(jacobian);
        }
        

};