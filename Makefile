CC=g++
CFLAGS=-I.
LIBS = -larmadillo -ljsoncpp
DEPS = KalmanFilter.h EKF.h
%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) 
#find_package(jsoncpp CONFIG REQUIRED)
#target_link_libraries(main PRIVATE jsoncpp_object jsoncpp_static)	
kalman_filter: main.cpp KalmanFilter.cpp
	$(CC) -o kalman_filter main.cpp KalmanFilter.cpp EKF.cpp $(LIBS) -I.