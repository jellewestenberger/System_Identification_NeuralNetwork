# SystemIdentification
TU Delft AE4320 Neural Networks Assignment

In this individual assignment for the course "System Identification of Aerospace Systems" a set measurements is being analyzed and approximated by various methods. 

The measurements consist of 10001 measurment points of the moment coefficient Cm at different values of angle of attack, angle of sideslip and airspeed (alpha, beta and V).
Furthermore, a set of 'perfect' accelerometer measurements are supplied. These values give at x,y,z accelerations of the aircraft.

It is known that the measured alpha is biased with respect to the true alpha. 
A simple extended kalman filter is constructed that estimates the bias (by deducting alpha from the accelerometers as well).

The measurements are corrected for the estimated bias and the data approximation process starts.

Firstly a simple least-squares estimated linear polynomial is constructed to match the data. 

Subsequently, a radial basis function Neural Network is constructed as well as a feed-forward neural network to approximate the data.


#Usage Instructions
Navigate from "main.m" to perform the different operations corresponding to the chapters in the report. 
Note that the operations of the NeuralNetworks can take a long time. Therefore you should select which sections of the chapter to run at the beginnging of RBFNN.m and FFNN.m
Additionally, at the beginning of each file some configuration flags are added such as: plotf, plottraining and savef. 
Set this values to 1 or 0 to enable/disable plotting the results, live plotting the neural network training (slows down training) and saving the results, respectively. 

