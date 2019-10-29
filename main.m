do_kalman=0;
do_parameter_estimation=0;
do_radial_basis_NN=0;
do_feedforward_NN=0;


%%Part 2a: State Estimation
if do_kalman
    kalman
end
%% part 2b: Parameter Estimation
if do_parameter_estimation
    ParEstimator
end
%% Part 3: Radial Basis Function Neural Network (at the beginning of the file you can select which part of the report should be executed
if do_radial_basis_NN
    RBFNN
end
%% Part 4: Feed-forward Neural Network
if do_feedforward_NN
    FFNN 
end