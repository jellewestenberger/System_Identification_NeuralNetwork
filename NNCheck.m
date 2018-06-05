function check=NNCheck(NNset,nrInput,nrNodesHidden,nrOutput)
check=1;
if not(size(NNset.IW,1)==nrNodesHidden) || not(size(NNset.IW,2)== nrInput)
    fprintf('Input weights array is not correct size \n Size should be: \n [#hidden neurons in first hidden layer, #input neurons]');
    check=0;
end

if not(size(NNset.LW,1)==nrOutput) || not(size(NNset.LW,2)== nrNodesHidden(end))
    fprintf('Output weights array is not correct size \n Size should be: \n [#output neurons, #hidden neurons in last hidden layer]');
    check=0;
end

if not(size(NNset.range,1)==nrInput) || not(size(NNset.range,2)== 2)
    fprintf('range array is not correct size \n Size should be: \n [#input neurons, 2]');
    check=0;
end

if strcmp(NNset.name,'feedforward')
    for i=1:(nrHiddenlayers+1) %already taking into account the possibility for multiple hidden layers (which might be out of scope of this assignment)
       if i<=nrHiddenlayers
           if not(size(NNset.b{i,1},1)==nrNodesHidden(i))
               fprintf('Input bias weight for layer \n');
               disp(i);
              fprintf('is not the correct size');
              check=0;
           end
           if i==(nrHiddenlayers+1)
                if not(size(NNset.b{i,1},1)==nrOutput)
                    fprintf('Input bias weight for output layer does not have the correct size\n Should be:\n [#output neurons, 1]')';
                    check=0;
                end    
           end
       end
    end
end
if strcmp(NNset.name,'rbf')
    if not(size(NNset.centers,1)==nrNodesHidden(1)) || not(size(NNset.centers,2)== nrInput)
        fprintf('Centers array has not correct size\n Should be:\n [#nr hidden neurons, #input neurons]');
        check=0;
    end
end
if check
    disp('Neural Network Structure has correct sizes')
end