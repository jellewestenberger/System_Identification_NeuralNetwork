function outputs=calcNNOutput(NNset,X)
hiddenoutput={};
Vj={};
nrHiddenlayers=size(NNset.IW,2);
nrMeasurements=size(X,2);
nrNodes=[];
nrInputs=[];
nrOutputs=[];

yi{1}=X(:,:); %input values for all layers (before input weight)
vj={};
yj={};

for k=1:nrHiddenlayers %loop over number of hidden layers
    nrNodes=[nrNodes,size(NNset.IW{k},1)];
    nrInputs=[nrInputs,size(yi{k},1)];
    nrOutputs=[nrOutputs, nrNodes(k)];
    vj{k}=zeros(nrNodes(k),nrMeasurements);
     dvjwij{k}=zeros(nrNodes(k),nrMeasurements);
    Nin=nrInputs(k);
    for i=1:Nin
        xij=yi{k}(i,:).*ones(nrNodes(k),nrMeasurements);
        
        cij=NNset.centers{k}(:,i).*ones(nrNodes(k),nrMeasurements);
        wj=NNset.IW{k}(:,i);
        vj{k}=vj{k}+(wj.*(xij-cij)).^2;
        dvjwij{k,i}=2*(wj.*(xij-cij).^2);
    end
    %output for hidden layer
    %disp(k);
    yj{k}=NNset.a{k}.*exp(-vj{k});
    yi{k+1}=yj{k};
    
    
    %output of current hidden layer is input for next hidden layer
end

%output of output layer;
vk=NNset.LW'.*yi{end};
yk=sum(vk,1);

outputs=struct();
outputs.yi=yi;
outputs.yk=yk;
outputs.vj=vj;
outputs.vk=vj;
outputs.dvjwij=dvjwij;

% hold on
% plot3(atrue,Btrue,Y2'-Cm,'.b');
end
