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
     dvjcij{k}=zeros(nrNodes(k),nrMeasurements);
    Nin=nrInputs(k);
    
    if strcmp(NNset.trainFunct{k,1},'radbas')
        for i=1:Nin
            xij=yi{k}(i,:).*ones(nrNodes(k),nrMeasurements);
            cij=NNset.centers{k}(:,i).*ones(nrNodes(k),nrMeasurements);
            wj=NNset.IW{k}(:,i);
            vj{k}=vj{k}+(wj.*(xij-cij)).^2;
            dvjwij{k,i}=2*(wj.*(xij-cij).^2);
            dvjcij{k,i}=-2*((wj.^2).*(xij-cij));
        end
        
        yj{k}=NNset.a{k}.*exp(-vj{k});%output for hidden layer
    end
    if strcmp(NNset.trainFunct{k,1},'tansig')
        vj{k}=(NNset.IW{k}*yi{k});
        if strcmp(NNset.name{1},'feedforward')
        vj{k}=vj{k}+NNset.b{k,1}*ones(1,nrMeasurements); %add bias if feedforward
        end
       yj{k}=(2./(1+exp(-2*vj{k})))-1;
    end
        
    yi{k+1}=yj{k}; %output hiddenlayer is input next hidden layer

end

%output of output layer;
vk=NNset.LW'.*yi{end};
% if strcmp(NNset.
if strcmp(NNset.name{1},'feedforward')
     yk=sum(vk,1)+NNset.b{end}*ones(1,nrMeasurements);    
elseif strcmp(NNset.name{1},'rbf')
    yk=sum(vk,1);
end
    

outputs=struct();
outputs.yi=yi;
outputs.yk=yk;
outputs.vj=vj;
outputs.vk=vj;
outputs.dvjwij=dvjwij;
outputs.dvjcij=dvjcij;
% hold on
% plot3(atrue,Btrue,Y2'-Cm,'.b');
end
