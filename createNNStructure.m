function NNset=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,inittype)
nrHiddenlayers=size(nrNodesHidden,2);
inputs=[nrInput, nrNodesHidden];
NNset.range=inputrange;

if strcmp(Networktype, 'ff')
    NNset.b{1,1}=zeros(nrNodesHidden,1); %input bias weights
    NNset.b{2,1}=zeros(nrOutput,1);     %output bias weights
    NNset.name{1,1}='feedforward';
elseif strcmp(Networktype,'rbf')
    NNset.trainFunct{1,1}='radbas';
    
    for h = 1:nrHiddenlayers
    NNset.centers{h}=zeros(nrNodesHidden(h),inputs(h));     %center locations 
    NNset.center_dist='uniform';                    %center location distribution type
        if h==1
            if strcmp(NNset.center_dist,'uniform') %uniform center distribution (based on input range)
                for i=1:nrInput
                   minin=NNset.range(i,1);
                   minout=NNset.range(i,2);
                   NNset.centers{h}(:,i)=linspace(minin,minout,nrNodesHidden(1))';
                end  
            end
        else
            NNset.centers{h}=zeros(nrNodesHidden(h),inputs(h));
        end
    if strcmp(inittype,'ones')   
    NNset.IW{h}=ones(nrNodesHidden(h),inputs(h)); %1's for now  INPUT WEIGHTS
    NNset.LW=ones(nrOutput,nrNodesHidden(end));%OUTPUT WEIGHTS (end because only last hidden layer connects to output) 
    NNset.a{h}=ones(nrNodesHidden(h),1);
    end
    if strcmp(inittype,'random')
    NNset.IW{h}=randn(nrNodesHidden(h),inputs(h)); %1's for now  INPUT WEIGHTS
    NNset.LW=randn(nrOutput,nrNodesHidden(end));%OUTPUT WEIGHTS (end because only last hidden layer connects to output) 
    NNset.a{h}=randn(nrNodesHidden(h),1);
    end
    end 
    NNset.name{1,1}='rbf';
end



NNset.trainParam.epochs=100;
NNset.trainParam.goal=0;
NNset.trainParam.min_grad=1e-10;
NNset.trainParam.mu=1e-3; %learning rate 
NNset.trainParam.mu_dec=0.1; 
NNset.trainParam.mu_inc=10;
NNset.trainParam.mu_max=1e10; 
NNset.trainalg=('trainlm');

end