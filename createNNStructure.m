function NNset=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,epoch,inittype,y)
nrHiddenlayers=size(nrNodesHidden,2);
inputs=[nrInput, nrNodesHidden];
NNset.range=inputrange;

if strcmp(Networktype, 'ff')
    for h=1:(nrHiddenlayers)
    NNset.b{h,1}=zeros(nrNodesHidden(h),1); %input bias weights     
    NNset.trainFunct{h,1}='tansig';
    end
    NNset.b{h+1,1}=zeros(nrOutput,1);
    NNset.trainFunct{h+1,1}='purelin';
    NNset.name{1,1}='feedforward';
    
elseif strcmp(Networktype,'rbf')
    NNset.trainFunct{1,1}='radbas';
    
    for h = 1:nrHiddenlayers
    NNset.trainFunct{h,1}='radbas' ; 
    NNset.centers{h}=zeros(nrNodesHidden(h),inputs(h));     %center locations 
    NNset.center_dist='uniform';                    %center location distribution type
        if h==1
            if strcmp(NNset.center_dist,'uniform') %uniform centeFr distribution (based on input range)
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
             NNset.a{h}=ones(nrNodesHidden(h),1);
        end
        if strcmp(inittype,'random')
            NNset.a{h}=randn(nrNodesHidden(h),1)*1e-2;
        end
    end 
    NNset.name{1,1}='rbf';
    NNset.trainFunct{nrHiddenlayers+1,1}='purelin' ;
end

for h =1:nrHiddenlayers
if strcmp(inittype,'ones')   
    NNset.IW{h}=ones(nrNodesHidden(h),inputs(h)); %1's for now  INPUT WEIGHTS
    NNset.LW=ones(nrOutput,nrNodesHidden(end));%OUTPUT WEIGHTS (end because only last hidden layer connects to output) 
   
end
if strcmp(inittype,'random')
    NNset.IW{h}=randn(nrNodesHidden(h),inputs(h)); %1's for now  INPUT WEIGHTS
    NNset.LW=randn(nrOutput,nrNodesHidden(end));%OUTPUT WEIGHTS (end because only last hidden layer connects to output) 
    
end
end


NNset.trainParam.epochs=epoch;
NNset.trainParam.goal=0;
NNset.trainParam.min_grad=1e-10;
NNset.trainParam.mu=1e-5; %learning rate 
NNset.trainParam.mu_dec=1e-2; 
NNset.trainParam.mu_inc=10;
NNset.trainParam.mu_max=1e5; 
NNset.trainalg=('trainlm');

end