function NNset=LevMar(NNset,Cm,X,mu_inc,mu_dec,evaltot,plotf)

atrue=X(1,:);
Btrue=X(2,:);
Vtrue=X(3,:);

%calculate dE/d Wjk   (Wjk = output weight);

outputs=calcNNOutput(NNset,X); 

El=[];
E2old=inf;
mu1=1;
mu2=1;
for eval=1:evaltot

        close all;
    yk=outputs.yk; %total output of neural network
    yi=outputs.yi; % inputs to input weights in each layer 

    if plotf
        figure
        plot3(atrue,Btrue,yk,'.b'); 
        title(strcat('evaluation ',num2str(eval)));
        hold on
        plot3(atrue,Btrue,Cm,'.k');
        refreshdata
        drawnow
        pause
    end

    ekq=Cm'-yk;         %IO mapping errors
    E1=sum(0.5*ekq.^2); %squared error
    El=[El, E1];

    if plotf
        figure
        plot(El)
        drawnow
    end

    disp(E1);
    %% calculate partial derivative wrt to output weights
    if 1

    dEdWjk=yi{1,2}*ekq'*(-1);
    J=dEdWjk;
    d=LM(J,E1,mu1);
    NNset.LW=NNset.LW-d;
    end

    outputs=calcNNOutput(NNset,X);
    yk=outputs.yk; %total output of neural network
    yi=outputs.yi;
    ekq=Cm'-yk;  
    E2=sum(0.5*ekq.^2);

    %% Adapt damping factor of LM algorithm
    if E2>E1  
        mu1=mu1+mu_inc;
    else
        mu1=mu1-mu_dec;
    end

    if E2old>E2
        mu2=mu2+mu_inc;
%         mu1=mu1+mu_inc;
    else
        mu2=mu2-mu_dec;
%         mu1=mu1-mu_dec;
    end

    %% calculate partial derivatives wrt input weights
    if 0 %switch LM on input weights on or off

    dPhijdVj=-NNset.a{1}.*exp(-outputs.vj{1});
    dVjdWij=outputs.dvjwij;

    d=[];

    for i=1:size(dVjdWij,2)
    dEdWij=dVjdWij{1,i}.*dPhijdVj*ekq';
    d=[d;LM(dEdWij,E2,mu2)];
    end
    d=d';
    NNset.IW{1}=NNset.IW{1}+d;
    end
    E2old=E2;    

end

function d=LM(J,E,mu)
d=((J'*J)+mu)^(-1)*J'*E;
end

end