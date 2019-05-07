function check_observability 

syms('u','v', 'w','Caup','u_dot','v_dot','w_dot');
x=[u;v;w;Caup];
x_0=[100;2;11;0.5];

f=[u_dot;v_dot;w_dot;0];
h=calc_MeasurementMat(x,[u;v;w]);

rank=kf_calcNonlinObsRank(f,h,x,x_0);
dummy=2;
end

