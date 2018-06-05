%Measurement matrix

function mat=calc_MeasurementMat(t,X,V)

u=X(1);
v=X(2);
w=X(3);
Ca=X(4);
va=V(1);
vb=V(2);
vv=V(3);

a_true=atan2(w,u);
b_true=atan2(v,(sqrt(u.^2+w.^2)));
V_true=sqrt(u.^2+v.^2+w.^2);
mat=[a_true*(1+Ca)+va; b_true+vb; V_true+vv];%Check if fraction is done correctly (elementwise or not) 
end