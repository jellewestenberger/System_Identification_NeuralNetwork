%calculate jacobian matrix of output

function Hx=calc_Jacob_out(X)
u=X(1);
v=X(2);
w=X(3);
Ca=X(4);

Hx=[(1/(1+(w/u)^2))*(-w/u^2)*(1+Ca) 0 (1+Ca)*(1/(1+(w/u)^2))*(1/u) atan2(w,u);
    (1/(1+(v/sqrt(u^2+w^2))^2))*(-u*v*(u^2+w^2)^(-3/2)) (1/(1+(v/sqrt(u^2+w^2))^2))*(1/sqrt(u^2+w^2)) (1/(1+(v/sqrt(u^2+w^2))^2))*(-v*w*(u^2+w^2)^(-3/2)) 0;
    u/sqrt(u^2+w^2+v^2) v/sqrt(u^2+w^2+v^2) w/sqrt(u^2+w^2+v^2) 0];

end