%used for ode45 solver at beginning of kalman filter
function Xdot= calc_f(t,X,U)
 Xdot=U; 
end
   