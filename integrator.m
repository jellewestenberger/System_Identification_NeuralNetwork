%function to integrate u,v,w

function [t,x1]=integrator(xi,dx,t,dt)
x1=xi+dx.*dt;
t=t+dt;
end