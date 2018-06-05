%calculate measured beta
function bm=calc_bm(u,v,w,vb)
bm=atan2(v,sqrt(u^2+w^2))+vb;
end