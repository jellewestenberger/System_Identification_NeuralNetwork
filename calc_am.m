%Calculate alpha_measured
function am=calc_am(u,w,cu,va)
    am=atan2(w,u)*(1+cu)+va;
end