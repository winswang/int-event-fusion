function [pos_dense, neg_dense, evf_duration] = binEventsTYXP(t, y, x, p)
t_curr = t(1);
frame_curr = zeros(180, 240);
evf_count = 0;
for i = 1:length(t)
    ti = t(i);
    yi = y(i)+1;
    xi = x(i)+1;
    if frame_curr(yi, xi) ~= 0
        evf_duration(evf_count + 1) = ti - t_curr;
        pos_dense(evf_count + 1) = sum(sum(frame_curr>0))/180./240.*100;
        neg_dense(evf_count + 1) = sum(sum(frame_curr<0))/180./240.*100;
        t_curr = ti;
        evf_count = evf_count + 1;
        frame_curr = zeros(180, 240);
    else
        if p(i) == 1
            frame_curr(yi, xi) = 1;
        else
            frame_curr(yi, xi) = -1;
        end
    end
end