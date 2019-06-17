%% read
addpath('matlab_functions');
eventsFile = strcat('F:\Winston\data\davis240\shapes_rotation', '\events.txt');
[t, x, y, p] = textread(eventsFile, '%f %d %d %d');
%% process
[pd, nd, duration] = binEventsTYXP(t, y, x, p);

%% plot
% # of events
total_events = length(p)
pos_events = sum(p==1)
neg_events = sum(p==0)
total_frames = length(duration)
x_axis = 1:length(pd);
pos_c = [63, 169, 245]/255.0;
neg_c = [212, 20, 90]/255.0;
axis_flag = 1;
figure(1);
stem(x_axis, pd+nd, 'color', pos_c, 'Marker', 'none');hold on
stem(x_axis, nd, 'color', neg_c, 'Marker', 'none');
if axis_flag == 1
    axis([1,length(duration), 0, 4.5])
end
xlabel('Binned event frames according to Binning 1');
ylabel('Event density (% per frame)');
legend('Positive', 'Negative');
hold off
figure(2);
stem(x_axis, (pd+nd)./duration/1e3, 'color', pos_c, 'Marker', 'none');hold on
stem(x_axis, nd./duration/1e3, 'color', neg_c, 'Marker', 'none');
xlabel('Binned event frames according to Binning 1');
ylabel('Event speed (% per frame per ms)');
if axis_flag == 1
    axis([1,length(duration), 0, 2.5])
end
legend('Positive', 'Negative');
hold off
figure(3);
plot(x_axis, duration*1e3, 'k.'); 
if axis_flag == 1
    axis([1,length(duration), 0, 35])
end
xlabel('Binned event frames according to Binning 1');
ylabel('Time duration of binned event frames t_{evf} (ms)');