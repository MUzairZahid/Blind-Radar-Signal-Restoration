clear all
addpath(genpath("\tftb-0.2")); % Modify the pathname in your pc
addpath 'waveform-types'
load('mixing_signals.mat', 'mixing_signals')
%% initial parameters configurations
fs = 100e6; % sample frequency
A = 1;      % amplitude
waveforms = {'LFM','Costas','BPSK','Frank','P1','P2','P3','P4','T1','T2','T3','T4'};% 12 LPI waveform codes
% datasetCWD = 'E:\EMD\7.4-2\train2';
% for i = 1 : length(waveforms)
%     % create the folders for dataset storage
%     mkdir(fullfile(datasetCWD,waveforms{i}));
% end

fps = 400;% the number of signal per SNR per waveform codes 400 for training. 150 for test
g=kaiser(63,0.5);
h=kaiser(63,0.5);
imgSize = 112;
N_long = 1024*2; % Signal Length that is fixed in or case.
N = 1024; % Signal Length that is fixed in or case.
SNR = -14 : 2 : 10;% snr range

total_signals = fps*length(waveforms)*length(SNR);
X_clean_signals_all = zeros(total_signals,2,1024);
X_distortion_awgn_all  = zeros(total_signals,2,1024);
X_distortion_echo_all  = zeros(total_signals,2,1024);
X_distortion_cci_all  = zeros(total_signals,2,1024);

X_distorted_signals_all = zeros(total_signals,2,1024);
Y_label_all = zeros(total_signals,1);
Y_SNR_all = zeros(total_signals,1);

%%
ik = 1;
for n = 1:length(SNR)
    % Get the random SNR value.
    minSNR = min(SNR);
    maxSNR = max(SNR);
    desired_snr_db = minSNR + (maxSNR - minSNR) * rand;

    disp(['SNR = ',sprintf('%+02d',SNR(n))])
    
%     %datasetCWD = ['E:\EMD\7.4-2\test2\',num2str(SNR(n)),'db\'];
%     datasetCWD = ['CWD\',num2str(SNR(n)),'db'];
%     for i = 1 : length(waveforms)
%         %create the folders for dataset storage
%         mkdir(fullfile(datasetCWD,waveforms{i}));
%     end
%     
    for K = 1 : length(waveforms)
        waveform = waveforms{K};
        switch waveform
            case 'LFM'
                disp(['Generating ',waveform, ' waveform ...']);
                
                % Define parameters
                fc = linspace(fs/6,fs/5,fps);
                fc = fc(randperm(fps));     % Randomize carrier frequencies
                B = linspace(fs/20, fs/16, fps);
                B = B(randperm(fps));       % Randomize bandwidths
                %N = linspace(1024,1920,fps);
                %N = round(N(randperm(fps)));% Randomize signal lengths
                sweepDirections = {'Down','Up'};
%                 waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    % Generate LFM waveform
                    wav = type_LFM(N_long,fs,A,fc(idx),B(idx),sweepDirections{randi(2)});

                    desired_snr_db = minSNR + (maxSNR - minSNR) * rand;

                    [clean_signal, distorted_signal,...
                        distortion_awgn, distortion_echo, distortion_cci] = ...
                        add_distortion(wav, desired_snr_db, mixing_signals);

        
                    % Clean Signals. 
                    X_clean_signals_all(ik,1,:) = real(clean_signal);
                    X_clean_signals_all(ik,2,:) = imag(clean_signal);
        
                    % Distorted Signals. 
                    X_distorted_signals_all(ik,1,:) = real(distorted_signal);
                    X_distorted_signals_all(ik,2,:) = imag(distorted_signal);
        
                    
                    X_distortion_awgn_all(ik,1,:) = real(distortion_awgn);
                    X_distortion_awgn_all(ik,2,:) = imag(distortion_awgn);
        
                    X_distortion_echo_all(ik,1,:) = real(distortion_echo);
                    X_distortion_echo_all(ik,2,:) = imag(distortion_echo);
        
                    X_distortion_cci_all(ik,1,:) = real(distortion_cci);
                    X_distortion_cci_all(ik,2,:) = imag(distortion_cci);
 
                    Y_label_all(ik) = K; %class label
                    Y_SNR_all(ik) = desired_snr_db;
                    ik = ik+1;


                    if (0)
                        plot_signals_blind(clean_signal, distorted_signal, distortion_awgn, distortion_echo, distortion_cci, desired_snr_db)
                        disp('plot')
        
                    end

                end
                
            case  'Costas'
                disp(['Generating ',waveform, ' waveform ...']);
                % Define parameters
                Lc = [3,4,5,6];
                fcmin = linspace(fs/30,fs/24,fps);
                fcmin=fcmin(randperm(fps)); % Randomize carrier frequencies
                %N = linspace(512,1920,fps);
                %N=round(N(randperm(fps)));  % Randomize signal lengths
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    NumHop = randperm(Lc(randi(3)));
                    % Generate waveform
                    wav = type_Costas(N_long, fs, A, fcmin(idx), NumHop);

                    if length(wav) ~= N_long
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N_long); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    desired_snr_db = minSNR + (maxSNR - minSNR) * rand;

                    [clean_signal, distorted_signal,...
                        distortion_awgn, distortion_echo, distortion_cci] = ...
                        add_distortion(wav, desired_snr_db, mixing_signals);

        
                    % Clean Signals. 
                    X_clean_signals_all(ik,1,:) = real(clean_signal);
                    X_clean_signals_all(ik,2,:) = imag(clean_signal);
        
                    % Distorted Signals. 
                    X_distorted_signals_all(ik,1,:) = real(distorted_signal);
                    X_distorted_signals_all(ik,2,:) = imag(distorted_signal);
        
                    
                    X_distortion_awgn_all(ik,1,:) = real(distortion_awgn);
                    X_distortion_awgn_all(ik,2,:) = imag(distortion_awgn);
        
                    X_distortion_echo_all(ik,1,:) = real(distortion_echo);
                    X_distortion_echo_all(ik,2,:) = imag(distortion_echo);
        
                    X_distortion_cci_all(ik,1,:) = real(distortion_cci);
                    X_distortion_cci_all(ik,2,:) = imag(distortion_cci);
 
                    Y_label_all(ik) = K; %class label
                    Y_SNR_all(ik) = desired_snr_db;
                    ik = ik+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
                
            case 'BPSK'
                disp(['Generating ',waveform, ' waveform ...']);
                Lc = [7,11,13];
                fc = linspace(fs/13,fs/10,fps);
                fc = fc(randperm(fps));
                Ncc = 20:24;
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    Bar = Lc(randi(3));
                    if Bar == 7
                        phaseCode = [0 0 0 1 1 0 1]*pi;
                    elseif Bar == 11
                        phaseCode = [0 0 0 1 1 1 0 1 1 0 1]*pi;
                    elseif Bar == 13
                        phaseCode = [0 0 0 0 0 1 1 0 0 1 0 1 0]*pi;
                    end
                    wav = type_Barker(Ncc, fs, A, fc(idx), phaseCode);
                    L = length(wav);
                    if length(wav) ~= N_long
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N_long); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    desired_snr_db = minSNR + (maxSNR - minSNR) * rand;

                    [clean_signal, distorted_signal,...
                        distortion_awgn, distortion_echo, distortion_cci] = ...
                        add_distortion(wav, desired_snr_db, mixing_signals);

        
                    % Clean Signals. 
                    X_clean_signals_all(ik,1,:) = real(clean_signal);
                    X_clean_signals_all(ik,2,:) = imag(clean_signal);
        
                    % Distorted Signals. 
                    X_distorted_signals_all(ik,1,:) = real(distorted_signal);
                    X_distorted_signals_all(ik,2,:) = imag(distorted_signal);
        
                    
                    X_distortion_awgn_all(ik,1,:) = real(distortion_awgn);
                    X_distortion_awgn_all(ik,2,:) = imag(distortion_awgn);
        
                    X_distortion_echo_all(ik,1,:) = real(distortion_echo);
                    X_distortion_echo_all(ik,2,:) = imag(distortion_echo);
        
                    X_distortion_cci_all(ik,1,:) = real(distortion_cci);
                    X_distortion_cci_all(ik,2,:) = imag(distortion_cci);
 
                    Y_label_all(ik) = K; %class label
                    Y_SNR_all(ik) = desired_snr_db;
                    ik = ik+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'Frank'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc = fc(randperm(fps));     % Randomize carrier frequencies
                Ncc = [3,4,5];              % Cycles per phase code
                M = [6, 7, 8];              % Number of frequency steps
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                snr_1=zeros(1,fps);
                for idx = 1:fps
                    % Generate waveform
                    wav = type_Frank(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));
                    L = length(wav);
                    if length(wav) ~= N_long
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N_long); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    desired_snr_db = minSNR + (maxSNR - minSNR) * rand;

                    [clean_signal, distorted_signal,...
                        distortion_awgn, distortion_echo, distortion_cci] = ...
                        add_distortion(wav, desired_snr_db, mixing_signals);

        
                    % Clean Signals. 
                    X_clean_signals_all(ik,1,:) = real(clean_signal);
                    X_clean_signals_all(ik,2,:) = imag(clean_signal);
        
                    % Distorted Signals. 
                    X_distorted_signals_all(ik,1,:) = real(distorted_signal);
                    X_distorted_signals_all(ik,2,:) = imag(distorted_signal);
        
                    
                    X_distortion_awgn_all(ik,1,:) = real(distortion_awgn);
                    X_distortion_awgn_all(ik,2,:) = imag(distortion_awgn);
        
                    X_distortion_echo_all(ik,1,:) = real(distortion_echo);
                    X_distortion_echo_all(ik,2,:) = imag(distortion_echo);
        
                    X_distortion_cci_all(ik,1,:) = real(distortion_cci);
                    X_distortion_cci_all(ik,2,:) = imag(distortion_cci);
 
                    Y_label_all(ik) = K; %class label
                    Y_SNR_all(ik) = desired_snr_db;
                    ik = ik+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'P1'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));   % Randomize carrier frequencies
                Ncc = [3,4,5];          % Cycles per phase code
                M = [6, 7, 8];          % Number of frequency steps
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    wav = type_P1(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));
                    L = length(wav);
                    if length(wav) ~= N_long
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N_long); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    desired_snr_db = minSNR + (maxSNR - minSNR) * rand;

                    [clean_signal, distorted_signal,...
                        distortion_awgn, distortion_echo, distortion_cci] = ...
                        add_distortion(wav, desired_snr_db, mixing_signals);

        
                    % Clean Signals. 
                    X_clean_signals_all(ik,1,:) = real(clean_signal);
                    X_clean_signals_all(ik,2,:) = imag(clean_signal);
        
                    % Distorted Signals. 
                    X_distorted_signals_all(ik,1,:) = real(distorted_signal);
                    X_distorted_signals_all(ik,2,:) = imag(distorted_signal);
        
                    
                    X_distortion_awgn_all(ik,1,:) = real(distortion_awgn);
                    X_distortion_awgn_all(ik,2,:) = imag(distortion_awgn);
        
                    X_distortion_echo_all(ik,1,:) = real(distortion_echo);
                    X_distortion_echo_all(ik,2,:) = imag(distortion_echo);
        
                    X_distortion_cci_all(ik,1,:) = real(distortion_cci);
                    X_distortion_cci_all(ik,2,:) = imag(distortion_cci);
 
                    Y_label_all(ik) = K; %class label
                    Y_SNR_all(ik) = desired_snr_db;
                    ik = ik+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'P2'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));   % Randomize carrier frequencies
                Ncc = [3,4,5];          % Cycles per phase code
                M = [6, 8];             % Number of frequency steps
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    % Generate waveform
                    wav = type_P2(Ncc(randi(3)), fs, A, fc(idx), M(randi(2)));
                    L = length(wav);
                    if length(wav) ~= N_long
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N_long); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    desired_snr_db = minSNR + (maxSNR - minSNR) * rand;

                    [clean_signal, distorted_signal,...
                        distortion_awgn, distortion_echo, distortion_cci] = ...
                        add_distortion(wav, desired_snr_db, mixing_signals);

        
                    % Clean Signals. 
                    X_clean_signals_all(ik,1,:) = real(clean_signal);
                    X_clean_signals_all(ik,2,:) = imag(clean_signal);
        
                    % Distorted Signals. 
                    X_distorted_signals_all(ik,1,:) = real(distorted_signal);
                    X_distorted_signals_all(ik,2,:) = imag(distorted_signal);
        
                    
                    X_distortion_awgn_all(ik,1,:) = real(distortion_awgn);
                    X_distortion_awgn_all(ik,2,:) = imag(distortion_awgn);
        
                    X_distortion_echo_all(ik,1,:) = real(distortion_echo);
                    X_distortion_echo_all(ik,2,:) = imag(distortion_echo);
        
                    X_distortion_cci_all(ik,1,:) = real(distortion_cci);
                    X_distortion_cci_all(ik,2,:) = imag(distortion_cci);
 
                    Y_label_all(ik) = K; %class label
                    Y_SNR_all(ik) = desired_snr_db;
                    ik = ik+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'P3'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));       % Randomize carrier frequencies
                Ncc = [3,4,5];              % Cycles per phase code
                p = [36, 49, 64];           % Number of subcodes
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    % Generate waveform
                    wav = type_P3(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));
                    L = length(wav);
                    if length(wav) ~= N_long
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N_long); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    desired_snr_db = minSNR + (maxSNR - minSNR) * rand;

                    [clean_signal, distorted_signal,...
                        distortion_awgn, distortion_echo, distortion_cci] = ...
                        add_distortion(wav, desired_snr_db, mixing_signals);

        
                    % Clean Signals. 
                    X_clean_signals_all(ik,1,:) = real(clean_signal);
                    X_clean_signals_all(ik,2,:) = imag(clean_signal);
        
                    % Distorted Signals. 
                    X_distorted_signals_all(ik,1,:) = real(distorted_signal);
                    X_distorted_signals_all(ik,2,:) = imag(distorted_signal);
        
                    
                    X_distortion_awgn_all(ik,1,:) = real(distortion_awgn);
                    X_distortion_awgn_all(ik,2,:) = imag(distortion_awgn);
        
                    X_distortion_echo_all(ik,1,:) = real(distortion_echo);
                    X_distortion_echo_all(ik,2,:) = imag(distortion_echo);
        
                    X_distortion_cci_all(ik,1,:) = real(distortion_cci);
                    X_distortion_cci_all(ik,2,:) = imag(distortion_cci);
 
                    Y_label_all(ik) = K; %class label
                    Y_SNR_all(ik) = desired_snr_db;
                    ik = ik+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'P4'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));       % Randomize carrier frequencies
                Ncc = [3,4,5];              % Cycles per phase code
                p = [36, 49, 64];           % Number of subcodes
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    % Generate waveform
                    wav = type_P4(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));
                    L = length(wav);
                    if length(wav) ~= N_long
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N_long); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    desired_snr_db = minSNR + (maxSNR - minSNR) * rand;

                    [clean_signal, distorted_signal,...
                        distortion_awgn, distortion_echo, distortion_cci] = ...
                        add_distortion(wav, desired_snr_db, mixing_signals);

        
                    % Clean Signals. 
                    X_clean_signals_all(ik,1,:) = real(clean_signal);
                    X_clean_signals_all(ik,2,:) = imag(clean_signal);
        
                    % Distorted Signals. 
                    X_distorted_signals_all(ik,1,:) = real(distorted_signal);
                    X_distorted_signals_all(ik,2,:) = imag(distorted_signal);
        
                    
                    X_distortion_awgn_all(ik,1,:) = real(distortion_awgn);
                    X_distortion_awgn_all(ik,2,:) = imag(distortion_awgn);
        
                    X_distortion_echo_all(ik,1,:) = real(distortion_echo);
                    X_distortion_echo_all(ik,2,:) = imag(distortion_echo);
        
                    X_distortion_cci_all(ik,1,:) = real(distortion_cci);
                    X_distortion_cci_all(ik,2,:) = imag(distortion_cci);
 
                    Y_label_all(ik) = K; %class label
                    Y_SNR_all(ik) = desired_snr_db;
                    ik = ik+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'T1'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));   % Randomize carrier frequencies
                Ng = [4,5,6];           % Number of segments
                Nps = 2;
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    % Generate waveform
                    wav = type_T1(fs, A, fc(idx),Nps,Ng(randi(3)));
                    L = length(wav);
                    if length(wav) ~= N_long
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N_long); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    desired_snr_db = minSNR + (maxSNR - minSNR) * rand;

                    [clean_signal, distorted_signal,...
                        distortion_awgn, distortion_echo, distortion_cci] = ...
                        add_distortion(wav, desired_snr_db, mixing_signals);

        
                    % Clean Signals. 
                    X_clean_signals_all(ik,1,:) = real(clean_signal);
                    X_clean_signals_all(ik,2,:) = imag(clean_signal);
        
                    % Distorted Signals. 
                    X_distorted_signals_all(ik,1,:) = real(distorted_signal);
                    X_distorted_signals_all(ik,2,:) = imag(distorted_signal);
        
                    
                    X_distortion_awgn_all(ik,1,:) = real(distortion_awgn);
                    X_distortion_awgn_all(ik,2,:) = imag(distortion_awgn);
        
                    X_distortion_echo_all(ik,1,:) = real(distortion_echo);
                    X_distortion_echo_all(ik,2,:) = imag(distortion_echo);
        
                    X_distortion_cci_all(ik,1,:) = real(distortion_cci);
                    X_distortion_cci_all(ik,2,:) = imag(distortion_cci);
 
                    Y_label_all(ik) = K; %class label
                    Y_SNR_all(ik) = desired_snr_db;
                    ik = ik+1;
                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
            case 'T2'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));   % Randomize carrier frequencies
                Ng = [4,5,6];           % Number of segments
                Nps = 2;
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    % Generate waveform
                    wav = type_T2(fs, A, fc(idx),Nps,Ng(randi(3)));
                    L = length(wav);
                    if length(wav) ~= N_long
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N_long); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    desired_snr_db = minSNR + (maxSNR - minSNR) * rand;

                    [clean_signal, distorted_signal,...
                        distortion_awgn, distortion_echo, distortion_cci] = ...
                        add_distortion(wav, desired_snr_db, mixing_signals);

        
                    % Clean Signals. 
                    X_clean_signals_all(ik,1,:) = real(clean_signal);
                    X_clean_signals_all(ik,2,:) = imag(clean_signal);
        
                    % Distorted Signals. 
                    X_distorted_signals_all(ik,1,:) = real(distorted_signal);
                    X_distorted_signals_all(ik,2,:) = imag(distorted_signal);
        
                    
                    X_distortion_awgn_all(ik,1,:) = real(distortion_awgn);
                    X_distortion_awgn_all(ik,2,:) = imag(distortion_awgn);
        
                    X_distortion_echo_all(ik,1,:) = real(distortion_echo);
                    X_distortion_echo_all(ik,2,:) = imag(distortion_echo);
        
                    X_distortion_cci_all(ik,1,:) = real(distortion_cci);
                    X_distortion_cci_all(ik,2,:) = imag(distortion_cci);
 
                    Y_label_all(ik) = K; %class label
                    Y_SNR_all(ik) = desired_snr_db;
                    ik = ik+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'T3'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/13,fs/10,fps);
                fc=fc(randperm(fps));   % Randomize carrier frequencies
                B = linspace(fs/20,fs/10,fps);
                B = B(randperm(fps));   % Randomize bandwidths
                Nps = 2;
                %N = linspace(512,1920,fps);
                %N=round(N(randperm(fps)));% Randomize signal lengths
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    % Generate waveform
                    wav = type_T3(N_long, fs, A, fc(idx), Nps,B(idx));
                    if length(wav) ~= N_long
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N_long); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    desired_snr_db = minSNR + (maxSNR - minSNR) * rand;

                    [clean_signal, distorted_signal,...
                        distortion_awgn, distortion_echo, distortion_cci] = ...
                        add_distortion(wav, desired_snr_db, mixing_signals);

        
                    % Clean Signals. 
                    X_clean_signals_all(ik,1,:) = real(clean_signal);
                    X_clean_signals_all(ik,2,:) = imag(clean_signal);
        
                    % Distorted Signals. 
                    X_distorted_signals_all(ik,1,:) = real(distorted_signal);
                    X_distorted_signals_all(ik,2,:) = imag(distorted_signal);
        
                    
                    X_distortion_awgn_all(ik,1,:) = real(distortion_awgn);
                    X_distortion_awgn_all(ik,2,:) = imag(distortion_awgn);
        
                    X_distortion_echo_all(ik,1,:) = real(distortion_echo);
                    X_distortion_echo_all(ik,2,:) = imag(distortion_echo);
        
                    X_distortion_cci_all(ik,1,:) = real(distortion_cci);
                    X_distortion_cci_all(ik,2,:) = imag(distortion_cci);
 
                    Y_label_all(ik) = K; %class label
                    Y_SNR_all(ik) = desired_snr_db;
                    ik = ik+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'T4'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/13,fs/10,fps);
                fc=fc(randperm(fps));       % Randomize carrier frequencies
                B = linspace(fs/20,fs/10,fps);
                B = B(randperm(fps));       % Randomize bandwidths
                Nps = 2;
                %N = linspace(512,1920,fps);
                %N=round(N(randperm(fps)));  % Randomize signal lengths
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    % Generate waveform
                    wav = type_T4(N_long, fs, A, fc(idx), Nps,B(idx));
                    if length(wav) ~= N_long
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N_long); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    desired_snr_db = minSNR + (maxSNR - minSNR) * rand;

                    [clean_signal, distorted_signal,...
                        distortion_awgn, distortion_echo, distortion_cci] = ...
                        add_distortion(wav, desired_snr_db, mixing_signals);

        
                    % Clean Signals. 
                    X_clean_signals_all(ik,1,:) = real(clean_signal);
                    X_clean_signals_all(ik,2,:) = imag(clean_signal);
        
                    % Distorted Signals. 
                    X_distorted_signals_all(ik,1,:) = real(distorted_signal);
                    X_distorted_signals_all(ik,2,:) = imag(distorted_signal);
        
                    
                    X_distortion_awgn_all(ik,1,:) = real(distortion_awgn);
                    X_distortion_awgn_all(ik,2,:) = imag(distortion_awgn);
        
                    X_distortion_echo_all(ik,1,:) = real(distortion_echo);
                    X_distortion_echo_all(ik,2,:) = imag(distortion_echo);
        
                    X_distortion_cci_all(ik,1,:) = real(distortion_cci);
                    X_distortion_cci_all(ik,2,:) = imag(distortion_cci);
 
                    Y_label_all(ik) = K; %class label
                    Y_SNR_all(ik) = desired_snr_db;
                    ik = ik+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
            otherwise
                disp('Done!')
        end
    end
end


saveDirectory = 'Dataset_1_blind_train';

[status,msg,msgID] = mkdir(saveDirectory);

variablesToSave = {
    'X_clean_signals_all', 
    'X_distorted_signals_all', 
    'X_distortion_awgn_all',
    'X_distortion_echo_all',
    'X_distortion_cci_all',
    'Y_label_all', 
    'Y_SNR_all'
};
% Loop through each variable and save it in a separate .mat file
for i = 1:numel(variablesToSave)
    variableName = variablesToSave{i};
    filename = fullfile(saveDirectory, [variableName, '.mat']); % Create the full file path
    save(filename, variableName, '-v7.3'); % Save the variable in a .mat file
end