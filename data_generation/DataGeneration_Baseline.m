clear all
addpath 'waveform-types'

%% initial parameters configurations
fs = 100e6; % sample frequency
A = 1;      % amplitude
waveforms = {'LFM','Costas','BPSK','Frank','P1','P2','P3','P4','T1','T2','T3','T4'};% 12 LPI waveform codes

% Parameters for training dataset
fps_train = 400;% Number of signals per SNR per waveform code for training
% Parameters for test dataset
fps_test = 150;% Number of signals per SNR per waveform code for testing

g = kaiser(63,0.5);
h = kaiser(63,0.5);
imgSize = 112;
N = 1024; % Signal Length that is fixed in our case.
SNR = -14 : 2 : 10;% snr range

% Calculate total signals for training and testing
total_signals_train = fps_train * length(waveforms) * length(SNR);
total_signals_test = fps_test * length(waveforms) * length(SNR);

% Initialize arrays for training data
X_clean_signals_train = zeros(total_signals_train, 2, 1024);
X_distorted_signals_train = zeros(total_signals_train, 2, 1024);
Y_label_train = zeros(total_signals_train, 1);
Y_SNR_train = zeros(total_signals_train, 1);

% Initialize arrays for test data
X_clean_signals_test = zeros(total_signals_test, 2, 1024);
X_distorted_signals_test = zeros(total_signals_test, 2, 1024);
Y_label_test = zeros(total_signals_test, 1);
Y_SNR_test = zeros(total_signals_test, 1);

%%
% Generate training data
disp('Generating training dataset...');
ik_train = 1;
for n = 1:length(SNR)
    snr=SNR(n);
    disp(['SNR = ',sprintf('%+02d',SNR(n))])
     
    for K = 1 : length(waveforms)
        waveform = waveforms{K};
        switch waveform
            case 'LFM'
                disp(['Generating ',waveform, ' waveform ...']);
                
                % Define parameters
                fc = linspace(fs/6,fs/5,fps_train);
                fc = fc(randperm(fps_train));     % Randomize carrier frequencies
                B = linspace(fs/20, fs/16, fps_train);
                B = B(randperm(fps_train));       % Randomize bandwidths
                %N = linspace(1024,1920,fps);
                %N = round(N(randperm(fps)));% Randomize signal lengths
                sweepDirections = {'Down','Up'};
%                 waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_train
                    % Generate LFM waveform
                    %wav = type_LFM(N(idx),fs,A,fc(idx),B(idx),sweepDirections{randi(2)});
                    wav = type_LFM(N,fs,A,fc(idx),B(idx),sweepDirections{randi(2)});
                    wav = wav';
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');
 

                    % Clean Signals. 
                    X_clean_signals_train(ik_train,1,:) = real(wav);
                    X_clean_signals_train(ik_train,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_train(ik_train,1,:) = real(wav_noisy);
                    X_distorted_signals_train(ik_train,2,:) = imag(wav_noisy);
                    Y_label_train(ik_train) = K; %class label
                    Y_SNR_train(ik_train) = SNR(n);
                    ik_train = ik_train+1;


                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end

                end
                
            case  'Costas'
                disp(['Generating ',waveform, ' waveform ...']);
                % Define parameters
                Lc = [3,4,5,6];
                fcmin = linspace(fs/30,fs/24,fps_train);
                fcmin=fcmin(randperm(fps_train)); % Randomize carrier frequencies
                %N = linspace(512,1920,fps);
                %N=round(N(randperm(fps)));  % Randomize signal lengths
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_train
                    NumHop = randperm(Lc(randi(3)));
                    % Generate waveform
                    wav = type_Costas(N, fs, A, fcmin(idx), NumHop);
                    wav = wav';

                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_train(ik_train,1,:) = real(wav);
                    X_clean_signals_train(ik_train,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_train(ik_train,1,:) = real(wav_noisy);
                    X_distorted_signals_train(ik_train,2,:) = imag(wav_noisy);
                    Y_label_train(ik_train) = K; %class label
                    Y_SNR_train(ik_train) = SNR(n);
                    ik_train = ik_train+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
                
            case 'BPSK'
                disp(['Generating ',waveform, ' waveform ...']);
                Lc = [7,11,13];
                fc = linspace(fs/13,fs/10,fps_train);
                fc = fc(randperm(fps_train));
                Ncc = 20:24;
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_train
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
                    wav = wav';
                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_train(ik_train,1,:) = real(wav);
                    X_clean_signals_train(ik_train,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_train(ik_train,1,:) = real(wav_noisy);
                    X_distorted_signals_train(ik_train,2,:) = imag(wav_noisy);
                    Y_label_train(ik_train) = K; %class label
                    Y_SNR_train(ik_train) = SNR(n);
                    ik_train = ik_train+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'Frank'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps_train);
                fc = fc(randperm(fps_train));     % Randomize carrier frequencies
                Ncc = [3,4,5];              % Cycles per phase code
                M = [6, 7, 8];              % Number of frequency steps
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                snr_1=zeros(1,fps_train);
                for idx = 1:fps_train
                    % Generate waveform
                    wav = type_Frank(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));
                    L = length(wav);
                    wav = wav';
                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_train(ik_train,1,:) = real(wav);
                    X_clean_signals_train(ik_train,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_train(ik_train,1,:) = real(wav_noisy);
                    X_distorted_signals_train(ik_train,2,:) = imag(wav_noisy);
                    Y_label_train(ik_train) = K; %class label
                    Y_SNR_train(ik_train) = SNR(n);
                    ik_train = ik_train+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'P1'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps_train);
                fc=fc(randperm(fps_train));   % Randomize carrier frequencies
                Ncc = [3,4,5];          % Cycles per phase code
                M = [6, 7, 8];          % Number of frequency steps
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_train
                    wav = type_P1(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));
                    L = length(wav);
                    wav = wav';
                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_train(ik_train,1,:) = real(wav);
                    X_clean_signals_train(ik_train,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_train(ik_train,1,:) = real(wav_noisy);
                    X_distorted_signals_train(ik_train,2,:) = imag(wav_noisy);
                    Y_label_train(ik_train) = K; %class label
                    Y_SNR_train(ik_train) = SNR(n);
                    ik_train = ik_train+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'P2'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps_train);
                fc=fc(randperm(fps_train));   % Randomize carrier frequencies
                Ncc = [3,4,5];          % Cycles per phase code
                M = [6, 8];             % Number of frequency steps
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_train
                    % Generate waveform
                    wav = type_P2(Ncc(randi(3)), fs, A, fc(idx), M(randi(2)));
                    L = length(wav);
                    wav = wav';
                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_train(ik_train,1,:) = real(wav);
                    X_clean_signals_train(ik_train,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_train(ik_train,1,:) = real(wav_noisy);
                    X_distorted_signals_train(ik_train,2,:) = imag(wav_noisy);
                    Y_label_train(ik_train) = K; %class label
                    Y_SNR_train(ik_train) = SNR(n);
                    ik_train = ik_train+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'P3'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps_train);
                fc=fc(randperm(fps_train));       % Randomize carrier frequencies
                Ncc = [3,4,5];              % Cycles per phase code
                p = [36, 49, 64];           % Number of subcodes
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_train
                    % Generate waveform
                    wav = type_P3(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));
                    L = length(wav);
                    wav = wav';
                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_train(ik_train,1,:) = real(wav);
                    X_clean_signals_train(ik_train,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_train(ik_train,1,:) = real(wav_noisy);
                    X_distorted_signals_train(ik_train,2,:) = imag(wav_noisy);
                    Y_label_train(ik_train) = K; %class label
                    Y_SNR_train(ik_train) = SNR(n);
                    ik_train = ik_train+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'P4'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps_train);
                fc=fc(randperm(fps_train));       % Randomize carrier frequencies
                Ncc = [3,4,5];              % Cycles per phase code
                p = [36, 49, 64];           % Number of subcodes
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_train
                    % Generate waveform
                    wav = type_P4(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));
                    L = length(wav);
                    wav = wav';
                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_train(ik_train,1,:) = real(wav);
                    X_clean_signals_train(ik_train,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_train(ik_train,1,:) = real(wav_noisy);
                    X_distorted_signals_train(ik_train,2,:) = imag(wav_noisy);
                    Y_label_train(ik_train) = K; %class label
                    Y_SNR_train(ik_train) = SNR(n);
                    ik_train = ik_train+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'T1'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps_train);
                fc=fc(randperm(fps_train));   % Randomize carrier frequencies
                Ng = [4,5,6];           % Number of segments
                Nps = 2;
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_train
                    % Generate waveform
                    wav = type_T1(fs, A, fc(idx),Nps,Ng(randi(3)));
                    L = length(wav);
                    wav = wav';
                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_train(ik_train,1,:) = real(wav);
                    X_clean_signals_train(ik_train,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_train(ik_train,1,:) = real(wav_noisy);
                    X_distorted_signals_train(ik_train,2,:) = imag(wav_noisy);
                    Y_label_train(ik_train) = K; %class label
                    Y_SNR_train(ik_train) = SNR(n);
                    ik_train = ik_train+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
            case 'T2'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps_train);
                fc=fc(randperm(fps_train));   % Randomize carrier frequencies
                Ng = [4,5,6];           % Number of segments
                Nps = 2;
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_train
                    % Generate waveform
                    wav = type_T2(fs, A, fc(idx),Nps,Ng(randi(3)));
                    L = length(wav);
                    wav = wav';
                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_train(ik_train,1,:) = real(wav);
                    X_clean_signals_train(ik_train,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_train(ik_train,1,:) = real(wav_noisy);
                    X_distorted_signals_train(ik_train,2,:) = imag(wav_noisy);
                    Y_label_train(ik_train) = K; %class label
                    Y_SNR_train(ik_train) = SNR(n);
                    ik_train = ik_train+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'T3'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/13,fs/10,fps_train);
                fc=fc(randperm(fps_train));   % Randomize carrier frequencies
                B = linspace(fs/20,fs/10,fps_train);
                B = B(randperm(fps_train));   % Randomize bandwidths
                Nps = 2;
                %N = linspace(512,1920,fps);
                %N=round(N(randperm(fps)));% Randomize signal lengths
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_train
                    % Generate waveform
                    wav = type_T3(N, fs, A, fc(idx), Nps,B(idx));
                    wav = wav';
                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_train(ik_train,1,:) = real(wav);
                    X_clean_signals_train(ik_train,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_train(ik_train,1,:) = real(wav_noisy);
                    X_distorted_signals_train(ik_train,2,:) = imag(wav_noisy);
                    Y_label_train(ik_train) = K; %class label
                    Y_SNR_train(ik_train) = SNR(n);
                    ik_train = ik_train+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'T4'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/13,fs/10,fps_train);
                fc=fc(randperm(fps_train));       % Randomize carrier frequencies
                B = linspace(fs/20,fs/10,fps_train);
                B = B(randperm(fps_train));       % Randomize bandwidths
                Nps = 2;
                %N = linspace(512,1920,fps);
                %N=round(N(randperm(fps)));  % Randomize signal lengths
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_train
                    % Generate waveform
                    wav = type_T4(N, fs, A, fc(idx), Nps,B(idx));
                    wav = wav';
                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_train(ik_train,1,:) = real(wav);
                    X_clean_signals_train(ik_train,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_train(ik_train,1,:) = real(wav_noisy);
                    X_distorted_signals_train(ik_train,2,:) = imag(wav_noisy);
                    Y_label_train(ik_train) = K; %class label
                    Y_SNR_train(ik_train) = SNR(n);
                    ik_train = ik_train+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
            otherwise
                disp('Unknown waveform type!');
        end
    end
end

%%
% Generate test data
disp('Generating test dataset...');
ik_test = 1;
for n = 1:length(SNR)
    snr=SNR(n);
    disp(['SNR = ',sprintf('%+02d',SNR(n))])
     
    for K = 1 : length(waveforms)
        waveform = waveforms{K};
        switch waveform
            case 'LFM'
                disp(['Generating ',waveform, ' waveform ...']);
                
                % Define parameters
                fc = linspace(fs/6,fs/5,fps_test);
                fc = fc(randperm(fps_test));     % Randomize carrier frequencies
                B = linspace(fs/20, fs/16, fps_test);
                B = B(randperm(fps_test));       % Randomize bandwidths
                %N = linspace(1024,1920,fps);
                %N = round(N(randperm(fps)));% Randomize signal lengths
                sweepDirections = {'Down','Up'};
%                 waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_test
                    % Generate LFM waveform
                    %wav = type_LFM(N(idx),fs,A,fc(idx),B(idx),sweepDirections{randi(2)});
                    wav = type_LFM(N,fs,A,fc(idx),B(idx),sweepDirections{randi(2)});
                    wav = wav';
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');
 

                    % Clean Signals. 
                    X_clean_signals_test(ik_test,1,:) = real(wav);
                    X_clean_signals_test(ik_test,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_test(ik_test,1,:) = real(wav_noisy);
                    X_distorted_signals_test(ik_test,2,:) = imag(wav_noisy);
                    Y_label_test(ik_test) = K; %class label
                    Y_SNR_test(ik_test) = SNR(n);
                    ik_test = ik_test+1;


                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end

                end
                
            case  'Costas'
                disp(['Generating ',waveform, ' waveform ...']);
                % Define parameters
                Lc = [3,4,5,6];
                fcmin = linspace(fs/30,fs/24,fps_test);
                fcmin=fcmin(randperm(fps_test)); % Randomize carrier frequencies
                %N = linspace(512,1920,fps);
                %N=round(N(randperm(fps)));  % Randomize signal lengths
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_test
                    NumHop = randperm(Lc(randi(3)));
                    % Generate waveform
                    wav = type_Costas(N, fs, A, fcmin(idx), NumHop);
                    wav = wav';

                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_test(ik_test,1,:) = real(wav);
                    X_clean_signals_test(ik_test,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_test(ik_test,1,:) = real(wav_noisy);
                    X_distorted_signals_test(ik_test,2,:) = imag(wav_noisy);
                    Y_label_test(ik_test) = K; %class label
                    Y_SNR_test(ik_test) = SNR(n);
                    ik_test = ik_test+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
                
            case 'BPSK'
                disp(['Generating ',waveform, ' waveform ...']);
                Lc = [7,11,13];
                fc = linspace(fs/13,fs/10,fps_test);
                fc = fc(randperm(fps_test));
                Ncc = 20:24;
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_test
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
                    wav = wav';
                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_test(ik_test,1,:) = real(wav);
                    X_clean_signals_test(ik_test,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_test(ik_test,1,:) = real(wav_noisy);
                    X_distorted_signals_test(ik_test,2,:) = imag(wav_noisy);
                    Y_label_test(ik_test) = K; %class label
                    Y_SNR_test(ik_test) = SNR(n);
                    ik_test = ik_test+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'Frank'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps_test);
                fc = fc(randperm(fps_test));     % Randomize carrier frequencies
                Ncc = [3,4,5];              % Cycles per phase code
                M = [6, 7, 8];              % Number of frequency steps
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                snr_1=zeros(1,fps_test);
                for idx = 1:fps_test
                    % Generate waveform
                    wav = type_Frank(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));
                    L = length(wav);
                    wav = wav';
                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_test(ik_test,1,:) = real(wav);
                    X_clean_signals_test(ik_test,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_test(ik_test,1,:) = real(wav_noisy);
                    X_distorted_signals_test(ik_test,2,:) = imag(wav_noisy);
                    Y_label_test(ik_test) = K; %class label
                    Y_SNR_test(ik_test) = SNR(n);
                    ik_test = ik_test+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'P1'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps_test);
                fc=fc(randperm(fps_test));   % Randomize carrier frequencies
                Ncc = [3,4,5];          % Cycles per phase code
                M = [6, 7, 8];          % Number of frequency steps
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_test
                    wav = type_P1(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));
                    L = length(wav);
                    wav = wav';
                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_test(ik_test,1,:) = real(wav);
                    X_clean_signals_test(ik_test,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_test(ik_test,1,:) = real(wav_noisy);
                    X_distorted_signals_test(ik_test,2,:) = imag(wav_noisy);
                    Y_label_test(ik_test) = K; %class label
                    Y_SNR_test(ik_test) = SNR(n);
                    ik_test = ik_test+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'P2'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps_test);
                fc=fc(randperm(fps_test));   % Randomize carrier frequencies
                Ncc = [3,4,5];          % Cycles per phase code
                M = [6, 8];             % Number of frequency steps
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_test
                    % Generate waveform
                    wav = type_P2(Ncc(randi(3)), fs, A, fc(idx), M(randi(2)));
                    L = length(wav);
                    wav = wav';
                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_test(ik_test,1,:) = real(wav);
                    X_clean_signals_test(ik_test,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_test(ik_test,1,:) = real(wav_noisy);
                    X_distorted_signals_test(ik_test,2,:) = imag(wav_noisy);
                    Y_label_test(ik_test) = K; %class label
                    Y_SNR_test(ik_test) = SNR(n);
                    ik_test = ik_test+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'P3'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps_test);
                fc=fc(randperm(fps_test));       % Randomize carrier frequencies
                Ncc = [3,4,5];              % Cycles per phase code
                p = [36, 49, 64];           % Number of subcodes
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_test
                    % Generate waveform
                    wav = type_P3(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));
                    L = length(wav);
                    wav = wav';
                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_test(ik_test,1,:) = real(wav);
                    X_clean_signals_test(ik_test,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_test(ik_test,1,:) = real(wav_noisy);
                    X_distorted_signals_test(ik_test,2,:) = imag(wav_noisy);
                    Y_label_test(ik_test) = K; %class label
                    Y_SNR_test(ik_test) = SNR(n);
                    ik_test = ik_test+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'P4'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps_test);
                fc=fc(randperm(fps_test));       % Randomize carrier frequencies
                Ncc = [3,4,5];              % Cycles per phase code
                p = [36, 49, 64];           % Number of subcodes
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_test
                    % Generate waveform
                    wav = type_P4(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));
                    L = length(wav);
                    wav = wav';
                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_test(ik_test,1,:) = real(wav);
                    X_clean_signals_test(ik_test,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_test(ik_test,1,:) = real(wav_noisy);
                    X_distorted_signals_test(ik_test,2,:) = imag(wav_noisy);
                    Y_label_test(ik_test) = K; %class label
                    Y_SNR_test(ik_test) = SNR(n);
                    ik_test = ik_test+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'T1'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps_test);
                fc=fc(randperm(fps_test));   % Randomize carrier frequencies
                Ng = [4,5,6];           % Number of segments
                Nps = 2;
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_test
                    % Generate waveform
                    wav = type_T1(fs, A, fc(idx),Nps,Ng(randi(3)));
                    L = length(wav);
                    wav = wav';
                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_test(ik_test,1,:) = real(wav);
                    X_clean_signals_test(ik_test,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_test(ik_test,1,:) = real(wav_noisy);
                    X_distorted_signals_test(ik_test,2,:) = imag(wav_noisy);
                    Y_label_test(ik_test) = K; %class label
                    Y_SNR_test(ik_test) = SNR(n);
                    ik_test = ik_test+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
            case 'T2'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps_test);
                fc=fc(randperm(fps_test));   % Randomize carrier frequencies
                Ng = [4,5,6];           % Number of segments
                Nps = 2;
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_test
                    % Generate waveform
                    wav = type_T2(fs, A, fc(idx),Nps,Ng(randi(3)));
                    L = length(wav);
                    wav = wav';
                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_test(ik_test,1,:) = real(wav);
                    X_clean_signals_test(ik_test,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_test(ik_test,1,:) = real(wav_noisy);
                    X_distorted_signals_test(ik_test,2,:) = imag(wav_noisy);
                    Y_label_test(ik_test) = K; %class label
                    Y_SNR_test(ik_test) = SNR(n);
                    ik_test = ik_test+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'T3'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/13,fs/10,fps_test);
                fc=fc(randperm(fps_test));   % Randomize carrier frequencies
                B = linspace(fs/20,fs/10,fps_test);
                B = B(randperm(fps_test));   % Randomize bandwidths
                Nps = 2;
                %N = linspace(512,1920,fps);
                %N=round(N(randperm(fps)));% Randomize signal lengths
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_test
                    % Generate waveform
                    wav = type_T3(N, fs, A, fc(idx), Nps,B(idx));
                    wav = wav';
                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_test(ik_test,1,:) = real(wav);
                    X_clean_signals_test(ik_test,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_test(ik_test,1,:) = real(wav_noisy);
                    X_distorted_signals_test(ik_test,2,:) = imag(wav_noisy);
                    Y_label_test(ik_test) = K; %class label
                    Y_SNR_test(ik_test) = SNR(n);
                    ik_test = ik_test+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
                
            case 'T4'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/13,fs/10,fps_test);
                fc=fc(randperm(fps_test));       % Randomize carrier frequencies
                B = linspace(fs/20,fs/10,fps_test);
                B = B(randperm(fps_test));       % Randomize bandwidths
                Nps = 2;
                %N = linspace(512,1920,fps);
                %N=round(N(randperm(fps)));  % Randomize signal lengths
                %waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps_test
                    % Generate waveform
                    wav = type_T4(N, fs, A, fc(idx), Nps,B(idx));
                    wav = wav';
                    if length(wav) ~= N
                        I = real(wav);
                        Q = imag(wav);
                        disp("Interpolation")
                        t1 = linspace(1,length(I),length(I)); 
                        t2 = linspace(1,length(I),N); 
                        I = interp1(t1,I,t2);
                        Q = interp1(t1,Q,t2);
                        
                        wav = I + 1i * Q;
                    
                    end
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');

                    % Clean Signals. 
                    X_clean_signals_test(ik_test,1,:) = real(wav);
                    X_clean_signals_test(ik_test,2,:) = imag(wav);
                    % Distorted Signals. 
                    X_distorted_signals_test(ik_test,1,:) = real(wav_noisy);
                    X_distorted_signals_test(ik_test,2,:) = imag(wav_noisy);
                    Y_label_test(ik_test) = K; %class label
                    Y_SNR_test(ik_test) = SNR(n);
                    ik_test = ik_test+1;

                    % To check plots. 
                    if (0)
                        plot_signals_awgn(wav, wav_noisy, SNR(n))
                        disp('plot')
                    
                    end
                end
            otherwise
                disp('Unknown waveform type!');
        end
    end
end

%%
% Save training dataset
trainDirectory = 'Dataset_baseline_train';
[status_train, msg_train, msgID_train] = mkdir(trainDirectory);

trainVariablesToSave = {
    'X_clean_signals_train', 
    'X_distorted_signals_train', 
    'Y_label_train', 
    'Y_SNR_train'
};

% Loop through each variable and save it in a separate .mat file
for i = 1:numel(trainVariablesToSave)
    variableName = trainVariablesToSave{i};
    filename = fullfile(trainDirectory, [variableName, '.mat']); % Create the full file path
    save(filename, variableName, '-v7.3'); % Save the variable in a .mat file
    disp(['Saved ', variableName, ' to ', filename]);
end

% Save test dataset
testDirectory = 'Dataset_baseline_test';
[status_test, msg_test, msgID_test] = mkdir(testDirectory);

testVariablesToSave = {
    'X_clean_signals_test', 
    'X_distorted_signals_test', 
    'Y_label_test', 
    'Y_SNR_test'
};

% Loop through each variable and save it in a separate .mat file
for i = 1:numel(testVariablesToSave)
    variableName = testVariablesToSave{i};
    filename = fullfile(testDirectory, [variableName, '.mat']); % Create the full file path
    save(filename, variableName, '-v7.3'); % Save the variable in a .mat file
    disp(['Saved ', variableName, ' to ', filename]);
end

disp('Data generation complete!');