function [clean_signal, distortedSignal, distortion_awgn, distortion_echo, distortion_cci] = add_distortion(long_signal, desired_snr_db, mixing_signals)
    % This function adds a combination of distortions (AWGN, Echo, CCI) to a given radar signal segment.
    % Inputs:
    %   long_signal: The original long radar signal from which a segment will be extracted.
    %   desired_snr_db: The desired Signal-to-Noise Ratio (in dB) for the added noise.
    %   mixing_signals: A matrix containing possible signals for Co-Channel Interference (CCI).
    % Outputs:
    %   clean_signal: The clean radar signal segment extracted from the original long signal.
    %   distortedSignal: The radar signal segment after applying the selected combination of distortions.
    %   distortion_awgn: The AWGN distortion added to the clean signal.
    %   distortion_echo: The Echo distortion added to the clean signal.
    %   distortion_cci: The CCI distortion added to the clean signal.

    % Desired length of the radar signal segment.
    desired_signal_length = 1024;

    % Define the range of possible delays for the echo effect in samples.
    delayRange = [128, 512];

    % Randomly choose a delay within the specified range for the echo effect.
    delay = randi([delayRange(1), delayRange(2)]);

    % Randomly choose a starting index for the signal segment.
    strt_indx = randi([1, length(long_signal) - (desired_signal_length + delay)]);

    % Extract the clean signal segment from the long radar signal.
    clean_signal = long_signal(strt_indx:strt_indx+desired_signal_length-1);

    % Generate the delayed signal for the echo effect.
    delayed_signal = long_signal(strt_indx+delay:strt_indx+delay+desired_signal_length-1);

    % Randomly select one of the mixing signals for CCI.
    n_mixing = size(mixing_signals, 1);
    indx = randi(n_mixing);
    mixingSignal = mixing_signals(indx, :);

    % Define all possible combinations of artifacts.
    combinations = {
        {'AWGN'}, {'Echo'}, {'CCI'}, ...        % Single artifacts
        {'AWGN', 'Echo'}, {'AWGN', 'CCI'}, {'Echo', 'CCI'}, ... % Pair of artifacts
        {'AWGN', 'Echo', 'CCI'}                % All artifacts
    };

    % Randomly select one combination of artifacts to apply.
    selectedComboIndex = randi(length(combinations));
    selectedCombo = combinations{selectedComboIndex};

    % Initialize distortion signals for recording individual distortion contributions.
    distortion_awgn = zeros(1, desired_signal_length);
    distortion_echo = zeros(1, desired_signal_length);
    distortion_cci = zeros(1, desired_signal_length);

    % If more than one artifact is selected, assign random weights for blending.
    if numel(selectedCombo) > 1
        weights = rand(1, numel(selectedCombo));
        weights = weights / sum(weights); % Normalize weights to sum to 1.
    else
        weights = 1; % Assign full weight to a single artifact.
    end

    % Total desired noise power
    signal_power = mean(abs(clean_signal).^2);
    total_noise_power = signal_power / 10^(desired_snr_db / 10);

    % Apply selected artifacts with calculated weights to the clean signal.
    for i = 1:numel(selectedCombo)
        artifactType = selectedCombo{i};

        % Allocate noise power to each distortion type
        distortion_power = total_noise_power * weights(i);


        % Apply artifacts based on the selected type and calculated SNR.
        switch artifactType
            case 'AWGN'
                % Generate AWGN
                noise = (randn(1, desired_signal_length) + 1i * randn(1, desired_signal_length));
                distortion_awgn = sqrt(distortion_power / mean(abs(noise).^2)) * noise;
            case 'Echo'
                % This step adjusts the amplitude of the echo to achieve the desired power level
                distortion_echo = sqrt(distortion_power / mean(abs(delayed_signal).^2)) * delayed_signal;
            case 'CCI'
                % Normalize mixingSignal to have the allocated power
                distortion_cci = sqrt(distortion_power / mean(abs(mixingSignal).^2)) * mixingSignal;
        end
    end
    
    distortedSignal = clean_signal + distortion_awgn + distortion_echo + distortion_cci;

end