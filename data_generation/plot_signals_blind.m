function plot_signals_blind(clean_signal, distorted_signal, distortion_awgn, distortion_echo, distortion_cci, assigned_snr_db)
    % Plot the clean signal, distorted signal, and each type of distortion in subplots
    % Also display calculated SNR and assigned SNR on the clean signal subplot

    figure;

    % Calculate the overall SNR of the distorted signal
    snr_distorted = calculate_snr(clean_signal, distorted_signal - clean_signal);

    % Subplot 1: Clean Signal
    subplot(5, 1, 1);
    plot(real(clean_signal));
    title(sprintf('Clean Signal - Assigned SNR: %.2f dB, Calculated SNR: %.2f dB', assigned_snr_db, snr_distorted));
    ylabel('Amplitude');
    xlabel('Sample Index');

    % Subplot 2: Distorted Signal
    subplot(5, 1, 2);
    plot(real(distorted_signal));
    title('Distorted Signal');
    ylabel('Amplitude');
    xlabel('Sample Index');

    % Subplot 3: AWGN Distortion
    subplot(5, 1, 3);
    plot(real(distortion_awgn));
    title('AWGN Distortion');
    ylabel('Amplitude');
    xlabel('Sample Index');

    % Subplot 4: Echo Distortion
    subplot(5, 1, 4);
    plot(real(distortion_echo));
    title('Echo Distortion');
    ylabel('Amplitude');
    xlabel('Sample Index');

    % Subplot 5: CCI Distortion
    subplot(5, 1, 5);
    plot(real(distortion_cci));
    title('CCI Distortion');
    ylabel('Amplitude');
    xlabel('Sample Index');

    % Adjust layout
    sgtitle('Signal and Distortions Visualization'); % Super title for the entire figure
    set(gcf, 'Position', get(0, 'Screensize')); % Maximize the figure window
end
