function plot_signals_awgn(clean_signal, distorted_signal, assigned_snr_db)
    % Plot the clean signal, distorted signal, and each type of distortion in subplots
    % Also display calculated SNR and assigned SNR on the clean signal subplot

    figure;

    % Calculate the overall SNR of the distorted signal
    snr_distorted = calculate_snr(clean_signal, distorted_signal - clean_signal);

    % Subplot 1: Clean Signal
    subplot(2, 1, 1);
    plot(real(clean_signal));
    title(sprintf('Clean Signal - Assigned SNR: %.2f dB, Calculated SNR: %.2f dB', assigned_snr_db, snr_distorted));
    ylabel('Amplitude');
    xlabel('Sample Index');

    % Subplot 2: Distorted Signal
    subplot(2, 1, 2);
    plot(real(distorted_signal));
    title('Distorted Signal');
    ylabel('Amplitude');
    xlabel('Sample Index');



    % Adjust layout
    sgtitle('Signal and Distortions Visualization'); % Super title for the entire figure
    set(gcf, 'Position', get(0, 'Screensize')); % Maximize the figure window
end
