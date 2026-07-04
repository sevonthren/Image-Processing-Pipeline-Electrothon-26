%% Loading datastore
folderPath = "C:\Users\kgowt\Downloads\cursed_schem";
imds = imageDatastore(folderPath, ...
    'FileExtensions', {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}, ...
    'IncludeSubfolders', false);
fprintf('Found %d images.\n', numel(imds.Files));

outputFolder = 'C:\Users\kgowt\Downloads\schem_output';
if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end

%% Processing each image
reset(imds);
idx = 1;

while hasdata(imds)
    raw = read(imds);
    [restored, spatialFiltered, logMagSpectrum, notchMask] = processImage(raw);

    % Save restored image
    [~, fname, ~] = fileparts(imds.Files{idx});
    imwrite(restored, fullfile(outputFolder, [fname '_processed.png']));

    % Generate 5-panel diagnostic figure
    plotDiagnostic(raw, spatialFiltered, logMagSpectrum, notchMask, restored, fname);

    idx = idx + 1;
end
fprintf('Done. Processed %d images.\n', idx - 1);


%  PROCESSING FUNCTION
%  Returns all intermediate stages for the diagnostic figure
function [restored, spatialFiltered, logMagSpectrum, notchMask] = processImage(img)

    % --- Convert to grayscale double [0,1] ---
    img = im2double(img);
    if size(img, 3) == 3, img = img(:,:,1); end

    [rows, cols] = size(img);
    crow = round(rows / 2);
    ccol = round(cols / 2);

    
    % STAGE 1 — Spatial filtering (median filter)
   
    spatialFiltered = medfilt2(img, [3 3], 'symmetric');

    % STAGE 2 — Frequency domain: adaptive notch filter

    F      = fft2(spatialFiltered);   % 2D FFT
    Fs     = fftshift(F);             % shift DC to centre for visualisation
    logMag = log1p(abs(Fs));          % log-compressed magnitude spectrum

    % Build DC protection zone
    [X, Y]   = meshgrid(1:cols, 1:rows);
    dcRadius = round(min(rows, cols) * 0.04);
    dcZone   = sqrt((X - ccol).^2 + (Y - crow).^2) <= dcRadius;

    % Adaptive threshold — top 0.1% of non-DC magnitudes are grid spikes
    outsideDC = logMag(~dcZone);
    threshold = quantile(outsideDC, 0.999);

    % Build notch mask: 0 = suppress, 1 = keep
    notchMask = ones(rows, cols);
    notchMask(logMag > threshold) = 0;  % zero out spike locations
    notchMask(dcZone) = 1;              % always preserve DC region

    % Apply mask in frequency domain and invert back to spatial domain
    freqCleaned = real(ifft2(ifftshift(Fs .* notchMask)));
    freqCleaned = (freqCleaned - min(freqCleaned(:))) / ...
                  (max(freqCleaned(:)) - min(freqCleaned(:)));

    
    % STAGE 3 — Post-processing: background flattening, contrast, sharpening
    
    flattened = freqCleaned;

    % Contrast stretch — clip bottom/top 1% to spread midtones
    lo      = quantile(flattened(:), 0.01);
    hi      = quantile(flattened(:), 0.99);
    stretched = (flattened - lo) / (hi - lo);
    restored = max(0, min(1, stretched));

    % Store log-magnitude spectrum for diagnostic plot
    logMagSpectrum = mat2gray(logMag);

end


%  DIAGNOSTIC FIGURE — 5-panel visual output
function plotDiagnostic(raw, spatialFiltered, logMagSpectrum, notchMask, restored, fname)

    figure('Name', fname, 'NumberTitle', 'off', ...
           'Units', 'normalized', 'Position', [0.05 0.1 0.9 0.5]);

    % Panel 1 — Original corrupted image
    subplot(1, 5, 1);
    imshow(im2double(raw), []);
    title('1. Original', 'FontSize', 10, 'FontWeight', 'bold');
    xlabel('Periodic grid + speckle noise', 'FontSize', 8);

    % Panel 2 — After spatial (median) filter
    subplot(1, 5, 2);
    imshow(spatialFiltered, []);
    title('2. Spatial filter', 'FontSize', 10, 'FontWeight', 'bold');
    xlabel('Median 3\times3 — speckle removed', 'FontSize', 8);

    % Panel 3 — 2D FFT magnitude spectrum
    subplot(1, 5, 3);
    imshow(logMagSpectrum, []);
    colormap(gca, 'hot');
    title('3. FFT spectrum', 'FontSize', 10, 'FontWeight', 'bold');
    xlabel('Bright spikes = grid harmonics', 'FontSize', 8);

    % Panel 4 — Notch mask
    subplot(1, 5, 4);
    imshow(notchMask, []);
    title('4. Notch mask', 'FontSize', 10, 'FontWeight', 'bold');
    xlabel('Black = suppressed frequencies', 'FontSize', 8);

    % Panel 5 — Final restored image
    subplot(1, 5, 5);
    imshow(restored, []);
    title('5. Restored', 'FontSize', 10, 'FontWeight', 'bold');
    xlabel('Grid removed + contrast enhanced', 'FontSize', 8);

    sgtitle(sprintf('Image Restoration Pipeline — %s', fname), ...
            'FontSize', 12, 'FontWeight', 'bold');

    % Save the figure
    saveas(gcf, fullfile('C:\Users\kgowt\Downloads\schem_output', ...
           [fname '_diagnostic.png']));

end