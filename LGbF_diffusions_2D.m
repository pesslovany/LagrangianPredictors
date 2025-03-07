function [t_diff, e_mean, e_var, D_KL] = LGbF_diffusions_2D(technique,Npa,Q,dtSpec)

%   Lagrangian based grid-based filter diffusion solutions
%   authors: Jakub Matousek, pesslovany@gmail.com
%            Jindrich Dunik, University of West Bohemia
%            Felix Govaers,
%            Joshua Gehlen,  Fraunhofer FKIE

%% Parameters and system simulation
whichDer = [0 0]; % which variables will come into cross derivatives
nx = 2; % state dimension
invQ = inv(Q);
sFactor = 5; % scaling factor (number of sigmas covered by the grid)


% Initial condition - Gaussian
% Gaussian mixture initial PMD
X01 = [2;-1];
X02 = [-2;1];
VARX01 = eye(nx);
VARX02 = eye(nx);
alpha1 = 0.4;
alpha2 = 1 - alpha1;
[meanX0,varX0] = gaussianMixture2(X01,X02,VARX01,VARX02,[alpha1 alpha2]);
F = [1, 1,;
    0, 1,]; % model dynamics NCV
x = mvnrnd(meanX0, varX0, 1)';
w = mvnrnd(zeros(nx,1),Q, 1)';
x(:,2) = F*x(:,1) + w(:,1);

% True preditive PDF
predMeans = [F * X01, F * X02];  
predGMM = gmdistribution(predMeans', cat(3, F * VARX01 * F' + Q, F * VARX02 * F' + Q), [alpha1, alpha2]); 

% Initial grid
[filtGrid, filtGridDelta, gridDimOld, gridCenter, gridRotation] = gridCreation(meanX0,varX0,sFactor,nx,Npa);

% Initial PMD - mixture
fixTerm = (filtGrid-X01);
denominator = sqrt((2*pi)^nx*det(VARX01));
filtPdf1 = ((exp(sum(-0.5*fixTerm'/(VARX01).*fixTerm',2)))/denominator); % Initial Gaussian Point mass density (PMD)


fixTerm = (filtGrid-X02);
denominator = sqrt((2*pi)^nx*det(VARX02));
filtPdf2 = ((exp(sum(-0.5*fixTerm'/(VARX02).*fixTerm',2)))/denominator); % Initial Gaussian Point mass density (PMD)

filtPdf = alpha1*filtPdf1 + alpha2*filtPdf2;
filtPdf = filtPdf/sum(filtPdf*prod(filtGridDelta));

% Auxiliary variables
predDenDenomW = sqrt((2*pi)^nx*det(Q)); % Denominator for convolution in predictive step

% Filtering mean and var
filtMeanPMF = filtGrid*filtPdf*prod(filtGridDelta); %#ok<*SAGROW> % Measurement update mean
chip_ = (filtGrid-filtMeanPMF);
chip_w = chip_.*repmat(filtPdf',nx,1);
filtVarPMF = chip_w*chip_' * prod(filtGridDelta); % Measurement update variance


%% Interpolation - to fix the grid shrinking/enlarging and account for Q
%(some bonus rotation of state space so that griddedInterpolant can be used)
% Grid to interp on (for small Q can be skipped)

predVarPom = F*filtVarPMF*F' + Q;
wantPredVar = diag(diag(predVarPom));
wantedPredMean = F*filtMeanPMF;

[predGridAdvect, predGridDelta, predGridDim, ~, ~] = gridCreation(wantedPredMean,wantPredVar,sFactor,nx,Npa); % create the grid to interp on
filtGridNew = inv(F)*predGridAdvect;
gridStepNew = inv(F)*predGridDelta;
% Interpolation
Fint = griddedInterpolant(gridDimOld,reshape(filtPdf,Npa),"linear","none"); % interpolant
inerpOn = inv(gridRotation)*(filtGridNew - gridCenter); % Grid to inter on transformed to rotated space
filtPdfNew = Fint(inerpOn'); % Interpolation
filtPdfNew(isnan(filtPdfNew)) = 0; % Zeros for extrapolation, otherwise artifacts would appear



%% Advection solution
predPdf = filtPdfNew/sum(filtPdfNew*prod(predGridDelta));

% True values
advPredTrue = F*meanX0;

%% Diff solution spectral
if technique == 2
    advSolWeights = predPdf; % Weights/prdictive PDF
    advSolGrid = predGridDim; % Grid given per axis
    advSolDelta = predGridDelta; % Equdistant grid resolution per axis
    advSolGridFull = predGridAdvect;
    
    tic

    filtDenDOTprodDeltas = (advSolWeights); % measurement PDF * measurement PDF step size
    filtDenDOTprodDeltasCub = reshape(filtDenDOTprodDeltas,Npa); % Into physical space

    filtDenDOTprodDeltasCub = setEdgesToZeros(filtDenDOTprodDeltasCub);

    dims = numel(gridDimOld);
    L = cellfun(@(g) g(end) - g(1), advSolGrid);

    kInd = cell(1, dims);
    kindFirst = cell(1, dims);
    gridSize = zeros(1, dims);

    for d = 1:dims
        gridSize(d) = Npa(d);
        % Derivatives coefficients
        kInd{d} = 2 * pi / L(d) * fftshift(-(Npa(d)) / 2 : (Npa(d)) / 2 - 1).';
    end

    % Initialize coefficient with ones of the final tensor shape
    coeficienty = zeros(gridSize);

    % Add diagonal terms - second derivatives
    for d = 1:dims
        shape = ones(1, dims);
        shape(d) = gridSize(d);
        coeficienty = coeficienty + reshape(kInd{d}.^2 * dtSpec * (Q(d, d) / 2), shape);
    end

    % Prepare the coefficients for the whole time step k -> k+1
    coeficienty = (1 ./(1 + coeficienty)).^(1 / dtSpec);

    %  Do the update
    dims = 1:1:nx;
    u = filtDenDOTprodDeltasCub;
    for dim = dims
        u = fft(u,Npa(dim),dim);
    end
    u = (coeficienty).*u;
    predDensityProb2cub = real(ifftn(u)); % realna cast

    % Normalize
    predDensityProb = reshape(predDensityProb2cub,length(advSolGridFull),1); % back to computational space
    predPdf = predDensityProb./(sum(predDensityProb)*prod(advSolDelta))'; % Normalizaton (theoretically not needed)
    t_diff = toc;
end

%% Diffusion solution - Matrix exponential-based -----------------------------------------


if technique == 0
    advSolWeights_mat = reshape(predPdf,Npa); % Weights in 'matrix/tensor format'
    advSolGrid = predGridDim; % Grid given per axis
    advSolDelta = predGridDelta; % Equdistant grid resolution per axis

    tic
    % Code adapted for arbitrary dimensions
    dims = numel(Npa);

    % Precompute second derivative matrices for each dimension
    D2 = cell(1, dims);
    M = cell(1, dims);

    for d = 1:dims
        vec_2 = zeros(Npa(d),1);
        vec_2(1) = -2;
        vec_2(2) = 1;
        D2{d} = toeplitz(vec_2) / advSolDelta(d)^2;

        % Diffusion operator
        [V, D] = eig(D2{d});
        M{d} = (V * diag(exp(1 / 2 * Q(d,d) * diag(D))) / V);
    end

    % Apply diffusion operator
    advSolWeights_mat = ttm(advSolWeights_mat, M, 1:dims);

    % reshape of the density to a vector
    predPdf = reshape(advSolWeights_mat, [], 1);


    t_diff = toc;
end



%--------------------------------------------------------------------------
%% Diffusion solution - FFT-based

%True value
meanCovTrueDiff = F*varX0*F' + Q;

if technique == 1
    advSolGrid = predGridDim;
    tic

    % FFT solution
    predPdf = reshape(predPdf,Npa);
    mesPdfDotDeltasTens = (predPdf*prod(gridStepNew)); % filt PDF * filt PDF step size
    halfGridInd = ceil(length(predGridAdvect)/2); % Index of middle point of predictive grid
    distPoints = (predGridAdvect(:,halfGridInd)'-(predGridAdvect)'); % Distance of transformed grid points to the new grid points
    convKer = ((exp(sum(-0.5*distPoints*invQ.*distPoints,2)))/predDenDenomW)';% Convolution kernel values
    convKerTens = reshape(convKer,Npa); % Convolution kernel values in tensor format

    [predDensityProb2cub, ~] = convFftN(mesPdfDotDeltasTens, convKerTens, Npa, nx, 0); % FFT convolution
    predPdf = reshape(predDensityProb2cub,length(predGridAdvect),1); % back to vector for easier manipulation

    predPdf = predPdf./(sum(predPdf)*prod(predGridDelta))'; % Normalizaton (theoretically not needed)
    predPdf(predPdf<0) = 0; % FFT approach sometimes can produce very small negative values

    t_diff = toc;
end

%%  Fast sine transform
if technique == 3
    advSolDelta = predGridDelta; % Equdistant grid resolution per axis
    predPdf = reshape(predPdf,Npa);
    advSolGrid = predGridDim;

    mesPdfDotDeltasTens = (predPdf*prod(gridStepNew)); % filt PDF * filt PDF step size

    tic

    a = (Q(1,1)*dtSpec)/(2*advSolDelta(1)^2); % Finite difference diffusion matrix diagonal values
    b = 1 - (Q(1,1)*dtSpec)/(2*advSolDelta(1)^2) - (Q(2,2)*dtSpec)/(2*advSolDelta(2)^2);
    c = Q(2,2)*dtSpec/(2*advSolDelta(2)^2);

    count = (1:1:Npa(1)); % Indices
    % Eigenvalues of diffusion matrix
    lambdaJ = b + 2*a*cos((count*pi)/(Npa(1)+1)) + 2*c*cos((count'*pi)/(Npa(1)+1)); 

    % Fast sine calculation of prediction
    posteriorInSine = (dtt2D(mesPdfDotDeltasTens,5)); 
    predDensityProb2cub = dtt2D(lambdaJ.^(1/dtSpec).*posteriorInSine,5);

    % Normalization reshaping
    predPdf = reshape(predDensityProb2cub,length(predGridAdvect),1); % back to vector for easier manipulation
    predPdf = predPdf./(sum(predPdf)*prod(predGridDelta)); % Normalizaton (theoretically not needed)
    predPdf(predPdf<0) = 0; % SFT approach sometimes can produce very small negative values

    t_diff = toc;
end

%%

% Predictive mean and var
predMeanPMF = predGridAdvect*predPdf*prod(predGridDelta); %#ok<*SAGROW> % Measurement update mean
chip_ = (predGridAdvect-predMeanPMF);
chip_w = chip_.*repmat(predPdf',nx,1);
predVarPMF = chip_w*chip_' * prod(predGridDelta); % Measurement update variance

% Check results
% fprintf('----------------------Diffusion + advection----------------------\n')
[e_mean, e_var] = printGaussianMixInfo(advPredTrue, meanCovTrueDiff, predMeanPMF, predVarPMF);
D_KL = compute_KL_divergence(predGMM, advSolGrid, predPdf);

end

%% Functions


function [mu_mix, Sigma_mix] = gaussianMixture2(X01, X02, VARX01, VARX02, weights)
% Calculates the mean and covariance of a Gaussian mixture with two components
%
% Inputs:
%   X01 - Mean of the first Gaussian component (nx x 1)
%   X02 - Mean of the second Gaussian component (nx x 1)
%   VARX01 - Covariance of the first Gaussian component (nx x nx)
%   VARX02 - Covariance of the second Gaussian component (nx x nx)
%   weights - Weights of the two components (1 x 2)
%
% Outputs:
%   mu_mix - Mean of the Gaussian mixture (nx x 1)
%   Sigma_mix - Covariance of the Gaussian mixture (nx x nx)

% Validate inputs
assert(length(weights) == 2, 'Weights must have two elements.');
assert(abs(sum(weights) - 1) < 1e-10, 'Weights must sum to 1.');

% Compute the mixture mean
mu_mix = weights(1) * X01 + weights(2) * X02;

% Compute the covariance
diff1 = X01 - mu_mix; % Deviation of the first mean
diff2 = X02 - mu_mix; % Deviation of the second mean

Sigma_mix = weights(1) * (VARX01 + diff1 * diff1') + ...
    weights(2) * (VARX02 + diff2 * diff2');
end


function [e_mean, e_var] = printGaussianMixInfo(meanX0, varX0, filtMeanPMF, filtVarPMF)
% Print Gaussian mixture information with matrices as matrices
%
% Inputs:
%   meanX0 - True mean (nx x 1)
%   varX0 - True covariance matrix (nx x nx)
%   filtMeanPMF - Predicted mean (nx x 1)
%   filtVarPMF - Predicted covariance matrix (nx x nx)

e_mean = mean(abs(meanX0 - filtMeanPMF));
e_var = (sum(abs(varX0 - filtVarPMF), "all")/sum(abs(filtVarPMF),"all"))*100;
end

function [pdfIn, kernel] = convFftN(pdfIn, kernel, Npa, nx, kernelFFT)
% Calculates convolution by FFT
%INPUTS:
% pdfIn - pdf for convolution
% kernel - kernel for convolution
% Npa - number of points per axis
% nx - dimension
% kernelFFT - is kernel already in frequency space?
%OUTPUTS:
% pdfIn - convolution result

dims = 1:1:nx;
% Will be used to truncate the padding need the do the convolution
ifun = @(m,n) ceil((n-1)/2)+(1:m);
subs(1:ndims(pdfIn)) = {':'};

for dim=dims % FFT over all dimensions
    % compute the FFT length with padding
    l = Npa(dim)+Npa(dim)-1;
    pdfIn = fft(pdfIn,l,dim); % FFT of the PDF
    if ~kernelFFT
        kernel = fft(kernel,l,dim); % FFT of the kernel
    end
    subs{dim} = ifun(Npa(dim),Npa(dim)); % Padding indices
end

% Perform convolution
pdfIn = pdfIn.*kernel;

% Back to state space
for dim=dims
    pdfIn = ifft(pdfIn,[],dim);
end

% Make sure the result is real
pdfIn = real(pdfIn(subs{:}));

end
