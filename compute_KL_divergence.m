function D_KL = compute_KL_divergence(mu, P, grid, T)
    % Compute the Kullback-Leibler divergence D_KL(p || q)
    % where:
    % - p(x) is a multivariate Gaussian with mean 'mu' and covariance 'P'
    % - q(x) is given as a discretized density (tensor T) on a grid X_grid
    %
    % Inputs:
    %   mu    : [d x 1] Mean vector of the Gaussian
    %   P     : [d x d] Covariance matrix of the Gaussian
    %   X_grid: [N x d] Discretized grid points (each row is a d-dimensional point)
    %   T     : [N x 1] Density values corresponding to each grid point in X_grid
    %
    % Output:
    %   D_KL  : Kullback-Leibler divergence D_KL(p || q)

    d = length(mu);
    dx = cellfun(@(x) x(2) - x(1), grid);  % Grid spacing

    % Create the grid mesh and reshape to N x d
    grid_mesh = cell(1, d);  % Initialize the cell array to store grid dimensions
    [grid_mesh{:}] = ndgrid(grid{:});  % Generate the full mesh grid

    % Reshape grid points to N x d
    X_grid = cell2mat(cellfun(@(x) x(:), grid_mesh, 'UniformOutput', false));

    % Compute the Gaussian density p(x) at each grid point
    p_x = mvnpdf(X_grid, mu', P);
    
    % Normalize p_x (Gaussian density)
    p_x = p_x / sum(p_x(:) * prod(dx));

    % Normalize T to make q(x) a valid probability density function
    T = T / sum(T(:) * prod(dx));

    % Ensure no division by zero or very small values in T and p_x
    min_val = 1e-12;
    p_x(p_x < min_val) = min_val;  % Avoid log(0) in p_x
    T(T < min_val) = min_val;  % Avoid log(0) in T

    % Compute the KL divergence using more numerically stable summation
    log_p_x = log(p_x);
    log_T = log(T);

    % Instead of multiplying the terms directly, use the log-sum-exp trick
    D_KL = sum(p_x .* (log_p_x - log_T) * prod(dx));  % Numerically stable summation
end
