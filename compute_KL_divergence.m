function D_KL = compute_KL_divergence(mixPDF, grid, T)
    % Compute the Kullback-Leibler divergence D_KL(p || q)
    % where:
    % - p(x) is a Gaussian mixture model (MATLAB gmdistribution)
    % - q(x) is given as a discretized density (tensor T) on a grid X_grid
    %
    % Inputs:
    %   mixPDF : MATLAB built-in Gaussian mixture model (gmdistribution)
    %   grid   : Cell array {grid_dim1, grid_dim2, ...} defining discretized grid points
    %   T      : [N x 1] Density values corresponding to each grid point in X_grid
    %
    % Output:
    %   D_KL   : Kullback-Leibler divergence D_KL(p || q)

    d = length(grid);  % Dimensionality
    dx = cellfun(@(x) x(2) - x(1), grid);  % Grid spacing

    % Create the grid mesh and reshape to N x d
    grid_mesh = cell(1, d);
    [grid_mesh{:}] = ndgrid(grid{:});
    X_grid = cell2mat(cellfun(@(x) x(:), grid_mesh, 'UniformOutput', false));

    % Compute Gaussian mixture density p(x) at each grid point
    p_x = pdf(mixPDF, X_grid);
    
    % Normalize p_x to be a valid probability density function
    p_x = p_x / sum(p_x(:) * prod(dx));
    
    % Normalize T to ensure q(x) is also a valid probability density function
    T = T / sum(T(:) * prod(dx));
    
    % Ensure no division by zero or log(0)
    min_val = eps('double');  % Machine epsilon for double precision
    p_x(p_x < min_val) = min_val;  % Avoid values too close to zero in p_x
    T(T < min_val) = min_val;      % Avoid values too close to zero in T
    
    % Compute KL divergence
    D_KL = sum(p_x .* (log(p_x) - log(T)) * prod(dx));    
end
