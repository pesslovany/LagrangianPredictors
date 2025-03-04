function A = setEdgesToZeros(A)
    dims = ndims(A);
    sizeA = size(A);

    % Loop over each dimension to set the edges to zero
    for dim = 1:dims
        % Create index vectors for all dimensions
        index = repmat({':'}, 1, dims);
        
        % Set the first edge along this dimension
        index{dim} = 1;
        A(index{:}) = 0;

        % Set the last edge along this dimension
        index{dim} = sizeA(dim);
        A(index{:}) = 0;
    end
end