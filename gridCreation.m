function [gridOut, gridDeltaOut, gridDim, meanIn, eigVect] = gridCreation(meanIn,covIn,sFactor,nx,Npa)
%gridCreation creates a grid based on first two moments

    [eigVect,eigVal] = eig(covIn); % eigenvalue and eigenvectors, for setting up the grid
    eigVal = diag(eigVal);
    gridBound = sqrt(eigVal)*sFactor; % Boundaries of grid
    
    % Ensure the grid steps are in the right order
    [~,I] = sort(diag(covIn));
    [~,I] = sort(I);

    [pom,Ipom] = sort(gridBound);
    gridBound = pom(I);

    pom2 = eigVect(:,Ipom);
    eigVect = pom2(:,I);

    gridDim = cell(nx,1);
    gridStep = nan(nx,1);
    for ind3 = 1:1:nx %Creation of propagated grid
        gridDim{ind3,1} = linspace(-gridBound(ind3),gridBound(ind3),Npa(ind3)); %New grid with middle in 0 in one coordinate
        gridStep(ind3,1) = abs(gridDim{ind3,1}(1)-gridDim{ind3,1}(2)); %Grid step
    end
    gridOut = eigVect*combvec(gridDim) + meanIn; %Grid rotation by eigenvectors and traslation to the counted mean

    gridDeltaOut = gridStep;

end

