function LaPlacian(Ap :: Matrix)
    A_lap = similar(Ap)

    LaPlacian!(A_lap, Ap)
end

function LaPlacian!(A_lap :: Matrix, Ap :: Matrix)
    D1, D2 = size(Ap)    
    neighbourhood = zeros(3, 3)
    
    for j in 1:D2
        for i in 1:D1    
            get_moore!(neighbourhood, Ap, i, j, D1, D2)
            
            centre = neighbourhood[2, 2]
            neighbourhood[2, 2] = 0.0
            
            A_lap[i,j] = -centre + sum(neighbourhood) / 8.0
        end
    end
end