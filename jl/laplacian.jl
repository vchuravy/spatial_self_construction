###
# Laplacian
###

function LaPlacianJl!(Ap :: Matrix, A_lap :: Matrix)
    d1, d2 = size(Ap)

    for j in 1:d2
        for i in 1:d1

            west  = j == 1  ? d2 : j-1
            east  = j == d2 ? 1  : j+1
            north = i == d1 ? 1  : i+1
            south = i == 1  ? d1 : i-1

            #8-neighbor
            #inflow

            A_lap[i,j] = -Ap[i,j] +(Ap[north, east] +
                                    Ap[north, j   ] +
                                    Ap[north, west] +
                                    Ap[i    , west] +
                                    Ap[south, west] +
                                    Ap[south, j   ] +
                                    Ap[south, east] +
                                    Ap[i    , east]) / 8 ;
        end
    end
end
