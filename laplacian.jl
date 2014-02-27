###
# Laplacian
###
function LaPlacian(Ap :: Matrix)
    d1, d2 = size(Ap)

    A_lap = zeros(d1, d2)

    for j in 1:d2
        for i in 1:d1

            west  = j == 1  ? d2 : j-1
            east  = j == d2 ? 1  : j+1
            north = i == d1 ? 1  : i+1
            south = i == 1  ? d1 : i-1

            #8-neighbor
            #inflow

            a = Ap[i,j]

            A_lap[i,j] -= a

            #outflow

            a8 = a / 8

            A_lap[north, east] += a8
            A_lap[north, j   ] += a8
            A_lap[north, west] += a8
            A_lap[i    , west] += a8

            A_lap[south, west] += a8
            A_lap[south, j   ] += a8
            A_lap[south, east] += a8
            A_lap[i    , east] += a8

        end
    end
    return A_lap
end
