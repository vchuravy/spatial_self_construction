###
#   transforms concs and dirality
###

using NumericExtensions

function diffusion(conc :: Matrix, pot :: Matrix) #concentration and direction
    d1, d2 = size(conc)

    p_move = zeros(Float64, d1, d2)
    temp   = zeros(Float64, 3, 3)
    ge     = zeros(Float64, 3, 3)

    for j in 1:d2
        for i in 1:d1

            west  = j == 1  ? d2 : j-1
            east  = j == d2 ? 1  : j+1
            north = i == d1 ? 1  : i+1
            south = i == 1  ? d1 : i-1

            ###
            # probabilities of movement #
            ##

            p = pot[i, j]

            temp[1, 1] = pot[south, west] - p
            temp[1, 2] = pot[south, j   ] - p
            temp[1, 3] = pot[south, east] - p
            temp[2, 1] = pot[i    , west] - p
            temp[2, 2] = 0.0
            temp[2, 3] = pot[i    , east] - p
            temp[3, 1] = pot[north, west] - p
            temp[3, 2] = pot[north, j   ] - p
            temp[3, 3] = pot[north, east] - p

            ge[1, 1] = temp[1, 1]
            ge[1, 2] = temp[1, 2]
            ge[1, 3] = temp[1, 3]
            ge[2, 1] = temp[2, 1]
            ge[2, 2] = 0.0
            ge[2, 3] = temp[2, 3]
            ge[3, 1] = temp[3, 1]
            ge[3, 2] = temp[3, 2]
            ge[3, 3] = temp[3, 3]

            temp = exp!(temp)
            temp = negate!(temp)
            temp = add!(temp, 1.0)
            ge = divide!(ge, temp)
            ge[2, 2] = 0.0

            ge = negate!(ge)

            for idx in 1:9
                if isnan(ge[idx])
                    ge[idx] = 1.0
                end
            end

            concentration = conc[i,j]

            #8-neighbor
            #inflow
            p_move[i,j] -= concentration * sum(ge) / 9.0

            #outflow
            ge = multiply!(ge, concentration / 9.0)

            p_move[south,west] += ge[1, 1]
            p_move[south,j   ] += ge[1, 2]
            p_move[south,east] += ge[1, 3]
            p_move[i    ,west] += ge[2, 1]

            p_move[i    ,east] += ge[2, 3]
            p_move[north,west] += ge[3, 1]
            p_move[north,j   ] += ge[3, 2]
            p_move[north,east] += ge[3, 3]
        end
    end
    return p_move
end