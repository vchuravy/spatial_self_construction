###
#   transforms concs and dirality
###
function diffusion(conc :: Matrix, pot :: Matrix) #concentration and direction
    d1, d2 = size(conc)

    p_move = zeros(d1, d2)
    ge = zeros(3, 3)

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

            ge[1, 1] = -1*(pot[south, west] - p) / (1-exp(pot[south, west] - p))
            ge[1, 2] = -1*(pot[south, j   ] - p) / (1-exp(pot[south, j   ] - p))
            ge[1, 3] = -1*(pot[south, east] - p) / (1-exp(pot[south, east] - p))
            ge[2, 1] = -1*(pot[i    , west] - p) / (1-exp(pot[i    , west] - p))
            ge[2, 2] = 0;
            ge[2, 3] = -1*(pot[i    , east] - p) / (1-exp(pot[i    , east] - p))
            ge[3, 1] = -1*(pot[north, west] - p) / (1-exp(pot[north, west] - p))
            ge[3, 2] = -1*(pot[north, j   ] - p) / (1-exp(pot[north, j   ] - p))
            ge[3, 3] = -1*(pot[north, east] - p) / (1-exp(pot[north, east] - p))

            ge[isnan(ge)] = 1


            #8-neighbor
            #inflow
            sm = sum(ge)/9 # ??? avg?
            p_move[i,j] -= conc[i,j] * sm

            #outflow
            p_move[south,west] = p_move[south, west] + conc[i, j] * ge[1, 1] / 9
            p_move[south,j   ] = p_move[south, j   ] + conc[i, j] * ge[1, 2] / 9
            p_move[south,east] = p_move[south, east] + conc[i, j] * ge[1, 3] / 9
            p_move[i    ,west] = p_move[i    , west] + conc[i, j] * ge[2, 1] / 9

            p_move[i    ,east] = p_move[i    , east] + conc[i, j] * ge[2, 3] / 9
            p_move[north,west] = p_move[north, west] + conc[i, j] * ge[3, 1] / 9
            p_move[north,j   ] = p_move[north, j   ] + conc[i, j] * ge[3, 2] / 9
            p_move[north,east] = p_move[north, east] + conc[i, j] * ge[3, 3] / 9
        end
    end
    return p_move
end



