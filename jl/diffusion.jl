###
#   transforms concs and dirality
###

require("jl/ocl_utils.jl")

function flow(Pot :: Matrix)
    d1, d2 = size(Pot)

    Flow = Array(Number8, d1, d2)

    for j in 1:d2
        for i in 1:d1

            west  = j == 1  ? d2 : j-1
            east  = j == d2 ? 1  : j+1
            north = i == d1 ? 1  : i+1
            south = i == 1  ? d1 : i-1

            ###
            # Calculate the probability of flow out of a cell based on the pot differnece

            p   = Pot[i    ,   j]    ;
            psw = Pot[south,west] - p;
            psj = Pot[south,j   ] - p;
            pse = Pot[south,east] - p;
            piw = Pot[i    ,west] - p;
            pie = Pot[i    ,east] - p;
            pnw = Pot[north,west] - p;
            pnj = Pot[north,j   ] - p;
            pne = Pot[north,east] - p;

            ge = Number8(psw, psj, pse, piw, pie, pnw, pnj, pne)

            ge = -1 * ge / (1-exp(ge))

            ###
            # Remove NaN
            ###

            ge_wo_na = Number8([isnan(x) ? 1.0 : x for x in toArray(ge)]...)

            ###
            # Normalize by eight.
            ###

            Flow[i,j] = ge_wo_na / 8.0
        end
    end
    return Flow
end

function diffusionJl!(conc :: Matrix, pot :: Matrix, p_move :: Matrix) #concentration and direction
    d1, d2 = size(conc)

    Flow = flow(pot)

    for j in 1:d2
        for i in 1:d1

            west  = j == 1  ? d2 : j-1
            east  = j == d2 ? 1  : j+1
            north = i == d1 ? 1  : i+1
            south = i == 1  ? d1 : i-1

            ###
            # Get the flow out of cell ij
            ###

            outflow =  conc[i,j] * sum(Flow[i,j])

            ###
            # Calculate the inflow based on the outflow from other cells into this on.
            ###

            inflow  =  conc[south, west] * Flow[south, west].s7 +
                       conc[south, j   ] * Flow[south, j   ].s6 +
                       conc[south, east] * Flow[south, east].s5 +
                       conc[i    , west] * Flow[i    , west].s4 +
                       conc[i    , east] * Flow[i    , east].s3 +
                       conc[north, east] * Flow[north, east].s0 +
                       conc[north, j   ] * Flow[north, j   ].s1 +
                       conc[north, west] * Flow[north, west].s2 ;

            ###
            # Inflow - outflow = change
            ###

            p_move[i,j] = inflow - outflow
        end
    end
end