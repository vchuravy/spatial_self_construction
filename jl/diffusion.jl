###
#   transforms concs and dirality
###
import NumericExtensions.negate!, NumericExtensions.divide!, NumericExtensions.add!, NumericExtensions.exp!

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
            pie = Pot[i    ,east] - p;
            pne = Pot[north,east] - p;
            pnj = Pot[north,j   ] - p;
            pnw = Pot[north,west] - p;
            piw = Pot[i    ,west] - p;

            ge = [psw, psj, pse, pie, pne, pnj, pnw, piw]
            ge2 = [psw, psj, pse, pie, pne, pnj, pnw, piw]
            ge = negate!(divide!(ge2, add!(negate!(exp!(ge)), 1))) #

            ###
            # Remove NaN
            ###

            ge_wo_na = Number8([isnan(x) ? 1.0 / 8.0 : x/8.0 for x in ge])

            ###
            # Normalize by eight.
            ###

            Flow[i,j] = ge_wo_na
        end
    end
    return Flow
end

function diffusionJl!(Conc :: Matrix, pot :: Matrix, p_move :: Matrix) #concentration and direction
    d1, d2 = size(Conc)

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

            outflow =  Conc[i,j] * sum(Flow[i,j])

            ###
            # Calculate the inflow based on the outflow from other cells into this on.
            ###

            inflow  =   Conc[south, west] * Flow[south, west].s4 +
                        Conc[south, j   ] * Flow[south, j   ].s5 +
                        Conc[south, east] * Flow[south, east].s6 +
                        Conc[i    , east] * Flow[i    , east].s7 +
                        Conc[north, east] * Flow[north, east].s0 +
                        Conc[north, j   ] * Flow[north, j   ].s1 +
                        Conc[north, west] * Flow[north, west].s2 +
                        Conc[i    , west] * Flow[i    , west].s3 ;

            ###
            # Inflow - outflow = change
            ###

            p_move[i,j] = inflow - outflow
        end
    end
end