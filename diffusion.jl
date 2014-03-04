###
#   transforms concs and dirality
###

using NumericExtensions

import Base.exp, Base./, Base.*, Base.-, Base.sum

immutable Number8 {T <: FloatingPoint}
    s0 :: T
    s1 :: T
    s2 :: T
    s3 :: T
    s4 :: T
    s5 :: T
    s6 :: T
    s7 :: T
end

function toArray(n :: Number8)
    return [n.s0, n.s1, n.s2, n.s3, n.s4, n.s5, n.s6, n.s7]
end

function exp(n :: Number8)
    an = toArray(n)
    exp!(an)
    return Number8(an...)
end

*(a :: Real, n :: Number8) = Number8(multiply!(toArray(n), a)...)
-(a :: Real, n :: Number8) = Number8((a - toArray(n))...)
/(n1 :: Number8, n2 :: Number8) =Number8(divide!(toArray(n1), toArray(n2))...)
/(n1 :: Number8, a :: Real) =Number8(divide!(toArray(n1), a)...)


function sum(n:: Number8)
    return sum(toArray(n))
end

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

function diffusion(conc :: Matrix, pot :: Matrix) #concentration and direction
    d1, d2 = size(conc)

    Flow = flow(pot)

    p_move = zeros(Float64, d1, d2)

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

            # fsw = Flow[south,west].s7
            # fsj = Flow[south,j   ].s6
            # fse = Flow[south,east].s5
            # fiw = Flow[i    ,west].s4
            # fie = Flow[i    ,east].s3
            # fnw = Flow[north,west].s0
            # fnj = Flow[north,j   ].s1
            # fne = Flow[north,east].s2

            # inflow  =  conc[south, west] * fsw +
            #            conc[south, j   ] * fsj +
            #            conc[south, east] * fse +
            #            conc[i    , west] * fiw +
            #            conc[i    , east] * fie +
            #            conc[north, east] * fne +
            #            conc[north, j   ] * fnj +
            #            conc[north, west] * fnw ;


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
    return p_move
end