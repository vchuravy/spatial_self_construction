###
# calculate interactive potential based on repulsion
# repulsion on hydrophobe given concentration of hydrophils
###

require("jl/ocl_utils.jl")

function potentialJl!{T <: FloatingPoint}(Afield ::  Array{T, 2}, Bfield ::  Array{T, 2}, Area :: Array{Number4{T},2}, Apot ::  Array{T, 2}, Bpot ::  Array{T, 2}, repulsion :: Real)
    d1, d2 = size(Afield)

    selfRepulsion = 1*repulsion # making this lower than repulsion allows for neighbors to have relative potential, so increases the chance that the hydrophobe will flow.

    for j in 1:d2
        for i in 1:d1

            west  = j == 1  ? d2 : j-1
            east  = j == d2 ? 1  : j+1
            north = i == d1 ? 1  : i+1
            south = i == 1  ? d1 : i-1

            a_ij = Area[i, j];

            r_ij = selfRepulsion * Afield[i,j];

            a_ne = Area[north, east].s0;
            a_nj = Area[north, j   ].s1;
            a_nw = Area[north, west].s2;
            a_iw = Area[i    , west].s3;

            a_sw = Area[south, west].s0;
            a_sj = Area[south, j   ].s1;
            a_se = Area[south, east].s2;
            a_ie = Area[i    , east].s3;

            area = [a_ne, a_nj, a_nw, a_iw, a_sw, a_sj, a_se, a_ie]

            r_ne = Afield[north, east];
            r_nj = Afield[north, j   ];
            r_nw = Afield[north, west];
            r_iw = Afield[i    , west];
            r_sw = Afield[south, west];
            r_se = Afield[south, east];
            r_sj = Afield[south, j   ];
            r_ie = Afield[i    , east];

            r_af = [r_ne, r_nj, r_nw, r_iw, r_sw, r_sj, r_se, r_ie]
            r_af = multiply!(r_af, repulsion)

            Bpot[i,j] = sum(multiply!(r_af, area)) + r_ij;

            Apot[i,j] = repulsion *(Bfield[north,east] * a_ij.s0 +
                                    Bfield[north,j   ] * a_ij.s1 +
                                    Bfield[north,west] * a_ij.s2 +
                                    Bfield[i,west    ] * a_ij.s3 +
                                    Bfield[south,west] * a_ij.s0 +
                                    Bfield[south,j   ] * a_ij.s1 +
                                    Bfield[south,east] * a_ij.s2 +
                                    Bfield[i,east    ] * a_ij.s3 + Bfield[i,j]);
        end
    end
end