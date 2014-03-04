###
# calculate interactive potential based on repulsion
# repulsion on hydrophobe given concentration of hydrophils
###

using NumericExtensions

immutable Number4 {T <: FloatingPoint}
    s0 :: T
    s1 :: T
    s2 :: T
    s3 :: T
end

function area{T <: FloatingPoint}(direction :: Array{T, 2}, LONG :: Real)
    D1, D2 = size(direction)
    SHORT = one(T)
    dout = Array(Number4, D1, D2)
    PI_4 = pi/4
    PI = pi

    for i in 1:length(direction)
            dir = direction[i];

            area1 = LONG*SHORT/2 * (   PI_4/2-dir - atan((SHORT-LONG)*sin(2*(  PI_4/2-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(  PI_4/2-dir)))));
            area2 = LONG*SHORT/2 * ( 3*PI_4/2-dir - atan((SHORT-LONG)*sin(2*(3*PI_4/2-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(3*PI_4/2-dir)))));
            area3 = LONG*SHORT/2 * ( 5*PI_4/2-dir - atan((SHORT-LONG)*sin(2*(5*PI_4/2-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(5*PI_4/2-dir)))));
            area4 = LONG*SHORT/2 * ( 7*PI_4/2-dir - atan((SHORT-LONG)*sin(2*(7*PI_4/2-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(7*PI_4/2-dir)))));
            area5 = LONG*SHORT/2 * ( 9*PI_4/2-dir - atan((SHORT-LONG)*sin(2*(9*PI_4/2-dir)) / (SHORT+LONG + (SHORT-LONG)*cos(2*(9*PI_4/2-dir)))));

            lps = SHORT * LONG * PI;

            s0 = (area2-area1)/lps;
            s1 = (area3-area2)/lps;
            s2 = (area4-area3)/lps;
            s3 = (area5-area4)/lps;

            dout[i] = Number4(s0, s1, s2, s3);
    end
    return dout
end


function potential(Afield :: Matrix, Bfield :: Matrix, direction :: Matrix, repulsion :: Real, long :: Real)
    d1, d2 = size(Afield)

    # is the anisotropic molecule

    Apot = zeros(Float64, d1, d2)
    Bpot = zeros(Float64, d1, d2)

    Area = area(direction, long)

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
            a_sj = Area[south, east].s1;
            a_se = Area[south, j   ].s2;
            a_ie = Area[i    , east].s3;

            r_ne = repulsion * Afield[north, east];
            r_nj = repulsion * Afield[north, j   ];
            r_nw = repulsion * Afield[north, west];
            r_iw = repulsion * Afield[i    , west];
            r_sw = repulsion * Afield[south, west];
            r_sj = repulsion * Afield[south, east];
            r_se = repulsion * Afield[south, j   ];
            r_ie = repulsion * Afield[i    , east];

            Bpot[i,j] =    fma(r_ne, a_ne,
                           fma(r_nj, a_nj,
                           fma(r_nw, a_nw,
                           fma(r_iw, a_iw,
                           fma(r_sw, a_sw,
                           fma(r_sj, a_sj,
                           fma(r_se, a_se,
                           fma(r_ie, a_ie, r_ij))))))));

            Apot[i,j] = repulsion *     fma(Bfield[north,east], a_ij.s0,
                                        fma(Bfield[north,j   ], a_ij.s1,
                                        fma(Bfield[north,west], a_ij.s2,
                                        fma(Bfield[i,west    ], a_ij.s3,
                                        fma(Bfield[south,west], a_ij.s0,
                                        fma(Bfield[south,j   ], a_ij.s1,
                                        fma(Bfield[south,east], a_ij.s2,
                                        fma(Bfield[i,east    ], a_ij.s3, Bfield[i,j]))))))));

        end
    end
    return (Apot,Bpot)
end

fma(a, b, c) = a * b + c

