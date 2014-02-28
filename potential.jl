###
# calculate interactive potential based on repulsion
# repulsion on hydrophobe given concentration of hydrophils
###

using NumericExtensions

function potential(Afield :: Matrix, Bfield :: Matrix, direction, repulsion, long)
    d1, d2 = size(Afield)

    # is the anisotropic molecule

    Apotential = zeros(d1,d2)
    Bpotential = zeros(d1,d2)

    short = 1
    selfRepulsion = 1*repulsion # making this lower than repulsion allows for neighbors to have relative potential, so increases the chance that the hydrophobe will flow.

    for j in 1:d2
        for i in 1:d1

            west  = j == 1  ? d2 : j-1
            east  = j == d2 ? 1  : j+1
            north = i == d1 ? 1  : i+1
            south = i == 1  ? d1 : i-1

            Apotential[i,j] += Bfield[i,j] * selfRepulsion

            d  = direction[i,j]

            area1 = area(1, long, short, d)
            area2 = area(3, long, short, d)
            area3 = area(5, long, short, d)
            area4 = area(7, long, short, d)
            area5 = area(9, long, short, d)

            lps = long*short*pi
            a1 = (area2-area1)/lps
            a2 = (area3-area2)/lps
            a3 = (area4-area3)/lps
            a4 = (area5-area4)/lps

            cRepulsion = repulsion * Bfield[i,j]

            Apotential[north,east] += cRepulsion*a1
            Apotential[north,j]    += cRepulsion*a2
            Apotential[north,west] += cRepulsion*a3
            Apotential[i,west]     += cRepulsion*a4

            Apotential[south,west] += cRepulsion*a1
            Apotential[south,j]    += cRepulsion*a2
            Apotential[south,east] += cRepulsion*a3
            Apotential[i,east]     += cRepulsion*a4

            Bpotential[i,j] += repulsion * ( Afield[i,j]
                                                + Afield[north,east]  * a1
                                                + Afield[north,j   ]  * a2
                                                + Afield[north,west]  * a3
                                                + Afield[i,west    ]  * a4
                                                + Afield[south,west]  * a1
                                                + Afield[south,j   ]  * a2
                                                + Afield[south,east]  * a3
                                                + Afield[i,east    ]  * a4
                                            )
        end
    end
    return (Apotential,Bpotential)
end

function area(x, long, short, d)
    p8 = x * pi/8
    long*short/2 * ( p8 - d - atan((short-long)*sin(2*(p8 - d)) / (short+long + (short-long)*cos(2*(p8 - d)))))
end