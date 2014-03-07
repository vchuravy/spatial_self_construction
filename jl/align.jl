###
# transforms directionality
###
import NumericExtensions.negate!, NumericExtensions.negate!, NumericExtensions.multiply!, NumericExtensions.subtract!, NumericExtensions.add!, NumericExtensions.map1!, NumericExtensions.SinFun, NumericExtensions.ModFun
require("jl/ocl_utils.jl")

function alignJl!(Conc :: Matrix, Dir :: Matrix, Newdir :: Matrix, attraction :: Real, step :: Real) #concentration and direction
    d1,d2 = size(Conc)

    diff = zeros(Float64, 3,3)
    tmp = zeros(Float64, 3,3)
    potential = zeros(Float64, 3,3)

    for j in 1:d2
        for i in 1:d1

            west  = j == 1  ? d2 : j-1
            east  = j == d2 ? 1  : j+1
            north = i == d1 ? 1  : i+1
            south = i == 1  ? d1 : i-1

            # Getting the data and storing it in two eight vectors
            direction = Dir[i, j];

            s0 = Conc[south,west];
            s1 = Conc[south,j   ];
            s2 = Conc[south,east];
            s3 = Conc[i    ,east];
            s4 = Conc[north,east];
            s5 = Conc[north,j   ];
            s6 = Conc[north,west];
            s7 = Conc[i    ,west];

            conc = [s0, s1, s2, s3, s4, s5, s6, s7]

            s0 = Dir[south,west];
            s1 = Dir[south,j   ];
            s2 = Dir[south,east];
            s3 = Dir[i    ,east];
            s4 = Dir[north,east];
            s5 = Dir[north,j   ];
            s6 = Dir[north,west];
            s7 = Dir[i    ,west];

            dir = [s0, s1, s2, s3, s4, s5, s6, s7]

            # Calculate
            diff = negate!(dir)
            diff = add!(diff, direction);
            diff = map1!(ModFun(), diff, PI);
            diff = negate!(map1!(SinFun(), (multiply!(diff,2))));

            dtheta = sum(multiply!(diff, conc));

            dtheta = attraction * dtheta / 8;
            ndir = direction + dtheta * step;

            Newdir[i,j] = fmod(ndir,PI);
        end
    end
end