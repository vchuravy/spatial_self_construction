###
# transforms directionality
###

using NumericExtensions

function align(conc, dir, attraction, step) #concentration and direction

    d1,d2 = size(conc)
    newdir = zeros(Float64, d1,d2)

    diff = zeros(Float64, 3,3)
    tmp = zeros(Float64, 3,3)
    potential = zeros(Float64, 3,3)

    for j in 1:d2
        for i in 1:d1

            west  = j == 1  ? d2 : j-1
            east  = j == d2 ? 1  : j+1
            north = i == d1 ? 1  : i+1
            south = i == 1  ? d1 : i-1

            direction = dir[i, j]

            diff[3,1] = direction - dir[north,west]
            diff[3,2] = direction - dir[north,j   ]
            diff[3,3] = direction - dir[north,east]
            diff[2,1] = direction - dir[i,west    ]

            diff[2,2] = 0

            diff[2,3] = direction - dir[i,east    ]
            diff[1,1] = direction - dir[south,west]
            diff[1,2] = direction - dir[south,j   ]
            diff[1,3] = direction - dir[south,east]

            diff = map!(ModFun(), diff, diff, pi)

            potential = map!(Multiply(), potential, diff, 2)
            potential = map!(SinFun(), potential, potential)
            potential = negate!(potential)

            concentration = conc[i,j]

            dtheta = zero(concentration)

            dtheta += concentration * conc[north,west] * potential[3, 1]
            dtheta += concentration * conc[north,j]    * potential[3, 2]
            dtheta += concentration * conc[north,east] * potential[3, 3]
            dtheta += concentration * conc[i,west]     * potential[2, 1]

            dtheta += concentration * conc[i,east]     * potential[2, 3]
            dtheta += concentration * conc[south,west] * potential[1, 1]
            dtheta += concentration * conc[south,j]    * potential[1, 2]
            dtheta += concentration * conc[south,east] * potential[1, 3]


            # update direction for cell
            dtheta = attraction * dtheta / 8 # multiply by attraction constant
            ndir = dir[i,j] + dtheta * step # update direction with stepsize

            newdir[i,j] = mod(ndir, pi)
        end
    end
    return newdir
end