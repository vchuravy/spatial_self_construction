###
# transforms directionality
###
function align(conc, dir, attraction, step) #concentration and direction

    d1,d2 = size(conc)
    newdir = zeros(d1,d2)
    dtheta = zeros(d1,d2)

    diff = zeros(3,3)
    potential = zeros(3,3)

    for i in 1:d1
        for j in 1:d2

            west  = j == 1  ? d2 : j-1
            east  = j == d2 ? 1  : j+1
            north = i == d1 ? 1  : i+1
            south = i == 1  ? d1 : i-1

            #diff=zeros(3,3) not necessary since it is going to be completly overwritten each step
            #potential=zeros(3,3)

            diff[3,1] = dir[i,j] - dir[north,west]
            diff[3,2] = dir[i,j] - dir[north,j   ]
            diff[3,3] = dir[i,j] - dir[north,east]
            diff[2,1] = dir[i,j] - dir[i,west    ]

            diff[2,2] = 0

            diff[2,3] = dir[i,j] - dir[i,east    ]
            diff[1,1] = dir[i,j] - dir[south,west]
            diff[1,2] = dir[i,j] - dir[south,j   ]
            diff[1,3] = dir[i,j] - dir[south,east]

            diff[diff .> pi] -= pi
            diff[diff .<= 0] += pi

            # for xx in 1:3
            #     for yy in 1:3
            #         potential[yy,xx] = -sin(2*diff[yy,xx])
            #     end
            # end

            potential = -sin(2*diff)

            concentration = conc[i,j]

            ###
            # TODO: Rewrite to be sum(concentration * conc[north:south, west:east] .* potential) if potential[i.j] = 0
            ###

            # dtheta[i, j] += sum(concentration * (conc[north:south, west:east] .* potential)

            dtheta[i, j] += concentration * conc[north,west] * potential[3, 1]
            dtheta[i, j] += concentration * conc[north,j]    * potential[3, 2]
            dtheta[i, j] += concentration * conc[north,east] * potential[3, 3]
            dtheta[i, j] += concentration * conc[i,west]     * potential[2, 1]

            dtheta[i, j] += concentration * conc[i,east]     * potential[2, 3]
            dtheta[i, j] += concentration * conc[south,west] * potential[1, 1]
            dtheta[i, j] += concentration * conc[south,j]    * potential[1, 2]
            dtheta[i, j] += concentration * conc[south,east] * potential[1, 3]


            # update direction for cell
            dtheta[i,j] = attraction * dtheta[i,j] / 8 # multiply by attraction constant
            newdir[i,j] = dir[i,j] + dtheta[i,j] * step # update direction with stepsize

            while newdir[i,j]>pi
                newdir[i,j] -= pi
            end
            while newdir[i,j] <= 0
                newdir[i,j] += pi
            end
        end
    end
    return newdir
end