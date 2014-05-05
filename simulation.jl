module Simulation

using ArrayViews
using NumericExtensions

export la_placian, la_placian!, flow, flow!, diffusion, diffusion!

function get_moore!(out, A, i, j, d1, d2)
    if 1 < i < d1 && 1 < j < d2 # We are away from the edges just use a view
        copy!(out, view(A, i-1 : i+1, j-1 : j+1))
    else # Do it the hard way
        obtain_moore!(out, A, i, j, d1, d2)
    end
end

function obtain_moore!(out, A, x, y, d1, d2)
    for i in -1:1
        u = x + i

        if u == 0
            u = d1
        elseif u > d1 
            u = 1
        end

        for j in -1:1
            v = y + j

            if v == 0
                v = d2
            elseif v > d2
                v = 1
            end

            @inbounds out[i + 2, j + 2] = A[u, v]
        end
    end
end

const centre_p = (2,2)
const north = (1,2)
const south = (3,2)
const west  = (2,1)
const east  = (2,3)
const north_west = (1,1)
const north_east = (1,3)
const south_west = (3,1)
const south_east = (3,3)

centre(A) = A[centre_p...]

const translate = [
	centre_p => centre_p, 
	north => south, 
	south => north, 
	west => east, 
	east => west, 
	north_west => south_east,
	south_east => north_west,
	north_east => south_west,
	south_west => north_east]

include("functions/LaPlacian.jl")
include("functions/Flow.jl")
include("functions/Diffusion.jl")

end