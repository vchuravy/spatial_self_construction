module Simulation

using ArrayViews

export LaPlacian, LaPlacian!

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

include("functions/LaPlacian.jl")

end