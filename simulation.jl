module Simulation

using ArrayViews
using NumericExtensions

export laplacian, laplacian!,
       flow, flow!,
       diffusion, diffusion!,
       align, align!,
       area, area!,
       potential!,
       align, align!,
       SimulationState

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

centre(A :: Matrix) = A[centre_p...]
zero_centre!{T <: Real}(A :: Matrix{T}) = A[centre_p...] = zero(T)

function translate(i,j)
    ind = 10 - (i + ((j-1) * 3))
    x = rem(ind -1 , 3) + 1
    y = div(ind - x, 3) + 1
    return tuple(x, y)
end

function translate_3x3!{T}(B :: Matrix{T}, A :: Matrix{T})
    for i in 1:3
        for j in 1:3
            B[i,j] = A[translate(i,j)...]
        end
    end
    return B
end

translate_3x3(A) = translate_3x3!(similar(A, (3, 3)), A)

function translated_copy!{T}(B :: Matrix{T}, A :: Matrix{Matrix{T}})
    for i in 1:3
        for j in 1:3
            B[i,j] = A[i,j][translate(i,j)...]
        end
    end
    return B 
end

translated_access{T}(A :: Matrix{Matrix{T}}) = translated_copy!(Array(T, (3, 3)), A)

include("functions/LaPlacian.jl")
include("functions/Flow.jl")
include("functions/Diffusion.jl")
include("functions/Area.jl")
include("functions/Potential.jl")
include("functions/Align.jl")
include("functions/SimulationState.jl")

end # Module