function flow{T <: Real}(potential :: Matrix{T})
    out = fill(similar(potential, (3,3)), size(potential))

    flow!(out, potential)
    return out
end

function flow!{T <: Real}(out :: Matrix{Matrix{T}}, potential :: Matrix{T})
    d1, d2 = size(potential)

    p   = similar(potential, (3,3))
    ge  = similar(p)

    for j in 1:d2
        for i in 1:d1

            get_moore!(p, potential, i, j, d1, d2)

            centre = p[2,2]

            p .-= centre
            p[2, 2] = 0.0

            ###
            # Calculate the probability of flow out of a cell based on the potential difference

            copy!(ge, p)

            add!(negate!(exp!(p)), 1) # - e^p + 1
            negate!(divide!(ge, p)) # - ge / ( - e ^ p + 1)

            ###
            # Remove NaN and normalize by 8
            ###

            @simd for i in 1:9
                @inbounds begin
                    if isnan(ge[i])
                        ge[i] = 0.125 # 1/8
                    else
                        ge[i] /= 8.0
                    end
                end
            end

            ### Exclusive moore-neighbourhood
            ge[2, 2] = 0.0

           copy!(out[i,j], ge)
        end
    end
end