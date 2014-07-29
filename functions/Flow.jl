function flow{T <: FloatingPoint}(potential :: Matrix{T})
    out = Array(Matrix{T}, size(potential))

    for i in 1:length(out)
        out[i] = similar(potential, (3,3))
    end

    flow!(out, potential)
    return out
end

function flow!{T <: FloatingPoint}(out :: Matrix{Matrix{T}}, potential :: Matrix{T})
    d1, d2 = size(potential)

    p   = similar(potential, (3,3))

    for j in 1:d2
        for i in 1:d1

            get_moore!(p, potential, i, j, d1, d2)

            p = p .- centre(p)

            ###
            # Calculate the probability of flow out of a cell based on the potential difference
            # Normalized by 8

            @simd for k in 1:length(p)
                @inbounds begin
                    p[k] = (- p[k] / (1.0 - exp(p[k]))) / 8.0
                end
            end

            ###
            # Remove NaN
            ###

            @simd for k in 1:length(p)
                @inbounds begin
                    if isnan(p[k])
                        p[k] = 0.125 # 1/8
                    end
                end
            end

            ### Exclusive moore-neighbourhood
            zero_centre!(p)

            copy!(out[i,j], p)
        end
    end
end