function align{T <: FloatingPoint}(concentration :: Matrix{T}, direction :: Matrix{T}, attraction :: Real, step :: Real)
    out = similar(concentration)

    align!(out, concentration, direction, attraction, step)

    return out
end


function align!{T <: FloatingPoint}(out :: Matrix{T}, concentration :: Matrix{T}, direction :: Matrix{T}, attraction :: Real, step :: Real)
    d1,d2 = size(concentration)

    conc = similar(concentration, (3,3))
    dir = similar(direction, (3,3))

    for j in 1:d2
        for i in 1:d1

            # Getting the data
            get_moore!(conc, concentration, i, j, d1, d2)
            get_moore!(dir, direction, i, j, d1, d2)

            θ = centre(dir)

            #Zero out middle term since we have a exclusive moore-neighbourhood
            zero_centre!(dir)
            zero_centre!(conc)

            # Calculate dθ
            dθ = zero(T)
            for k in 1:9
                 dθ += conc[k] * -sin(2 * mod(θ - dir[k], π))
            end
            dθ = attraction * dθ / 8.0

            #Calculate the new value

            θ += dθ * step

            out[i,j] = mod(θ, π) # make sure we are in [0, π)
        end
    end
end