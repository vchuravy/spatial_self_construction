function align{T <: Real}(concentration :: Matrix{T}, direction :: Matrix{T}, attraction :: Real, step :: Real)
    out = similar(concentration)

    align!(out, concentration, direction, attraction, step)

    return out
end


function align!(out :: Matrix{T}, concentration :: Matrix{T}, direction :: Matrix{T}, attraction :: Real, step :: Real)
    d1,d2 = size(concentration)

    conc = similar(concentration, (3,3))
    dir = similar(direction, (3,3))

    for j in 1:d2
        for i in 1:d1

            # Getting the data
            get_moore!(conc, concentration, i, j, d1, d2)
            get_moore!(dir, direction, i, j, d1, d2)

            old_dir = centre(dir)

            #Zero out middle term since we have a exclusive moore-neighbourhood
            zero_centre!(dir)
            zero_centre!(conc)

            # Calculate dtheta being space constant on dir
            # add!(negate!(dir), old_direction) # central_direction .- dir 
            # map1!(ModFun(), dir, pi) # dir .% PI
            # negate!(map1!(SinFun(), (multiply!(dir,2)))) # -sin((2 .* dir))

            @smid for i in 1:9
                @inbounds  dir[i] = conc[i] * (- sin(2 * mod(old_dir - dir[i], pi)))
            end

            dtheta = attraction * sum(dir) / 8 # attraction * sum(dir .* conc) / 8

            #Calculate the new value

            new_dir = old_dir + dtheta * step

            out[i,j] = mod(new_dir, pi) # make sure we are in [0, pi)
        end
    end
end