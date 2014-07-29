###
# calculate interactive potential based on repulsion
# repulsion on hydrophobe given concentration of hydrophils
###

function potential!{T <: FloatingPoint}(apot ::  Matrix{T}, bpot ::  Matrix{T}, afield :: Matrix{T}, bfield :: Matrix{T}, area :: Array{T, 3}, repulsion :: Real)
    d1, d2 = size(afield)

    r_af = similar(afield, (3,3))
    local_bfield = similar(bfield, (3,3))
    local_area = similar(area, (3,3))

    selfRepulsion = 1*repulsion # making this lower than repulsion allows for neighbours to have relative potential, so increases the chance that the hydrophobe will flow.

    for j in 1:d2
        for i in 1:d1
            a_ij = area[:, i, j] # view

            get_moore!(r_af, afield, i, j, d1, d2)
            get_moore!(local_bfield, bfield, i, j, d1, d2)
            r_ij = selfRepulsion * centre(r_af)

            counter = 0
            for k in 1:3
                for l in 1:3
                    counter += 1
                    if k == 2 && l == 2
                        local_area[k,l] = zero(T)
                        counter = 0
                    else
                        local_area[k,l] = area[counter, translate(k,l)...]
                    end
                end
            end
            
            r_af = multiply!(r_af, repulsion)
            bpot[i,j] = sum(multiply!(r_af, local_area)) + r_ij

            result = centre(local_bfield)
            counter = 0 
            for k in 1:3
                for l in 1:3
                    counter +=1
                    if k == 2 && l == 2
                        counter = 0
                    else
                        result += local_bfield[k,l] * a_ij[counter]
                    end
                end
            end

            apot[i,j] = repulsion * result
        end
    end
end