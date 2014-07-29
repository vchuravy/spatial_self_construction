function la_placian{T <: FloatingPoint}(Ap :: Matrix{T})
    A_lap = similar(Ap)

    la_placian!(A_lap, Ap)
    return A_lap
end

function la_placian!{T <: FloatingPoint}(A_lap :: Matrix{T}, Ap :: Matrix{T})
    D1, D2 = size(Ap)    
    neighbourhood = similar(Ap, (3,3))
    
    for j in 1:D2
        for i in 1:D1    
            get_moore!(neighbourhood, Ap, i, j, D1, D2)
            
            c = centre(neighbourhood)
            zero_centre!(neighbourhood)
            
            A_lap[i,j] = -c + sum(neighbourhood) / 8.0
        end
    end
end