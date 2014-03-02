###
# Taken from http://csousahome.wordpress.com/2014/02/27/defining-julia-element-wise-exponentiation-of-matrices-for-integer-exponents/
###
import Base.(.^)

function .^{T<:FloatingPoint, N}(A::Array{T, N}, x::Integer)
    if abs(x) > 42 # the "Answer to the Ultimate Question of Life, the Universe, and Everything"
        A.^float(x)
    elseif x > 1
        B = similar(A)
        @inbounds for i in 1:length(A)
            B[i] = A[i]
            for k in 1:x-1 B[i] *= A[i] end
        end
        B
    elseif x < 0
        B = similar(A)
        @inbounds for i in 1:length(A)
            B[i] = one(T)
            for k in 1:abs(x) B[i] *= A[i] end
            B[i] \= one(T)
        end
        B
    elseif x == 1
        copy(A)
    else   #  x == 0
        ones(A)
    end
end