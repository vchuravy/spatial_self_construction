import Base.exp, Base./, Base.*, Base.-, Base.sum, Base.mod, Base.sin
#import NumericExtensions.fma

immutable Number8{T <: FloatingPoint}
    array :: Array{T,1}
    s0 :: T
    s1 :: T
    s2 :: T
    s3 :: T
    s4 :: T
    s5 :: T
    s6 :: T
    s7 :: T
end

immutable Number4 {T <: FloatingPoint}
    s0 :: T
    s1 :: T
    s2 :: T
    s3 :: T
end

function Number8{T <: FloatingPoint}(elems :: Array{T, 1})
    @assert length(elems) == 8
    Number8(elems, elems[1], elems[2], elems[3], elems[4], elems[5], elems[6], elems[7], elems[8])
end

function Number8{T <: FloatingPoint}(s0 :: T, s1 :: T, s2 :: T, s3 :: T, s4 :: T, s5 :: T, s6 :: T, s7 :: T)
    Number8([s0, s1, s2, s3, s4, s5, s6, s7])
end


exp(x :: Number8) = Number8(exp(x.array))
mod(n :: Number8, r :: Real) = Number8(mod(n.array, r))
sin(n :: Number8) = Number8(sin(n.array))


*(a :: Real, n :: Number8) = Number8(a * n.array)
*(n1 :: Number8, n2 :: Number8) = Number8(n1.array .* n2.array)
-(a :: Real, n :: Number8) = Number8(a - n.array)
-(n :: Number8) = Number8(-n.array)
/(n1 :: Number8, n2 :: Number8) = Number8(n1.array ./ n2.array)
/(n :: Number8, a :: Real) = Number8(n.array ./ a)

sum(n:: Number8) = sum(n.array)

fmod(a,b) = mod(a,b)

const PI = pi
