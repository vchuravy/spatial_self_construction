using NumericExtensions

using NumericExtensions

import Base.exp, Base./, Base.*, Base.-, Base.sum

immutable Number8 {T <: FloatingPoint}
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

function toArray(n :: Number8)
    return [n.s0, n.s1, n.s2, n.s3, n.s4, n.s5, n.s6, n.s7]
end

function exp(n :: Number8)
    an = toArray(n)
    exp!(an)
    return Number8(an...)
end

*(a :: Real, n :: Number8) = Number8(multiply!(toArray(n), a)...)
-(a :: Real, n :: Number8) = Number8((a - toArray(n))...)
/(n1 :: Number8, n2 :: Number8) =Number8(divide!(toArray(n1), toArray(n2))...)
/(n1 :: Number8, a :: Real) =Number8(divide!(toArray(n1), a)...)


function sum(n:: Number8)
    return sum(toArray(n))
end

fma(a, b, c) = a * b + c