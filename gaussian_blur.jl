###
# Code taken from https://github.com/timholy/Images.jl
###

# The MIT License (MIT)
# Copyright (c) 2012 Timothy E. Holy
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


using Cartesian

# When there are no NaNs, the normalization is separable and hence can be computed far more efficiently
# This speeds the algorithm by approximately twofold
function imfilter_gaussian_no_nans!{T<:FloatingPoint}(data::Array{T}, sigma::Vector; emit_warning = true)
    nd = ndims(data)
    if length(sigma) != nd
        error("Dimensionality mismatch")
    end
    _imfilter_gaussian!(data, sigma, emit_warning=emit_warning)
    denom = Array(Vector{T}, nd)
    for i = 1:nd
        denom[i] = ones(T, size(data, i))
        if sigma[i] > 0
            _imfilter_gaussian!(denom[i], sigma[i:i], emit_warning=false)
        end
    end
    imfgnormalize!(data, denom)
    return data
end

for N in 1:5
    @eval begin
        function imfgnormalize!{T}(data::Array{T,$N}, denom)
            @nextract $N denom denom
            @nloops $N i data begin
                den = one(T)
                @nexprs $N d->(den *= denom_d[i_d])
                (@nref $N data i) /= den
            end
        end
    end
end

function iir_gaussian_coefficients(T::Type, sigma::Number; emit_warning::Bool = true)
    if sigma < 1 && emit_warning
        warn("sigma is too small for accuracy")
    end
    m0 = convert(T,1.16680)
    m1 = convert(T,1.10783)
    m2 = convert(T,1.40586)
    q = convert(T,1.31564*(sqrt(1+0.490811*sigma*sigma) - 1))
    scale = (m0+q)*(m1*m1 + m2*m2  + 2m1*q + q*q)
    B = m0*(m1*m1 + m2*m2)/scale
    B *= B
    # This is what Young et al call b, but in filt() notation would be called a
    a1 = q*(2*m0*m1 + m1*m1 + m2*m2 + (2*m0+4*m1)*q + 3*q*q)/scale
    a2 = -q*q*(m0 + 2m1 + 3q)/scale
    a3 = q*q*q/scale
    a = [-a1,-a2,-a3]
    Mdenom = (1+a1-a2+a3)*(1-a1-a2-a3)*(1+a2+(a1-a3)*a3)
    M = [-a3*a1+1-a3^2-a2      (a3+a1)*(a2+a3*a1)  a3*(a1+a3*a2);
          a1+a3*a2            -(a2-1)*(a2+a3*a1)  -(a3*a1+a3^2+a2-1)*a3;
          a3*a1+a2+a1^2-a2^2   a1*a2+a3*a2^2-a1*a3^2-a3^3-a3*a2+a3  a3*(a1+a3*a2)]/Mdenom;
    return a, B, M
end


function _imfilter_gaussian!{T<:FloatingPoint}(A::Array{T}, sigma::Vector; emit_warning::Bool = true)
    nd = ndims(A)
    szA = [size(A,i) for i = 1:nd]
    strdsA = [stride(A,i) for i = 1:nd]
    for d = 1:nd
        if sigma[d] == 0
            continue
        end
        if size(A, d) < 3
            error("All filtered dimensions must be of size 3 or larger")
        end
        a, B, M = iir_gaussian_coefficients(T, sigma[d], emit_warning=emit_warning)
        a1 = a[1]
        a2 = a[2]
        a3 = a[3]
        n1 = size(A,1)
        keepdims = [false,trues(nd-1)]
        if d == 1
            x = zeros(T, 3)
            vstart = zeros(T, 3)
            szhat = szA[keepdims]
            strdshat = strdsA[keepdims]
            if isempty(szhat)
                szhat = [1]
                strdshat = [1]
            end
            @forcartesian c szhat begin
                coloffset = offset(c, strdshat)
                A[2+coloffset] -= a1*A[1+coloffset]
                A[3+coloffset] -= a1*A[2+coloffset] + a2*A[1+coloffset]
                for i = 4:n1
                    A[i+coloffset] -= a1*A[i-1+coloffset] + a2*A[i-2+coloffset] + a3*A[i-3+coloffset]
                end
                copytail!(x, A, coloffset, 1, n1)
                A_mul_B!(vstart, M, x)
                A[n1+coloffset] = vstart[1]
                A[n1-1+coloffset] -= a1*vstart[1]   + a2*vstart[2] + a3*vstart[3]
                A[n1-2+coloffset] -= a1*A[n1-1+coloffset] + a2*vstart[1] + a3*vstart[2]
                for i = n1-3:-1:1
                    A[i+coloffset] -= a1*A[i+1+coloffset] + a2*A[i+2+coloffset] + a3*A[i+3+coloffset]
                end
            end
        else
            x = Array(T, 3, n1)
            vstart = similar(x)
            keepdims[d] = false
            szhat = szA[keepdims]
            szd = szA[d]
            strdshat = strdsA[keepdims]
            strdd = strdsA[d]
            if isempty(szhat)
                szhat = [1]
                strdshat = [1]
            end
            @forcartesian c szhat begin
                coloffset = offset(c, strdshat)  # offset for the remaining dimensions
                for i = 1:n1 A[i+strdd+coloffset] -= a1*A[i+coloffset] end
                for i = 1:n1 A[i+2strdd+coloffset] -= a1*A[i+strdd+coloffset] + a2*A[i+coloffset] end
                for j = 3:szd-1
                    for i = 1:n1 A[i+j*strdd+coloffset] -= a1*A[i+(j-1)*strdd+coloffset] + a2*A[i+(j-2)*strdd+coloffset] + a3*A[i+(j-3)*strdd+coloffset] end
                end
                copytail!(x, A, coloffset, strdd, szd)
                A_mul_B!(vstart, M, x)
                for i = 1:n1 A[i+(szd-1)*strdd+coloffset] = vstart[1,i] end
                for i = 1:n1 A[i+(szd-2)*strdd+coloffset] -= a1*vstart[1,i]   + a2*vstart[2,i] + a3*vstart[3,i] end
                for i = 1:n1 A[i+(szd-3)*strdd+coloffset] -= a1*A[i+(szd-2)*strdd+coloffset] + a2*vstart[1,i] + a3*vstart[2,i] end
                for j = szd-4:-1:0
                    for i = 1:n1 A[i+j*strdd+coloffset] -= a1*A[i+(j+1)*strdd+coloffset] + a2*A[i+(j+2)*strdd+coloffset] + a3*A[i+(j+3)*strdd+coloffset] end
                end
            end
        end
        for i = 1:length(A)
            A[i] *= B
        end
    end
    A
end

function offset(c::Vector{Int}, strds::Vector{Int})
    o = 0
    for i = 1:length(c)
        o += (c[i]-1)*strds[i]
    end
    o
end

function copytail!(dest, A, coloffset, strd, len)
    for j = 1:3
        for i = 1:size(dest, 2)
            tmp = A[i + coloffset + (len-j)*strd]
            dest[j,i] = tmp
        end
    end
    dest
end