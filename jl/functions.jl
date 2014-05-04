import NumericExtensions; const ne = NumericExtensions

include("ocl_utils.jl")
include("potential.jl")
include("diffusion.jl")
include("laplacian.jl")
include("align.jl")
include("calcRow.jl")
include("delta.jl")
include("deltaStep2.jl")
include("area.jl")

diffusion!(buff_Xfield, buff_Xpot, buff_Xlap) = diffusionJl!(buff_Xfield, buff_Xpot, buff_Xlap)
potential!(buff_Xfield, buff_Yfield, buff_Zfield, buff_Xpot, buff_Ypot, repulsion) = potentialJl!(buff_Xfield, buff_Yfield, buff_Zfield, buff_Xpot, buff_Ypot, repulsion)
area!(buff_in, buff_out) = areaJl!(buff_in, buff_out, long_direction)
align!(buff_Xfield, buff_Yfield, buff_OUTfield) = alignJl!(buff_Xfield, buff_Yfield, buff_OUTfield, attractionRate, stepIntegration)
laplacian!(buff_in, buff_out) = LaPlacianJl!(buff_in, buff_out)
calcRow!(buff_Xfield, buff_Yfield, buff_Zfield, buff_Wfield, buff_OUT, x, y, z, w) = calcRowJl!(buff_Xfield, buff_Yfield, buff_Zfield, buff_Wfield, buff_OUT, x, y, z, w)
delta!(buff1, buff2, buff3, buff4, buff5, buff6, buff_out, x1, x2, x3, x4, x5, x6) = deltaJl!(buff1, buff2, buff3, buff4, buff5, buff6, buff_out, x1, x2, x3, x4, x5, x6)
delta2!(buff_D, buff_Xfield, buff_Yfield, buff_L,buff_out, decay) = delta2Jl!(buff_D, buff_Xfield, buff_Yfield, buff_L, buff_out, decay)


smul!(X, buff_in, buff_out) = ne.multiply!(copy!(bouff_out, buff_in), X)
c_add!(in1, in2, out) = ne.add!(copy!(out, in1), in2)

#Julia style smul and add
smul!(out, x) = ne.multiply!(out, x)
c_add!(out, in2) = ne.add!(out, in2)

c_read(source) = copy(source)
c_copy!(target, source) = Base.copy!(target, source)

create{T <: FloatingPoint}(:: Type{T}) = Array(T, fieldResY, fieldResX)
create_n4{T <: FloatingPoint}(:: Type{T}) = Array(Number4{T}, fieldResY, fieldResX)
create_const_n4{T <: FloatingPoint}(x :: T) = fill(Number4(x, x, x, x), fieldResY, fieldResX)