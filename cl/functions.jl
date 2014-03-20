import OpenCL: Buffer, CmdQueue, Context, Program, Kernel, call, build!

include("potentialCL.jl")
include("diffusionCL.jl")
include("laplacianCL.jl")
include("addCL.jl")
include("alignCL.jl")
include("calcRowCL.jl")
include("deltaCL.jl")
include("smulCL.jl")
include("deltaStep2CL.jl")
include("areaCL.jl")

### Define Module variables
ctx = nothing
NT = nothing
queue = nothing

potentialProgram = nothing
areaProgram = nothing
diffusionProgram = nothing
addProgram = nothing
alignProgram = nothing
rowProgram = nothing
deltaProgram = nothing
smulProgram = nothing
delta2Program = nothing
laplacianProgram = nothing

function createComputeContext{T <: FloatingPoint}(:: Type{T}, c, q)
    global ctx = c
    global NT = T
    global queue = q

    global potentialProgram = Program(ctx, source=getPotentialKernel(T)) |> build!
    global areaProgram = Program(ctx, source=getAreaKernel(T)) |> build!
    global diffusionProgram = Program(ctx, source=getDiffusionKernel(T)) |> build!
    global addProgram = Program(ctx, source=getAddKernel(T)) |> build!
    global alignProgram = Program(ctx, source=getAlignKernel(T)) |> build!
    global rowProgram = Program(ctx, source=getRowKernel(T)) |> build!
    global deltaProgram = Program(ctx, source=getDeltaKernel(T)) |> build!
    global smulProgram = Program(ctx, source=getSMulKernel(T)) |> build!
    global delta2Program = Program(ctx, source=getDelta2Kernel(T)) |> build!
    global laplacianProgram = Program(ctx, source=getLaplacianKernel(T)) |> build!
end

function destroyComputeContext()
    global ctx = nothing
    global queue = nothing
    global fieldResY = nothing
    global fieldResX = nothing

    global potentialProgram = nothing
    global areaProgram = nothing
    global diffusionProgram = nothing
    global addProgram = nothing
    global alignProgram = nothing
    global rowProgram = nothing
    global deltaProgram = nothing
    global smulProgram = nothing
    global delta2Program = nothing
    global laplacianProgram = nothing
end

###
# Define clfunctions
# IMPORTANT: Julia convention is func!(X, Y) overrides X with the operation func(X,Y)
# For the OCL functions the target is always the LAST BUFFER.

diffusion!(buff_Xfield, buff_Xpot, buff_Xlap) = diffusionCL!(buff_Xfield, buff_Xpot, buff_Xlap, fieldResY, fieldResX, ctx, queue, diffusionProgram, NT)
potential!(buff_Xfield, buff_Yfield, buff_Zfield, buff_Xpot, buff_Ypot, repulsion) = potentialCL!(buff_Xfield, buff_Yfield, buff_Zfield, buff_Xpot, buff_Ypot, convert(NT, repulsion), fieldResY, fieldResX, ctx, queue, potentialProgram)
area!(buff_in, buff_out) = areaCL!(buff_in, buff_out, long_direction, fieldResY, fieldResX, ctx, queue, areaProgram, NT)
align!(buff_Xfield, buff_Yfield, buff_OUTfield) = alignCL!(buff_Xfield, buff_Yfield, buff_OUTfield, convert(NT, attractionRate), convert(NT, stepIntegration), fieldResY, fieldResX, ctx, queue, alignProgram)
calcRow!(buff_Xfield, buff_Yfield, buff_Zfield, buff_Wfield, buff_OUT, x, y, z, w) = calcRowCL!(buff_Xfield, buff_Yfield, buff_Zfield, buff_Wfield, buff_OUT, x, y, z, w, fieldResY, fieldResX, ctx, queue, rowProgram, NT)
laplacian!(buff_in, buff_out) = laplacianCL!(buff_in, buff_out, fieldResY, fieldResX, ctx, queue, laplacianProgram)
delta!(buff1, buff2, buff3, buff4, buff5, buff6, buff_out, x1, x2, x3, x4, x5, x6) = deltaCL!(buff1, buff2, buff3, buff4, buff5, buff6, buff_out, x1, x2, x3, x4, x5, x6, fieldResY, fieldResX, ctx, queue, deltaProgram, NT)
delta2!(buff_D, buff_Xfield, buff_Yfield, buff_L,buff_out, decay) = delta2CL!(buff_D, buff_Xfield, buff_Yfield, buff_L, buff_out, decay, fieldResY, fieldResX, ctx, queue, delta2Program, NT)

smul!(X, buff_in, buff_out) =  smulCL!(X, buff_in, buff_out, fieldResY, fieldResX, ctx, queue, smulProgram)
c_add!(buff_in1, buff_in2, buff_out) = addCL!(buff_in1, buff_in2, buff_out, fieldResY, fieldResX, ctx, queue, addProgram)

#Julia style smul and add
smul!(buff, X) =  smulCL!(X, buff, buff, fieldResY, fieldResX, ctx, queue, smulProgram, NT)
c_add!(buff_out, buff_in) = addCL!(buff_out, buff_in, buff_out, fieldResY, fieldResX, ctx, queue, addProgram)

# Standard Julia Convention target first
c_copy!(target, source) = OpenCL.copy!(queue, target, source)
c_read(source) = OpenCL.read(queue, source)

create{T <: FloatingPoint}(:: Type{T}) = Buffer(T, ctx, :rw, fieldResX * fieldResY)
create_n4{T <: FloatingPoint}(:: Type{T}) = Buffer(T, ctx, :rw, fieldResX * fieldResY * 4)
function create_const_n4{T <: FloatingPoint}(x :: T)
    vals = fill(x, fieldResY * fieldResX * 4)
    buff = Buffer(T, ctx, :r, fieldResX * fieldResY * 4)
    OpenCL.copy!(queue, buff, vals)
    return buff
end