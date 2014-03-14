###
# Main
###

using MAT
using Datetime
using Distributions
import NumericExtensions.sumabs
import NumericExtensions
const ne = NumericExtensions
import OpenCL
const cl = OpenCL

include("config.jl")
include("drawcircle.jl")
include("jl/potential.jl")
include("jl/diffusion.jl")
include("jl/laplacian.jl")
include("jl/align.jl")
include("jl/calcRow.jl")
include("jl/delta.jl")
include("jl/deltaStep2.jl")
include("jl/area.jl")
include("cl/potentialCL.jl")
include("cl/diffusionCL.jl")
include("cl/laplacianCL.jl")
include("cl/addCL.jl")
include("cl/alignCL.jl")
include("cl/calcRowCL.jl")
include("cl/deltaCL.jl")
include("cl/smulCL.jl")
include("cl/deltaStep2CL.jl")
include("cl/areaCL.jl")
include("gaussian_blur.jl")

###
# set up initial configuration
###

function determineCapabilities(allow32Bit = false, forceJuliaImpl = false)
    try
        if forceJuliaImpl
            error("forced usage of Julia implementation.")
        end
        device = deviceWith64Bit()
        if device != nothing
            ctx = cl.Context([device])
            queue = CmdQueue(ctx)
            return (true, true, ctx, queue)
        else
            warn("No OpenCL device with Float64 support found!")
            if allow32Bit
                warn("Searching for device with Float32 support.")
                device, ctx, queue = cl.create_compute_context()
                return (false, true, ctx, queue)
            else
                throw(Exception())
            end
        end
    catch e
        println("Got exception: $e")
        warn("OpenCL is not supported falling back to Julia computation")
        return (true, false, nothing, nothing)
    end
end

function deviceWith64Bit()
    amd = "cl_amd_fp64"
    khr = "cl_khr_fp64"
    for device in cl.devices(:gpu)
        ext = cl.info(device, :extensions)
        if (khr in ext) || (amd in ext)
            return device
        end
    end
    for device in cl.devices(:cpu)
        ext = cl.info(device, :extensions)
        if (khr in ext) || (amd in ext)
            warn("Only found 64bit support on the CPU")
            return device
        end
    end
    return nothing
end

function apply_punch_down!(A, x0, y0, a, b)
			    d1, d2 = size(A)
			    @assert x0 in 1:d1
			    @assert y0 in 1:d2
			    #dist(x, y) = sqrt((x - x0)^2 + (y-y0)^2)
			    #dist(x, y) = abs(x-x0) + abs(y-y0)
			    dist(x, y) = floor(sqrt((x - x0)^2 + (y-y0)^2))
			    #punch(d) = 1- sech(1/b * d) ^ a
			    punch(x)=a*exp(-x^2/(2*b^2))+1
			    for j in 1:d2
			        for i in 1:d1
			            d = dist(i, j)
			            A[i,j] *= punch(d)
			        end
			    end
			end

macro at_proc(p, ex)
   quote
           remotecall( $p, ()->eval(Main,$(Expr(:quote,ex))))
   end
end

function main(config=Dict(), disturbances=Dict(), cluster=false; enableVis :: Bool = false, enableDirFieldVis = false, fileName = "", loadTime = -1, debug = false, allow32Bit = false, forceJuliaImpl = false)
    useVis = enableVis && ((length(procs()) == 1) || (!(myid() in workers()))) && !cluster
    ###
    # Prepare GPU
    ###
    const P64BIT, USECL, ctx, queue = determineCapabilities(allow32Bit, forceJuliaImpl)

    ###
    # Prepare GUI
    ###

    guiproc = if useVis && (length(procs()) <= 1)
        first(addprocs(1))
    elseif useVis
        last(procs())
    else
        -1
    end

    rref = if useVis
       @at_proc guiproc using PyPlot
    end

   value = simulation(cluster, useVis, enableDirFieldVis, fileName, loadTime, USECL, P64BIT ? Float64 : Float32, debug, config, disturbances, guiproc, ctx, queue, rref)
   gc()
   return value
end

function simulation{T <: FloatingPoint}(cluster, enableVis, enableDirFieldVis, fileName, loadTime, USECL, :: Type{T}, testCL :: Bool, config :: Dict, disturbances :: Dict, guiproc :: Int, ctx, queue, gui_rref)
worker = (length(procs()) > 1) && (myid() in workers())


# initialize membrane fields
Afield = nothing
Mfield = nothing
Ffield = nothing
Wfield = nothing
directionfield = nothing

loadConfig(baseConfig)

fileConfig = if fileName != "" && !cluster
    matread("data/$(fileName).mat")
else
    Dict()
end

loadConfig(fileConfig, dataVars)
loadConfig(config, dataVars)
updateDependentValues()

if loadTime >= 0
    if cluster && checkVars(historyVars, collect(keys(config)))
        Wfield = config["history_W"][:,:,loadTime]
        Afield = config["history_A"][:,:,loadTime]
        Mfield = config["history_M"][:,:,loadTime]
        Ffield = config["history_F"][:,:,loadTime]
        directionfield = config["history_dir"][:,:,loadTime]
    elseif checkVars(historyVars, collect(keys(fileConfig)))
        Wfield = fileConfig["history_W"][:,:,loadTime]
        Afield = fileConfig["history_A"][:,:,loadTime]
        Mfield = fileConfig["history_M"][:,:,loadTime]
        Ffield = fileConfig["history_F"][:,:,loadTime]
        directionfield = fileConfig["history_dir"][:,:,loadTime]
    else
        error("Could not find historyVars despite being given a loadTime of $loadTime")
    end
else
    Afield = zeros(T, fieldResY, fieldResX)
    Mfield = zeros(T, fieldResY, fieldResX)
    Ffield = avgF * ones(T, fieldResY, fieldResX)
    Wfield = ones(T, fieldResY, fieldResX)
    directionfield = pi/4 * ones(T, fieldResY, fieldResX)

    # draw membrane circle
    M_circ = zeros(T, fieldResY, fieldResX)

    for r in mR:0.01:(mR+mT-1)
        drawcircle!(M_circ, xc, yc, r)
    end
    Mfield += avgM * M_circ + 0.01* rand(fieldResY, fieldResX)

    # fill in with autocatalyst
    A_circ = zeros(T, fieldResY, fieldResX)

    for r in 0:0.01:mR
        drawcircle!(A_circ, xc, yc, r)
    end
    Afield += avgA * A_circ

    Wfield -= (Mfield + Afield)

    ###
    # directionality initialization
    ###

    # make directionality in a ring
    for q1 in 1:fieldResX
        for q2 in 1:fieldResY
            directionfield[q2,q1] = atan((q2-yc)/(q1-xc))-pi/2

            # correct for 180 degrees
            while directionfield[q2, q1] > pi
                 directionfield[q2, q1] -= pi
            end
            while directionfield[q2, q1] <= 0 # ??? < 0
                 directionfield[q2, q1] += pi
            end
        end
    end

    directionfield[isnan(directionfield)] = 0
end

###
# Data storing
###

# set times at which field activities are stored
tStoreFields = 1:stepVisualization:timeTotal

# create 3d matrices to store field activities
history_A = zeros(T, fieldResY, fieldResX, length(tStoreFields))
history_F = zeros(T, fieldResY, fieldResX, length(tStoreFields))
history_T = zeros(T, fieldResY, fieldResX, length(tStoreFields))
history_M = zeros(T, fieldResY, fieldResX, length(tStoreFields))
history_M_pot = zeros(T, fieldResY, fieldResX, length(tStoreFields))
history_W = zeros(T, fieldResY, fieldResX, length(tStoreFields))
history_dir = zeros(T, fieldResY, fieldResX, length(tStoreFields))

# index of the current position in the history matrices
iHistory = 1

#vectors to save global concentrations across time

vecL = iround(timeTotal / stepIntegration)
Avec = zeros(T, vecL)
Fvec = zeros(T, vecL)
Mvec = zeros(T, vecL)
Wvec = zeros(T, vecL)
DAvec = zeros(T, vecL)
DMvec = zeros(T, vecL)

###
# Prepare simulation
###

#define scaled diffusion
diffA = diffusionA*fieldRes/fieldSize
diffF = diffusionF*fieldRes/fieldSize
diffM = diffusionM*fieldRes/fieldSize
diffW = diffusionW*fieldRes/fieldSize

tx = [0:stepIntegration:timeTotal-stepIntegration]

dF = zeros(T, fieldResY, fieldResX)

if USECL
###
# CL prepare programs.

potentialProgram = cl.Program(ctx, source=getPotentialKernel(T)) |> cl.build!
areaProgram = cl.Program(ctx, source=getAreaKernel(T)) |> cl.build!
diffusionProgram = cl.Program(ctx, source=getDiffusionKernel(T)) |> cl.build!
addProgram = cl.Program(ctx, source=getAddKernel(T)) |> cl.build!
alignProgram = cl.Program(ctx, source=getAlignKernel(T)) |> cl.build!
rowProgram = cl.Program(ctx, source=getRowKernel(T)) |> cl.build!
deltaProgram = cl.Program(ctx, source=getDeltaKernel(T)) |> cl.build!
smulProgram = cl.Program(ctx, source=getSMulKernel(T)) |> cl.build!
delta2Program = cl.Program(ctx, source=getDelta2Kernel(T)) |> cl.build!
laplacianProgram = cl.Program(ctx, source=getLaplacianKernel(T)) |> cl.build!

###
# Define clfunctions
# IMPORTANT: Julia convention is func!(X, Y) overrides X with the operation func(X,Y)
# For the OCL functions the target is always the LAST BUFFER.

diffusion!(buff_Xfield, buff_Xpot, buff_Xlap) = diffusionCL!(buff_Xfield, buff_Xpot, buff_Xlap, fieldResY, fieldResX, ctx, queue, diffusionProgram)
potential!(buff_Xfield, buff_Yfield, buff_Zfield, buff_Xpot, buff_Ypot, repulsion) = potentialCL!(buff_Xfield, buff_Yfield, buff_Zfield, buff_Xpot, buff_Ypot, repulsion, fieldResY, fieldResX, ctx, queue, potentialProgram)
area!(buff_in, buff_out) = areaCL!(buff_in, buff_out, long_direction, fieldResY, fieldResX, ctx, queue, areaProgram)
align!(buff_Xfield, buff_Yfield, buff_OUTfield) = alignCL!(buff_Xfield, buff_Yfield, buff_OUTfield, attractionRate, stepIntegration, fieldResY, fieldResX, ctx, queue, alignProgram)
calcRow!(buff_Xfield, buff_Yfield, buff_Zfield, buff_Wfield, buff_OUT, x, y, z, w) = calcRowCL!(buff_Xfield, buff_Yfield, buff_Zfield, buff_Wfield, buff_OUT, x, y, z, w, fieldResY, fieldResX, ctx, queue, rowProgram)
laplacian!(buff_in, buff_out) = laplacianCL!(buff_in, buff_out, fieldResY, fieldResX, ctx, queue, laplacianProgram )
delta!(buff1, buff2, buff3, buff4, buff5, buff6, buff_out, x1, x2, x3, x4, x5, x6) = deltaCL!(buff1, buff2, buff3, buff4, buff5, buff6, buff_out, x1, x2, x3, x4, x5, x6, fieldResY, fieldResX, ctx, queue, deltaProgram)
delta2!(buff_D, buff_Xfield, buff_Yfield, buff_L,buff_out, decay) = delta2CL!(buff_D, buff_Xfield, buff_Yfield, buff_L, buff_out, decay, fieldResY, fieldResX, ctx, queue, delta2Program)

smul!(X, buff_in, buff_out) =  smulCL!(X, buff_in, buff_out, fieldResY, fieldResX, ctx, queue, smulProgram)
add!(buff_in1, buff_in2, buff_out) = addCL!(buff_in1, buff_in2, buff_out, fieldResY, fieldResX, ctx, queue, addProgram)

#Julia style smul and add
smul!(buff, X) =  smulCL!(X, buff, buff, fieldResY, fieldResX, ctx, queue, smulProgram)
add!(buff_out, buff_in) = addCL!(buff_out, buff_in, buff_out, fieldResY, fieldResX, ctx, queue, addProgram)

# Standard Julia Convention target first
copy!(target, source) = cl.copy!(queue, target, source)
read(source) = cl.read(queue, source)

create() = cl.Buffer(T, ctx, :rw, fieldResX * fieldResY)
create_n4() = cl.Buffer(T, ctx, :rw, fieldResX * fieldResY * 4)
function create_const_n4(x :: T)
    vals = fill(x, fieldResY * fieldResX * 4)
    buff = cl.Buffer(T, ctx, :r, fieldResX * fieldResY * 4)
    cl.copy!(queue, buff, vals)
    return buff
end
else
    diffusion!(buff_Xfield, buff_Xpot, buff_Xlap) = diffusionJl!(buff_Xfield, buff_Xpot, buff_Xlap)
    potential!(buff_Xfield, buff_Yfield, buff_Zfield, buff_Xpot, buff_Ypot, repulsion) = potentialJl!(buff_Xfield, buff_Yfield, buff_Zfield, buff_Xpot, buff_Ypot, repulsion)
    area!(buff_in, buff_out) = areaJl!(buff_in, buff_out, long_direction)
    align!(buff_Xfield, buff_Yfield, buff_OUTfield) = alignJl!(buff_Xfield, buff_Yfield, buff_OUTfield, attractionRate, stepIntegration)
    laplacian!(buff_in, buff_out) = LaPlacianJl!(buff_in, buff_out)
    calcRow!(buff_Xfield, buff_Yfield, buff_Zfield, buff_Wfield, buff_OUT, x, y, z, w) = calcRowJl!(buff_Xfield, buff_Yfield, buff_Zfield, buff_Wfield, buff_OUT, x, y, z, w)
    delta!(buff1, buff2, buff3, buff4, buff5, buff6, buff_out, x1, x2, x3, x4, x5, x6) = deltaJl!(buff1, buff2, buff3, buff4, buff5, buff6, buff_out, x1, x2, x3, x4, x5, x6)
    delta2!(buff_D, buff_Xfield, buff_Yfield, buff_L,buff_out, decay) = delta2Jl!(buff_D, buff_Xfield, buff_Yfield, buff_L, buff_out, decay)


    smul!(X, buff_in, buff_out) = ne.multiply!(copy!(bouff_out, buff_in), X)
    add!(in1, in2, out) = ne.add!(copy!(out, in1), in2)

    #Julia style smul and add
    smul!(out, x) = ne.multiply!(out, x)
    add!(out, in2) = ne.add!(out, in2)

    read(source) = copy(source)
    copy!(target, source) = Base.copy!(target, source)

    create() = Array(T, fieldResY, fieldResX)
    create_n4() = Array(Number4{T}, fieldResY, fieldResX)
    create_const_n4(x :: T) = fill(Number4(x, x, x, x), fieldResY, fieldResX)
end

# create temp arrays
buff_mpot = create()
buff_mpot1 = create()
buff_mpot2 = create()
buff_wpot = create()
buff_apot = create()
buff_apot1 = create()


buff_mfield = create()
buff_wfield = create()
buff_afield = create()
buff_dfield = create()
buff_ndfield = create()
buff_ffield = create()

buff_wlap = create()
buff_alap = create()
buff_mlap = create()
buff_flap = create()

buff_row1 = create()
buff_row2 = create()
buff_row3 = create()
buff_row4 = create()
buff_row5 = create()
buff_row6 = create()

buff_dW = create()
buff_dA = create()
buff_dM = create()
buff_dF = create()

buff_temp = create()
buff_area = create_n4()
buff_const_area = create_const_n4(convert(T, 1.0/8.0))


# Make sure that the fields are in the correct DataFormat
Mfield = convert(Array{T}, Mfield)
Afield = convert(Array{T}, Afield)
Wfield = convert(Array{T}, Wfield)
Ffield = convert(Array{T}, Ffield)
directionfield = convert(Array{T}, directionfield)

# Stability and structure
stable = false
timeToStable = -1

old_Mfield = copy(Mfield)
old_Ffield = copy(Ffield)
old_Wfield = copy(Wfield)
old_Afield = copy(Afield)
old_directionfield = copy(directionfield)

###
# Simulation
###

t = 1

meanMField = mean(Mfield)
meanAField = mean(Afield)

###
# Disturbance steps
###
tT = timeTotal

println("Starting loop")
while (t <= tT) && !isnan(meanMField) && !isnan(meanAField)
	if t in keys(disturbances)
		val = disturbances[t]
		method = first(val)
		args = val[2:end]
		if method == :global && length(args) == 2
			(mu, sig) = args
			Mfield += rand(Normal(mu, sig), (40, 40))
			Afield += rand(Normal(mu, sig), (40, 40))
			Ffield += rand(Normal(mu, sig), (40, 40))
			Wfield += rand(Normal(mu, sig), (40, 40))
			directionfield += rand(Normal(mu, sig), (40, 40))

			# Normalize
			Mfield[Mfield .< 0] = 0
			Afield[Afield .< 0] = 0
			Ffield[Ffield .< 0] = 0
			Wfield[Wfield .< 0] = 0

			map!(ModFun(), directionfield, pi)
		elseif method == :punch_local
			(x, y, alpha, beta) = args
			if x == 0 || y == 0
				warn("Taking a membrane point at random")
				lin_idx = sample(find((s) -> s > 0.5, Mfield))
				y = lin_idx % fieldResY
				x = lin_idx - (y*fieldResY)
			end
			apply_punch_down!(Mfield, x, y, alpha, beta)
			apply_punch_down!(Afield, x, y, alpha, beta)
			apply_punch_down!(Ffield, x, y, alpha, beta)
			apply_punch_down!(Wfield, x, y, alpha, beta)
			# apply_punch_down!(directionfield, x, y, alpha, beta)
		elseif method == :gaussian_blur && length(args) == 1
			sigma = args[1]
			Images.imfilter_gaussian_no_nans!(Mfield, [sigma,sigma])
			Images.imfilter_gaussian_no_nans!(Afield, [sigma,sigma])
			Images.imfilter_gaussian_no_nans!(Ffield, [sigma,sigma])
			Images.imfilter_gaussian_no_nans!(Wfield, [sigma,sigma])
			# directionfield = imfilter_gaussian(directionfield, [sigma,sigma])
		elseif method == :tear_membrane && length(args) == 1
			tearsize = args[1]

		    tS = iround(fieldRes/2+tearSize*fieldRes/fieldSize)
		    t0 = iround(fieldRes/2)
		    Mfield[fieldRes/2:fieldRes,t0:tS] = 0
		elseif method == :tear_dfield && length(args) == 1
			tearsize = args[1]

		    tS = iround(fieldRes/2+tearSize*fieldRes/fieldSize)
		    t0 = iround(fieldRes/2)
		    directionfield[fieldRes/2:fieldRes,t0:tS] = pi .* rand(iround(fieldRes-fieldRes/2+1),tS-t0+1)
		elseif method == :tear && length(args) == 1
			tearsize = args[1]

		    tS = iround(fieldRes/2+tearSize*fieldRes/fieldSize)
		    t0 = iround(fieldRes/2)

		    Mfield[fieldRes/2:fieldRes,t0:tS] = 0
		    directionfield[fieldRes/2:fieldRes,t0:tS] = pi .* rand(iround(fieldRes-fieldRes/2+1),tS-t0+1)
		else
			warn("Don't know how to handle $method with arguments $args, carry on.")
		end
	end

    ###
    # Pushing the fields on the GPU
    ###

    copy!(buff_mfield, Mfield)
    copy!(buff_wfield, Wfield)
    copy!(buff_afield, Afield)
    copy!(buff_ffield, Ffield)
    copy!(buff_dfield, directionfield)

    ###
    # calculate potential based on repulsion
    ###
    area!(buff_dfield, buff_area)

    potential!(buff_mfield, buff_wfield, buff_area, buff_mpot, buff_wpot, MW_repulsion)
    potential!(buff_mfield, buff_afield, buff_area, buff_mpot1, buff_apot, MA_repulsion)
    potential!(buff_mfield, buff_mfield, buff_area, buff_mpot2, buff_temp, MM_repulsion)

    potential!(buff_afield, buff_afield, buff_const_area, buff_apot1, buff_temp, AA_repulsion)

    add!(buff_apot, buff_apot1)
    add!(buff_mpot, buff_mpot1)
    add!(buff_mpot, buff_mpot2)

    ###
    # move molecules and update directionality
    ###

    diffusion!(buff_mfield, buff_mpot, buff_mlap)
    smul!(buff_mlap, diffM)

    diffusion!(buff_wfield, buff_wpot, buff_wlap)
    smul!(buff_wlap, diffW)

    diffusion!(buff_afield, buff_apot, buff_alap)
    smul!(buff_alap, diffA)


    # Laplacian for diffusion
    # Todo calculate in on GPU to be coherent and to minimize data transfers.
    laplacian!(buff_ffield, buff_flap)
    smul!(buff_flap, diffF)

    # update direction field based on alignment
    align!(buff_mfield, buff_dfield, buff_ndfield)
    copy!(directionfield, buff_ndfield)

    ###
    # reactions
    ##

    calcRow!(buff_mfield, buff_afield, buff_ffield, buff_wfield, buff_row1, m11, a11, f11, w11)
    calcRow!(buff_mfield, buff_afield, buff_ffield, buff_wfield, buff_row2, m12, a12, f12, w12)
    calcRow!(buff_mfield, buff_afield, buff_ffield, buff_wfield, buff_row3, m21, a21, f21, w21)
    calcRow!(buff_mfield, buff_afield, buff_ffield, buff_wfield, buff_row4, m22, a22, f22, w22)
    calcRow!(buff_mfield, buff_afield, buff_ffield, buff_wfield, buff_row5, m31, a31, f31, w31)
    calcRow!(buff_mfield, buff_afield, buff_ffield, buff_wfield, buff_row6, m32, a32, f32, w32)

    ##
    # Calculate dA
    ##

    delta!(buff_row1, buff_row2, buff_row3, buff_row4, buff_row5, buff_row6, buff_dA,
            fsm(a12, a11, kf1), fsm(a11, a12, kb1), fsm(a22, a21, kf2), fsm(a21, a22, kb2), fsm(a32, a31, kf3), fsm(a31, a32, kb3))
    delta2!(buff_dA, buff_mfield, buff_afield, buff_alap, buff_dA, decayA)

    ##
    # Calculate dM
    ##

    delta!(buff_row1, buff_row2, buff_row3, buff_row4, buff_row5, buff_row6, buff_dM,
            fsm(m12, m11, kf1), fsm(m11, m12, kb1), fsm(m22, m21, kf2), fsm(m21, m22, kb2), fsm(m32, m31, kf3), fsm(m31, m32, kb3))
    delta2!(buff_dM, buff_afield, buff_mfield, buff_mlap, buff_dM, decayM)

    ##
    # Calculate dW
    ##

    delta!(buff_row1, buff_row2, buff_row3, buff_row4, buff_row5, buff_row6, buff_dW,
            fsm(w12, w11, kf1), fsm(w11, w12, kb1), fsm(w22, w21, kf2), fsm(w21, w22, kb2), fsm(w32, w31, kf3), fsm(w31, w32, kb3))

    add!(buff_dW, buff_wlap)

    ##
    # Calculate dF
    ##

    delta!(buff_row1, buff_row2, buff_row3, buff_row4, buff_row5, buff_row6, buff_dF,
            fsm(f12, f11, kf1), fsm(f11, f12, kb1), fsm(f22, f21, kf2), fsm(f21, f22, kb2), fsm(f32, f31, kf3), fsm(f31, f32, kb3))

    add!(buff_dF, buff_flap)

    copy!(dF, buff_dF)

    dF[FrefillBinMask] += flowRateF * (saturationF .- Ffield[FrefillBinMask])

    dT = 0

    # update values
    smul!(buff_dA, stepIntegration)
    add!(buff_afield, buff_dA)
    copy!(Afield, buff_afield)

    Ffield += dF * stepIntegration

    smul!(buff_dM, stepIntegration)
    add!(buff_mfield, buff_dM)
    copy!(Mfield, buff_mfield)

    smul!(buff_dW, stepIntegration)
    add!(buff_wfield, buff_dW)
    copy!(Wfield, buff_wfield)

    # calulate stability criteria

    sumabs_dA = sumabs(read(buff_dA))
    sumabs_dM = sumabs(read(buff_dM))
    sumabs_dF = sumabs(read(buff_dF))
    sumabs_dW = sumabs(read(buff_dW))

    stableCondition = (sumabs_dA < epsilon) && (sumabs_dM < epsilon) #&& (sumabs_dF < epsilon) && (sumabs_dW < epsilon)

    if !stable && stableCondition
        println("reached a stable config after $t")
        stable = true
        tT = t + stableTime
        timeToStable = t

        #increase storage
        if(tT > timeTotal)
            z = zeros(T, iceil((tT-timeTotal)/stepIntegration))
            append!(Avec, z)
            append!(Fvec, z)
            append!(Mvec, z)
            append!(Wvec, z)
            append!(DAvec, z)
            append!(DMvec, z)
        end
    elseif stable && !stableCondition
        stable = false
        warn("Lost stability")
    end

    #save values for visualization
    meanAField = mean(Afield)
    meanMField = mean(Mfield)

    Avec[iround(t/stepIntegration)] = meanAField
    Fvec[iround(t/stepIntegration)] = mean(Ffield)
    Mvec[iround(t/stepIntegration)] = meanMField
    Wvec[iround(t/stepIntegration)] = mean(Wfield)
    DAvec[iround(t/stepIntegration)] = sumabs_dA
    DMvec[iround(t/stepIntegration)] = sumabs_dM

    if t in tStoreFields
      history_A[:, :, iHistory] = Afield
      history_F[:, :, iHistory] = Ffield
      history_M[:, :, iHistory] = Mfield
      history_M_pot[:, :, iHistory] = read(buff_mpot)
      history_W[:, :, iHistory] = Wfield
      history_dir[:, :, iHistory] = directionfield
      iHistory += 1
    end

    if enableVis && (t % visInterval == 0)
        if !isready(gui_rref)
            println()
            println("Waiting for GUI to be initialized.")
            @time wait(gui_rref)
            p.tfirst = time()
        end
        @spawnat guiproc begin
          hold(false)
          # Timeseries plot
          subplot(241)
          plot(tx, Avec, "-", linewidth=2)
          title("Avec")

          subplot(242)
          plot(tx, Mvec, "-", linewidth=2)
          title("Mvec")

          subplot(243)
          plot(tx, DAvec, "-", linewidth=2)
          title("DAvec")
          axis([0, length(tx), 0, 0.4])

          subplot(244)
          plot(tx, DMvec, "-", linewidth=2)
          title("DMvec")
          axis([0, length(tx), 0, 0.4])

          subplot(245)
          pcolormesh(Mfield, vmin=0, vmax=0.6)
          title("Mfield")

          subplot(246)
          pcolormesh(Afield, vmin=0, vmax=0.6)
          title("Afield")

          subplot(247)
          pcolormesh(Ffield, vmin=0, vmax=1)
          title("Ffield")


            subplot(248)
            title("DirectionField")

            U = cos(directionfield)
            V = sin(directionfield)
            d1, d2 = size(directionfield)
            plt.quiver([1:d1], [1:d2], U, V, linewidth=1.5, headwidth = 0.5)
        end
    end

    t += stepIntegration
end # While

###
# Find the reason we terminated
###

print("Finished because after $t: ")
if t > tT
  println("Time done")
elseif isnan(meanMField) || isnan(meanAField)
   println("Found NaNs")
end

###
# Export
###
dt=now()
result = {
    "history_W" => history_W,
    "history_A" => history_A,
    "history_M" => history_M,
    "history_M_pot" => history_M_pot,
    "history_F" => history_F,
    "history_dir" => history_dir,
    "Avec" => Avec,
    "Fvec" => Fvec,
    "Mvec" => Mvec,
    "Wvec" => Wvec,
    "DAvec" => DAvec,
    "DMvec" => DMvec
}
merge!(result, saveConfig(collect(keys(baseConfig))))
matwrite("results/$(year(dt))-$(month(dt))-$(day(dt))_$(hour(dt)):$(minute(dt)):$(second(dt)).mat", result)
if !(worker || cluster)
println("Press any key to exit program.")
readline(STDIN)
end

structM = sumabs(old_Mfield .- Mfield)
structA = sumabs(old_Afield .- Afield)
structF = sumabs(old_Ffield .- Ffield)
structW = sumabs(old_Wfield .- Wfield)
structd = sumabs(old_directionfield .- directionfield)

if USECL
    cl.release!(queue)
end

println("Simulation finished")
return (t, timeToStable, stable, meanMField, meanAField, structM, structA, structF, structW, structd)
end #Function

function fsm(x, y, k)
  if (k == zero(k)) || (x == y)
    return zero(k)
  else
    return ((x-y) * k )
  end
end
