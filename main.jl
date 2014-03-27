###
# Main
###

using MAT
using Datetime
using Distributions
import NumericExtensions.sumabs
import NumericExtensions
const ne = NumericExtensions

include("config.jl")
include("drawcircle.jl")
include("gaussian_blur.jl")
include("utils.jl")

###
# set up initial configuration
###
function main(config=Dict(), disturbances=Dict(), cluster=false; enableVis :: Bool = false, fileName = "", loadTime = nothing, allow32Bit = false, forceJuliaImpl = false, resultFolder="results")
    useVis = enableVis && ((length(procs()) == 1) || (!(myid() in workers()))) && !cluster
    ###
    # Prepare GPU
    ###
    const P64BIT, USECL, ctx, queue = determineCapabilities(cluster, allow32Bit, forceJuliaImpl)

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

    value = simulation(cluster, useVis, fileName, loadTime, USECL, P64BIT ? Float64 : Float32, config, disturbances, guiproc, ctx, queue, rref, resultFolder)

    ctx = nothing
    queue = nothing
    gc()
    return value
end

function simulation{T <: FloatingPoint}(cluster, enableVis, fileName, loadTime, USECL, :: Type{T}, config :: Dict, disturbances :: Dict, guiproc :: Int, ctx, queue, gui_rref, resultFolder)
worker = (length(procs()) > 1) && (myid() in workers())


# initialize membrane fields
Afield = nothing
Mfield = nothing
Ffield = nothing
Wfield = nothing
directionfield = nothing

# Prepare the configuration values and load the default settings
loadConfig(baseConfig)

# If we have a non empty fileName and we don't run on a cluster
# Load configuration values from data directory.
fileConfig = if fileName != "" && !cluster
    matread("data/$(fileName).mat")
else
    Dict()
end

# Load file config
loadConfig(fileConfig, dataVars)

# Overwrite configuration values from a
loadConfig(config, dataVars)
updateDependentValues() # Update the values that depend on the configuration

# Load initial state.
if loadTime != nothing
    history_W, history_A, history_M, history_F, history_dir =
        if cluster && checkVars(historyVars, collect(keys(config)))
            (config["history_W"], config["history_A"], config["history_M"], config["history_F"], config["history_dir"])
        elseif checkVars(historyVars, collect(keys(fileConfig)))
            (fileConfig["history_W"], fileConfig["history_A"], fileConfig["history_M"], fileConfig["history_F"], fileConfig["history_dir"])
        else
            error("Could not find historyVars despite being given a loadTime of $loadTime")
        end

    if loadTime == -1
        Wfield = history_W[:,:,end]
        Afield = history_A[:,:,end]
        Mfield = history_M[:,:,end]
        Ffield = history_F[:,:,end]
        directionfield = history_dir[:,:,end]
    else
        Wfield = history_W[:,:,loadTime]
        Afield = history_A[:,:,loadTime]
        Mfield = history_M[:,:,loadTime]
        Ffield = history_F[:,:,loadTime]
        directionfield = history_dir[:,:,loadTime]
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
    Mfield += avgM * M_circ + 0.00001* rand(fieldResY, fieldResX)

    # fill in with autocatalyst
    A_circ = zeros(T, fieldResY, fieldResX)

    for r in 0:0.01:mR-1
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

# create 3d matrices to store field activities
history_A = Array{T, 2}[]
history_F = Array{T, 2}[]
history_T = Array{T, 2}[]
history_M = Array{T, 2}[]
history_W = Array{T, 2}[]
history_dir = Array{T, 2}[]

#vectors to save global concentrations across time

Avec = T[]
Fvec = T[]
Mvec = T[]
Wvec = T[]
DAvec = T[]
DMvec = T[]

###
# Decide which compute context should be used
# Either OpenCL or Julia
###

if USECL
    include("cl/functions.jl")
else
    include("jl/functions.jl")
end

createComputeContext(T, ctx, queue)

###
# Prepare simulation
###

#define scaled diffusion
diffA = diffusionA*fieldRes/fieldSize
diffF = diffusionF*fieldRes/fieldSize
diffM = diffusionM*fieldRes/fieldSize
diffW = diffusionW*fieldRes/fieldSize

dF = zeros(T, fieldResY, fieldResX)

# create temp arrays
mpot = create(T)
mpot1 = create(T)
mpot2 = create(T)
wpot = create(T)
apot = create(T)
apot1 = create(T)


buff_Mfield = create(T)
buff_Wfield = create(T)
buff_Afield = create(T)
buff_Dfield = create(T)
buff_NewDfield = create(T)
buff_Ffield = create(T)

wlap = create(T)
alap = create(T)
mlap = create(T)
flap = create(T)

row1 = create(T)
row2 = create(T)
row3 = create(T)
row4 = create(T)
row5 = create(T)
row6 = create(T)

dW = create(T)
dA = create(T)
dM = create(T)
buff_dF = create(T)

temp = create(T)
area = create_n4(T)
const_area = create_const_n4(convert(T, 1.0/8.0))


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
outFileName = ""

###
# Set up dynamic precision
###

precision = 1.0
skipFlag = false
precisionCounter = 0
bufferStep = 2 # Minimum 1.0 increase to allow for higher precision over a second timeStep

previous_Mfield = copy(Mfield)
previous_Ffield = copy(Ffield)
previous_Wfield = copy(Wfield)
previous_Afield = copy(Afield)
previous_directionfield = copy(directionfield)

tx = Float64[]

while (t <= tT) && (meanAField < 1.0) !isnan(meanMField) && !isnan(meanAField)
    previous_Mfield = copy(Mfield)
    previous_Ffield = copy(Ffield)
    previous_Wfield = copy(Wfield)
    previous_Afield = copy(Afield)
    previous_directionfield = copy(directionfield)

	if t in keys(disturbances)
		val = disturbances[t]
		method = first(val)
		args = val[2:end]
		if method == :global && length(args) == 2
			(mu, sig) = args
            mask = !FrefillBinMask
			Mfield += mask .* rand(Normal(mu, sig), (fieldResY, fieldResX))
			Afield += mask .* rand(Normal(mu, sig), (fieldResY, fieldResX))
			Ffield += mask .* rand(Normal(mu, sig), (fieldResY, fieldResX))
			Wfield += mask .* rand(Normal(mu, sig), (fieldResY, fieldResX))
			directionfield += mask .* rand(Normal(mu, sig), (fieldResY, fieldResX))

			# Normalize
			Mfield[Mfield .< 0] = 0
			Afield[Afield .< 0] = 0
			Ffield[Ffield .< 0] = 0
			Wfield[Wfield .< 0] = 0

			directionfield = map(x -> mod(x, pi), directionfield)
		elseif method == :punch_local
			(x, y, alpha, beta) = args
			apply_punch_down!(Mfield, x, y, alpha, beta)
			apply_punch_down!(Afield, x, y, alpha, beta)
			apply_punch_down!(Ffield, x, y, alpha, beta)
			apply_punch_down!(Wfield, x, y, alpha, beta)
            # apply_punch_down!(directionfield, x, y, alpha, beta)
        elseif method == :punch_random
            (alpha, beta) = args
            lin_idx = sample(find((s) -> s > 0.5, Mfield))
            x, y = ind2sub(size(Mfield), lin_idx)
            apply_punch_down!(Mfield, x, y, alpha, beta)
            apply_punch_down!(Afield, x, y, alpha, beta)
            apply_punch_down!(Ffield, x, y, alpha, beta)
            apply_punch_down!(Wfield, x, y, alpha, beta)
		elseif method == :gaussian_blur && length(args) == 1
			sigma = args[1]
			imfilter_gaussian_no_nans!(Mfield, [sigma,sigma])
			imfilter_gaussian_no_nans!(Afield, [sigma,sigma])
			imfilter_gaussian_no_nans!(Ffield, [sigma,sigma])
			imfilter_gaussian_no_nans!(Wfield, [sigma,sigma])
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

    c_copy!(buff_Mfield, Mfield)
    c_copy!(buff_Wfield, Wfield)
    c_copy!(buff_Afield, Afield)
    c_copy!(buff_Ffield, Ffield)
    c_copy!(buff_Dfield, directionfield)

    ###
    # calculate potential based on repulsion
    ###
    area!(buff_Dfield, area)

    potential!(buff_Mfield, buff_Wfield, area, mpot, wpot, MW_repulsion)
    potential!(buff_Mfield, buff_Afield, area, mpot1, apot, MA_repulsion)
    potential!(buff_Mfield, buff_Mfield, area, mpot2, temp, MM_repulsion)

    potential!(buff_Afield, buff_Afield, const_area, apot1, temp, AA_repulsion)

    c_add!(apot, apot1)
    c_add!(mpot, mpot1)
    c_add!(mpot, mpot2)

    ###
    # move molecules and update directionality
    ###

    diffusion!(buff_Mfield, mpot, mlap)
    smul!(mlap, diffM)

    diffusion!(buff_Wfield, wpot, wlap)
    smul!(wlap, diffW)

    diffusion!(buff_Afield, apot, alap)
    smul!(alap, diffA)


    # Laplacian for diffusion
    # Todo calculate in on GPU to be coherent and to minimize data transfers.
    laplacian!(buff_Ffield, flap)
    smul!(flap, diffF)

    # update direction field based on alignment
    align!(buff_Mfield, buff_Dfield, buff_NewDfield)
    c_copy!(directionfield, buff_NewDfield)

    ###
    # reactions
    ##

    calcRow!(buff_Mfield, buff_Afield, buff_Ffield, buff_Wfield, row1, m11, a11, f11, w11)
    calcRow!(buff_Mfield, buff_Afield, buff_Ffield, buff_Wfield, row2, m12, a12, f12, w12)
    calcRow!(buff_Mfield, buff_Afield, buff_Ffield, buff_Wfield, row3, m21, a21, f21, w21)
    calcRow!(buff_Mfield, buff_Afield, buff_Ffield, buff_Wfield, row4, m22, a22, f22, w22)
    calcRow!(buff_Mfield, buff_Afield, buff_Ffield, buff_Wfield, row5, m31, a31, f31, w31)
    calcRow!(buff_Mfield, buff_Afield, buff_Ffield, buff_Wfield, row6, m32, a32, f32, w32)

    ##
    # Calculate dA
    ##

    delta!(row1, row2, row3, row4, row5, row6, dA,
            fsm(a12, a11, kf1), fsm(a11, a12, kb1), fsm(a22, a21, kf2), fsm(a21, a22, kb2), fsm(a32, a31, kf3), fsm(a31, a32, kb3))
    delta2!(dA, buff_Mfield, buff_Afield, alap, dA, decayA)

    ##
    # Calculate dM
    ##

    delta!(row1, row2, row3, row4, row5, row6, dM,
            fsm(m12, m11, kf1), fsm(m11, m12, kb1), fsm(m22, m21, kf2), fsm(m21, m22, kb2), fsm(m32, m31, kf3), fsm(m31, m32, kb3))
    delta2!(dM, buff_Afield, buff_Mfield, mlap, dM, decayM)

    ##
    # Calculate dW
    ##

    delta!(row1, row2, row3, row4, row5, row6, dW,
            fsm(w12, w11, kf1), fsm(w11, w12, kb1), fsm(w22, w21, kf2), fsm(w21, w22, kb2), fsm(w32, w31, kf3), fsm(w31, w32, kb3))

    c_add!(dW, wlap)

    ##
    # Calculate dF
    ##

    delta!(row1, row2, row3, row4, row5, row6, buff_dF,
            fsm(f12, f11, kf1), fsm(f11, f12, kb1), fsm(f22, f21, kf2), fsm(f21, f22, kb2), fsm(f32, f31, kf3), fsm(f31, f32, kb3))

    c_add!(buff_dF, flap)

    c_copy!(dF, buff_dF)

    dF[FrefillBinMask] += flowRateF * (saturationF .- Ffield[FrefillBinMask])

    dT = 0

    # update values
    smul!(dA, (stepIntegration / precision))
    c_add!(buff_Afield, dA)
    c_copy!(Afield, buff_Afield)

    Ffield += dF * (stepIntegration / precision)

    smul!(dM, (stepIntegration / precision))
    c_add!(buff_Mfield, dM)
    c_copy!(Mfield, buff_Mfield)

    smul!(dW, (stepIntegration / precision))
    c_add!(buff_Wfield, dW)
    c_copy!(Wfield, buff_Wfield)

    # calulate stability criteria

    sumabs_dA = sumabs(c_read(dA))
    sumabs_dM = sumabs(c_read(dM))
    sumabs_dF = sumabs(dF)
    sumabs_dW = sumabs(c_read(dW))

    negativeValues = any(Mfield .< 0) || any(Afield .< 0) || any(Ffield .< 0) || any(Wfield .< 0)

    if negativeValues
        warn("Integration error at t=$t increase precision from $precision to $(2 * precision)")
        precision = 2 * precision

        if precision > 1024
            error("Precision just exceeded 1024 clearly something is wrong here.")
        end

        Mfield = copy(previous_Mfield)
        Ffield = copy(previous_Ffield)
        Wfield = copy(previous_Wfield)
        Afield = copy(previous_Afield)
        directionfield = copy(previous_directionfield)
        skipFlag = true

        # If we increase it for the first time
        if precisionCounter == 0
            precisionCounter = bufferStep*precision
        else
            precisionCounter *= 2
        end
    end

    #save values for visualization
    meanAField = mean(Afield)
    meanMField = mean(Mfield)

    if !skipFlag
        stableCondition = (sumabs_dA < epsilon / precision) && (sumabs_dM < epsilon / precision) #&& (sumabs_dF < epsilon) && (sumabs_dW < epsilon)

        if !stable && stableCondition
            println("reached a stable config after $t")
            stable = true
            tT = t + stableTime
            timeToStable = t
        elseif stable && !stableCondition
            stable = false
            tT = t + timeTotal
            println("Lost stability")
        end

        push!(Avec, meanAField)
        push!(Mvec, meanMField)
        push!(Fvec, mean(Ffield))
        push!(Wvec, mean(Wfield))
        push!(DAvec, sumabs_dA)
        push!(DMvec, sumabs_dM)

        if (t % storeStep == 0)
          push!(history_A, Afield)
          push!(history_F, Ffield)
          push!(history_M, Mfield)
          push!(history_W, Wfield)
          push!(history_dir, directionfield)
        end

        push!(tx, t)
    end

    if enableVis && (t % visInterval == 0)
        if !isready(gui_rref)
            println()
            println("Waiting for GUI to be initialized.")
            @time wait(gui_rref)
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
          axis([0, t, 0, 0.4])

          subplot(244)
          plot(tx, DMvec, "-", linewidth=2)
          title("DMvec")
          axis([0, t, 0, 0.4])

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

    yield() # to be able to answer question about the status
    if !skipFlag
        if precisionCounter > 0
            precisionCounter -= 1
            t += (stepIntegration / precision)
        else
            t += stepIntegration
        end
    else
        skipFlag = false
    end

    if precisionCounter == 0 && precision > 1.0
        if t % stepIntegration != 0
            warn("t=$t and integration=$stepIntegration t % integration should be zero but it is. $(t%stepIntegration)")
        end
        warn("Resetting precision to 1.0 at t=$t")
        precision = 1.0
    end
end # While

###
# Find the reason we terminated
###

print("Finished because after $t: ")
if t > tT
  println("Time done")
elseif isnan(meanMField) || isnan(meanAField)
   println("Found NaNs")
elseif meanAField >= 1.0
    println("Mean of Afield is $meanAField >= 1.0")
end

###
# Export
###
try
    dt=now()
    stack(X) = reshape(hcat(X...), fieldResY, fieldResX, length(X))
    result = {
        "history_W" => stack(history_W),
        "history_A" => stack(history_A),
        "history_M" => stack(history_M),
        "history_F" => stack(history_F),
        "history_dir" => stack(history_dir),
        "Avec" => Avec,
        "Fvec" => Fvec,
        "Mvec" => Mvec,
        "Wvec" => Wvec,
        "DAvec" => DAvec,
        "DMvec" => DMvec,
        "tx" => tx
    }
    merge!(result, saveConfig(collect(keys(baseConfig))))
    prefix = if cluster
        "$resultFolder/$(myid())-"
    else
        "$resultFolder/"
    end
    outFileName = "$prefix$(year(dt))-$(month(dt))-$(day(dt))_$(hour(dt)):$(minute(dt)):$(second(dt)).mat"
    matwrite(outFileName, result)
catch e
    warn("Write failed because of $e")
end

if !(worker || cluster)
println("Press any key to exit program.")
readline(STDIN)
end

structM = sumabs(old_Mfield .- Mfield)
structA = sumabs(old_Afield .- Afield)
structF = sumabs(old_Ffield .- Ffield)
structW = sumabs(old_Wfield .- Wfield)
structd = sumabs(old_directionfield .- directionfield)

valueOfInterest =   if stable
                        if (meanMField >= deathEpsilon) || (meanAField >= deathEpsilon)
                            timeToStable # stable and living
                        else
                            -timeToStable # death through diffusion and decay
                        end
                    elseif (meanAField >= 1.0) || isnan(meanMField) || isnan(meanAField) # explosion
                        - t
                    end

destroyComputeContext()

return (t, timeToStable, valueOfInterest, stable, meanMField, meanAField, structM, structA, structF, structW, structd, outFileName)
end #Function

function fsm(x, y, k)
  if (k == zero(k)) || (x == y)
    return zero(k)
  else
    return ((x-y) * k )
  end
end
