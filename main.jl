###
# Main
###

using MAT
using Datetime
using Distributions
import NumericExtensions: sumabs

include("config.jl")
include("drawcircle.jl")
include("gaussian_blur.jl")
include("utils.jl")
include("simulation.jl")
using Simulation

###
# set up initial configuration
###
function main(config=Dict(), disturbances=Dict(), cluster=false; enableVis :: Bool = false, fileName = "", loadTime = nothing, resultFolder="results")
    useVis = enableVis && ((length(procs()) == 1) || (!(myid() in workers()))) && !cluster
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

    value = simulation(Float64, cluster, useVis, fileName, loadTime, config, disturbances, guiproc, rref, resultFolder)

    return value
end

function simulation{T <: FloatingPoint}(:: Type{T}, cluster, enableVis, fileName, loadTime, config :: Dict, disturbances :: Dict, guiproc :: Int, gui_rref, resultFolder)
worker = (length(procs()) > 1) && (myid() in workers())

# initialize membrane fields
Afield = Array(T, 1 , 1)
Mfield = Array(T, 1 , 1)
Ffield = Array(T, 1 , 1)
Wfield = Array(T, 1 , 1)
θfield = Array(T, 1 , 1)

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
        θfield = history_dir[:,:,end]
    else
        Wfield = history_W[:,:,loadTime]
        Afield = history_A[:,:,loadTime]
        Mfield = history_M[:,:,loadTime]
        Ffield = history_F[:,:,loadTime]
        θfield = history_dir[:,:,loadTime]
    end
else
    Afield = zeros(T, fieldResY, fieldResX)
    Mfield = zeros(T, fieldResY, fieldResX)
    Ffield = avgF * ones(T, fieldResY, fieldResX)
    Wfield = ones(T, fieldResY, fieldResX)
    θfield = pi/4 * ones(T, fieldResY, fieldResX)

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
            θfield[q2,q1] = atan((q2-yc)/(q1-xc))-pi/2

            # correct for 180 degrees
            while θfield[q2, q1] > pi
                 θfield[q2, q1] -= pi
            end
            while θfield[q2, q1] <= 0 # ??? < 0
                 θfield[q2, q1] += pi
            end
        end
    end

    θfield[isnan(θfield)] = 0
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
# Prepare simulation
###

#define scaled diffusion
diffA = diffusionA*fieldRes/fieldSize
diffF = diffusionF*fieldRes/fieldSize
diffM = diffusionM*fieldRes/fieldSize
diffW = diffusionW*fieldRes/fieldSize

state = SimulationState{T}(Mfield, Afield, Wfield, Ffield, θfield)

# destroy

Mfield = nothing
Afield = nothing
Wfield = nothing
Ffield = nothing
θfield = nothing

temp = similar(state.Mfield)

const_area = convert(Array{T}, fill(1.0/8.0, 4, fieldResX, fieldResY))

# Stability and structure
stable = false
timeToStable = -1

old_Mfield = copy(state.Mfield)
old_Ffield = copy(state.Ffield)
old_Wfield = copy(state.Wfield)
old_Afield = copy(state.Afield)
old_θfield = copy(state.θfield)

###
# Simulation
###

t = 1

meanMField = mean(state.Mfield)
meanAField = mean(state.Afield)

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

previous_Mfield = copy(state.Mfield)
previous_Ffield = copy(state.Ffield)
previous_Wfield = copy(state.Wfield)
previous_Afield = copy(state.Afield)
previous_θfield = copy(state.θfield)

tx = Float64[]

while (t <= tT) && (meanAField < 1.0) !isnan(meanMField) && !isnan(meanAField)
    previous_Mfield = copy(state.Mfield)
    previous_Ffield = copy(state.Ffield)
    previous_Wfield = copy(state.Wfield)
    previous_Afield = copy(state.Afield)
    previous_θfield = copy(state.θfield)

	if t in keys(disturbances)
		val = disturbances[t]
		method = first(val)
		args = val[2:end]
		if method == :global && length(args) == 2
			(μ, σ) = args
            mask = !FrefillBinMask
			state.Mfield += mask .* rand(Normal(μ, σ), (fieldResY, fieldResX))
			state.Afield += mask .* rand(Normal(μ, σ), (fieldResY, fieldResX))
			state.Ffield += mask .* rand(Normal(μ, σ), (fieldResY, fieldResX))
			state.Wfield += mask .* rand(Normal(μ, σ), (fieldResY, fieldResX))
			state.θfield += mask .* rand(Normal(μ, σ), (fieldResY, fieldResX))

			# Normalize
			state.Mfield[state.Mfield .< 0] = 0
			state.Afield[state.Afield .< 0] = 0
			state.Ffield[state.Ffield .< 0] = 0
			state.Wfield[state.Wfield .< 0] = 0

			map!(x -> mod(x, pi), state.θfield, state.θfield)
		elseif method == :punch_local
			(x, y, α, β) = args
			apply_punch_down!(state.Mfield, x, y, α, β)
			apply_punch_down!(state.Afield, x, y, α, β)
			apply_punch_down!(state.Ffield, x, y, α, β)
			apply_punch_down!(state.Wfield, x, y, α, β)
            # apply_punch_down!(θfield, x, y, alpha, beta)
    elseif method == :punch_random
      (α, β) = args
      lin_idx = sample(find((s) -> s > 0.5, state.Mfield))
      x, y = ind2sub(size(state.Mfield), lin_idx)
      apply_punch_down!(state.Mfield, x, y, α, β)
      apply_punch_down!(state.Afield, x, y, α, β)
      apply_punch_down!(state.Ffield, x, y, α, β)
      apply_punch_down!(state.Wfield, x, y, α, β)
		elseif method == :gaussian_blur && length(args) == 1
			σ = args[1]
			imfilter_gaussian_no_nans!(state.Mfield, [σ,σ])
			imfilter_gaussian_no_nans!(state.Afield, [σ,σ])
			imfilter_gaussian_no_nans!(state.Ffield, [σ,σ])
			imfilter_gaussian_no_nans!(state.Wfield, [σ,σ])
			# θfield = imfilter_gaussian(θfield, [σ,σ])
		elseif method == :tear_membrane && length(args) == 1
			tearsize = args[1]

		    tS = iround(fieldRes/2+tearSize*fieldRes/fieldSize)
		    t0 = iround(fieldRes/2)
		    state.Mfield[fieldRes/2:fieldRes,t0:tS] = 0
		elseif method == :tear_dfield && length(args) == 1
			tearsize = args[1]

		    tS = iround(fieldRes/2+tearSize*fieldRes/fieldSize)
		    t0 = iround(fieldRes/2)
		    state.θfield[fieldRes/2:fieldRes,t0:tS] = pi .* rand(iround(fieldRes-fieldRes/2+1),tS-t0+1)
		elseif method == :tear && length(args) == 1
			tearsize = args[1]

		    tS = iround(fieldRes/2+tearSize*fieldRes/fieldSize)
		    t0 = iround(fieldRes/2)

		    state.Mfield[fieldRes/2:fieldRes,t0:tS] = 0
		    state.θfield[fieldRes/2:fieldRes,t0:tS] = pi .* rand(iround(fieldRes-fieldRes/2+1),tS-t0+1)
		else
			warn("Don't know how to handle $method with arguments $args, carry on.")
		end
	end

    ###
    # calculate potential based on repulsion
    ###
    area!(state.Area, state.θfield, long_direction)

    potential!(state.Mpot,  state.Wpot, state.Mfield, state.Wfield, state.Area, MW_repulsion)
    potential!(state.Mpot1, state.Apot, state.Mfield, state.Afield, state.Area, MA_repulsion)
    potential!(state.Mpot2, temp,       state.Mfield, state.Mfield, state.Area, MM_repulsion)

    potential!(state.Apot1, temp,       state.Afield, state.Afield, const_area, AA_repulsion)

    state.Apot += state.Apot1
    state.Mpot += state.Mpot1 + state.Mpot2

    ###
    # move molecules and update directionality
    ###

    flow!(state.Mflow, state.Mpot)
    flow!(state.Aflow, state.Apot)
    flow!(state.Wflow, state.Wpot)

    diffusion!(state.Mlap, state.Mfield, state.Mflow, diffM)
    diffusion!(state.Wlap, state.Wfield, state.Wflow, diffW)
    diffusion!(state.Alap, state.Afield, state.Aflow, diffA)

    # Laplacian for diffusion
    laplacian!(state.Flap, state.Ffield, diffF)

    # update direction field based on alignment
    align!(state.θfield, state.Mfield, state.θfield, attractionRate, stepIntegration)

    ###
    # reactions
    ##
    for j in 1:fieldResY
      for i in 1:fieldResX
        ##
        # Calculate row values once
        ##
        row1 =  state.Mfield[i, j] ^ m11 * state.Afield[i, j] ^ a11 * state.Ffield[i, j] ^ f11 * state.Wfield[i, j] ^ w11
        row2 =  state.Mfield[i, j] ^ m12 * state.Afield[i, j] ^ a12 * state.Ffield[i, j] ^ f12 * state.Wfield[i, j] ^ w12
        row3 =  state.Mfield[i, j] ^ m21 * state.Afield[i, j] ^ a21 * state.Ffield[i, j] ^ f21 * state.Wfield[i, j] ^ w21
        row4 =  state.Mfield[i, j] ^ m22 * state.Afield[i, j] ^ a22 * state.Ffield[i, j] ^ f22 * state.Wfield[i, j] ^ w22
        row5 =  state.Mfield[i, j] ^ m31 * state.Afield[i, j] ^ a31 * state.Ffield[i, j] ^ f31 * state.Wfield[i, j] ^ w31
        row6 =  state.Mfield[i, j] ^ m32 * state.Afield[i, j] ^ a32 * state.Ffield[i, j] ^ f32 * state.Wfield[i, j] ^ w32

        ##
        # Calculate dA
        ##
        state.dA[i, j] = (
                          ((a12 - a11) * kf1)  * row1 +
                          ((a11 - a12) * kb2)  * row2 +
                          ((a22 - a12) * kf2)  * row3 +
                          ((a21 - a22) * kb2)  * row4 +
                          ((a32 - a31) * kf3)  * row5 +
                          ((a31 - a32) * kb3)  * row6
                        ) / (1 + state.Mfield[i,j]) - decayA * state.Afield[i, j] + state.Alap[i, j]

        ##
        # Calculate dM
        ##
        state.dM[i, j] = (
                          ((m12 - m11) * kf1)  * row1 +
                          ((m11 - m12) * kb2)  * row2 +
                          ((m22 - m12) * kf2)  * row3 +
                          ((m21 - m22) * kb2)  * row4 +
                          ((m32 - m31) * kf3)  * row5 +
                          ((m31 - m32) * kb3)  * row6
                        ) / (1 + state.Afield[i,j]) - decayM * state.Mfield[i, j] + state.Mlap[i, j]

        ##
        # Calculate dW
        ##
        state.dW[i, j] =  ((w12 - w11) * kf1)  * row1 +
                          ((w11 - w12) * kb2)  * row2 +
                          ((w22 - w12) * kf2)  * row3 +
                          ((w21 - w22) * kb2)  * row4 +
                          ((w32 - w31) * kf3)  * row5 +
                          ((w31 - w32) * kb3)  * row6 +
                          state.Wlap[i, j]

        ##
        # Calculate dF
        ##
        state.dF[i, j] =  ((f12 - f11) * kf1)  * row1 +
                          ((f11 - f12) * kb2)  * row2 +
                          ((f22 - f12) * kf2)  * row3 +
                          ((f21 - f22) * kb2)  * row4 +
                          ((f32 - f31) * kf3)  * row5 +
                          ((f31 - f32) * kb3)  * row6 +
                          state.Flap[i, j]

        if FrefillBinMask[i,j]
          state.dF[i, j] += flowRateF * (saturationF .- state.Ffield[i, j])
        end
      end
    end

    dT = 0

    # update values
    Δ = (stepIntegration / precision)
    for j in 1:fieldResY
      for i in 1:fieldResX
        state.Afield[i, j] += Δ * state.dA[i, j]
        state.Ffield[i, j] += Δ * state.dF[i, j]
        state.Mfield[i, j] += Δ * state.dM[i, j]
        state.Wfield[i, j] += Δ * state.dW[i, j]
      end
    end

    # calulate stability criteria

    sumabs_dA = sumabs(state.dA)
    sumabs_dM = sumabs(state.dM)
    sumabs_dF = sumabs(state.dF)
    sumabs_dW = sumabs(state.dW)

    negativeValues = any(state.Mfield .< 0) || any(state.Afield .< 0) || any(state.Ffield .< 0) || any(state.Wfield .< 0)

    if negativeValues
        warn("Integration error at t=$t increase precision from $precision to $(2 * precision)")
        precision = 2 * precision

        if precision > 1024
            error("Precision just exceeded 1024 clearly something is wrong here.")
        end

        state.Mfield = copy(previous_Mfield)
        state.Ffield = copy(previous_Ffield)
        state.Wfield = copy(previous_Wfield)
        state.Afield = copy(previous_Afield)
        state.θfield = copy(previous_θfield)
        skipFlag = true

        # If we increase it for the first time
        if precisionCounter == 0
            precisionCounter = bufferStep*precision
        else
            precisionCounter *= 2
        end
    end

    #save values for visualization
    meanAField = mean(state.Afield)
    meanMField = mean(state.Mfield)

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
        push!(Fvec, mean(state.Ffield))
        push!(Wvec, mean(state.Wfield))
        push!(DAvec, sumabs_dA)
        push!(DMvec, sumabs_dM)

        if (t % storeStep == 0)
          push!(history_A, state.Afield)
          push!(history_F, state.Ffield)
          push!(history_M, state.Mfield)
          push!(history_W, state.Wfield)
          push!(history_dir, state.θfield)
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
          pcolormesh(state.Mfield, vmin=0, vmax=0.6)
          title("Mfield")

          subplot(246)
          pcolormesh(state.Afield, vmin=0, vmax=0.6)
          title("Afield")

          subplot(247)
          pcolormesh(state.Ffield, vmin=0, vmax=1)
          title("Ffield")


            subplot(248)
            title("DirectionField")

            U = cos(state.θfield)
            V = sin(state.θfield)
            d1, d2 = size(state.θfield)
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

structM = sumabs(old_Mfield .- state.Mfield)
structA = sumabs(old_Afield .- state.Afield)
structF = sumabs(old_Ffield .- state.Ffield)
structW = sumabs(old_Wfield .- state.Wfield)
structd = sumabs(old_θfield .- state.θfield)

valueOfInterest =   if stable
                        if (meanMField >= deathEpsilon) || (meanAField >= deathEpsilon)
                            timeToStable # stable and living
                        else
                            -timeToStable # death through diffusion and decay
                        end
                    elseif (meanAField >= 1.0) || isnan(meanMField) || isnan(meanAField) # explosion
                        - t
                    end

return (t, timeToStable, valueOfInterest, stable, meanMField, meanAField, structM, structA, structF, structW, structd, outFileName)
end #Function

function fsm(x, y, k)
  if (k == zero(k)) || (x == y)
    return zero(k)
  else
    return ((x-y) * k )
  end
end
