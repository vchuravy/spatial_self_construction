###
# Main
###

using MAT
using ProgressMeter
using Datetime
using PyPlot
import NumericExtensions.sumabs
import OpenCL
const cl = OpenCL

include("config.jl")
include("drawcircle.jl")
#include("potential.jl")
#include("diffusion.jl")
#include("laplacian.jl")
#include("align.jl")
include("cl/potentialCL.jl")
include("cl/diffusionCL.jl")
include("cl/laplacianCL.jl")
include("cl/addCL.jl")
include("cl/alignCL.jl")
include("cl/calcRowCL.jl")
include("cl/deltaCL.jl")
include("cl/smulCL.jl")
include("cl/deltaStep2CL.jl")

###
# set up initial configuration
###

function main(config=Dict();enableVis = false, enableDirFieldVis = false, fileName = "", loadTime = 0, debug = false)
###
# Prepare GPU
###
device, ctx, queue = cl.create_compute_context()
const CL64BIT =
    if "cl_khr_fp64" in cl.info(device, :extensions)
         true
    elseif "cl_amd_fp64" in cl.info(device, :extensions)
        true
    else
        warn("No Float64 support.")
        false
    end

    ###
    # Prepare GUI
    ###

    guiproc = if enableVis && (length(procs()) <= 1)
        first(addprocs(1))
    else
        last(procs())
    end
    if enableVis
        @everywhere using PyPlot
    end

    simulation(enableVis, enableDirFieldVis, fileName, loadTime, device, ctx, queue, CL64BIT, CL64BIT ? Float64 : Float32, debug, config, guiproc)

end

function simulation{T <: FloatingPoint}(enableVis, enableDirFieldVis, fileName, loadTime, device, ctx, queue, CL64BIT, :: Type{T}, testCL :: Bool, config :: Dict, guiproc :: Int)
# initialize membrane fields
Afield = zeros(T, fieldResY, fieldResX)
Mfield = zeros(T, fieldResY, fieldResX)
Ffield = avgF * ones(T, fieldResY, fieldResX)
Tfield = zeros(T, fieldResY, fieldResX)
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

if fileName != ""
    vars = matread("data/$(fileName).mat")

    for k in keys(vars)
       s = symbol(k)
       expr = :($s = $(vars[k]))
       eval(expr)
    end

    Wfield = histWfield[:,:,loadTime]
    Afield = histAfield[:,:,loadTime]
    Mfield = histMfield[:,:,loadTime]
    Ffield = histFfield[:,:,loadTime]
end

for k in keys(config)
       s = symbol(k)
       expr = :($s = $(config[k]))
       eval(expr)
end

Frefill = ones(fieldRes, fieldRes)
frefillX = ceil(fieldRes/2)-mR-mT-fD+1 : ceil(fieldRes/2)+mR+mT+fD
frefillY = ceil(fieldRes/2)-mR-mT-fD+1 : ceil(fieldRes/2)+mR+mT+fD
Frefill[frefillX, frefillY] = 0
FrefillBinMask = Frefill .> 0.5

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
Tvec = zeros(T, vecL)
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
diffT = diffusionT*fieldRes/fieldSize
diffM = diffusionM*fieldRes/fieldSize
diffW = diffusionW*fieldRes/fieldSize


dM = 0

tx = [0:stepIntegration:timeTotal-stepIntegration]


qX = zeros(T, fieldResX*fieldResY, 1)
qY = zeros(T, fieldResX*fieldResY, 1)
qU = zeros(T, fieldResX*fieldResY, 1)
qV = zeros(T, fieldResX*fieldResY, 1)

ind=1
for xx in 1:fieldResX
    for yy in 1:fieldResY
        qX[ind] = xx
        qY[ind] = yy
        ind += 1
    end
end

dF = zeros(T, fieldResY, fieldResX)

###
# Simulation
###

t = 1
p = Progress(length(tx), 1)

meanMField = mean(Mfield)
meanAField = mean(Afield)

###
# CL prepare programs.

potentialProgram = cl.Program(ctx, source=getPotentialKernel(T)) |> cl.build!
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
# For the OCL functionts the target is always LAST.

diffusion!(buff_Xfield, buff_Xpot, buff_Xlap) = diffusionCL!(buff_Xfield, buff_Xpot, buff_Xlap, fieldResY, fieldResX, ctx, queue, diffusionProgram)
smul!(X, buff_in, buff_out) =  smulCL!(X, buff_in, buff_out, fieldResY, fieldResX, ctx, queue, smulProgram)
add!(buff_in1, buff_in2, buff_out) = addCL!(buff_in1, buff_in2, buff_out, fieldResY, fieldResX, ctx, queue, addProgram)
potential!(buff_Xfield, buff_Yfield, buff_Zfield, buff_Xpot, buff_Ypot, repulsion) = potentialCL!(buff_Xfield, buff_Yfield, buff_Zfield, buff_Xpot, buff_Ypot, repulsion, long_direction, fieldResY, fieldResX, ctx, queue, potentialProgram)
align!(buff_Xfield, buff_Yfield, buff_OUTfield) = alignCL!(buff_Xfield, buff_Yfield, buff_OUTfield, attractionRate, stepIntegration, fieldResY, fieldResX, ctx, queue, alignProgram)
calcRow!(buff_Xfield, buff_Yfield, buff_Zfield, buff_Wfield, buff_OUT, x, y, z, w) = calcRowCL!(buff_Xfield, buff_Yfield, buff_Zfield, buff_Wfield, buff_OUT, x, y, z, w, fieldResY, fieldResX, ctx, queue, rowProgram)
laplacian!(buff_in, buff_out) = laplacianCL!(buff_in, buff_out, fieldResY, fieldResX, ctx, queue, laplacianProgram )

#create buffers on device
bufferSize = fieldResX * fieldResY

buff_mpot1 = cl.Buffer(T, ctx, :rw, bufferSize)
buff_wpot = cl.Buffer(T, ctx, :rw, bufferSize)
buff_mpot2 = cl.Buffer(T, ctx, :rw, bufferSize)
buff_apot = cl.Buffer(T, ctx, :rw, bufferSize)
buff_mpot = cl.Buffer(T, ctx, :rw, bufferSize)

buff_mfield = cl.Buffer(T, ctx, :rw, bufferSize)
buff_wfield = cl.Buffer(T, ctx, :rw, bufferSize)
buff_afield = cl.Buffer(T, ctx, :rw, bufferSize)
buff_dfield = cl.Buffer(T, ctx, :rw, bufferSize)
buff_ndfield = cl.Buffer(T, ctx, :rw, bufferSize)
buff_ffield = cl.Buffer(T, ctx, :rw, bufferSize)

buff_wlap = cl.Buffer(T, ctx, :rw, bufferSize)
buff_alap = cl.Buffer(T, ctx, :rw, bufferSize)
buff_mlap = cl.Buffer(T, ctx, :rw, bufferSize)
buff_flap = cl.Buffer(T, ctx, :rw, bufferSize)

buff_row1 = cl.Buffer(T, ctx, :rw, bufferSize)
buff_row2 = cl.Buffer(T, ctx, :rw, bufferSize)
buff_row3 = cl.Buffer(T, ctx, :rw, bufferSize)
buff_row4 = cl.Buffer(T, ctx, :rw, bufferSize)
buff_row5 = cl.Buffer(T, ctx, :rw, bufferSize)
buff_row6 = cl.Buffer(T, ctx, :rw, bufferSize)

buff_dW = cl.Buffer(T, ctx, :rw, bufferSize)
buff_dA = cl.Buffer(T, ctx, :rw, bufferSize)
buff_dM = cl.Buffer(T, ctx, :rw, bufferSize)
buff_dF = cl.Buffer(T, ctx, :rw, bufferSize)


# Make sure that the fields are in the correct DataFormat
Mfield = convert(Array{T}, Mfield)
Afield = convert(Array{T}, Afield)
Wfield = convert(Array{T}, Wfield)
Ffield = convert(Array{T}, Ffield)
Tfield = convert(Array{T}, Tfield)

while (t <= timeTotal) && (meanMField < 2) && (meanMField > 0.001) && (meanAField < 2) && (meanAField > 0.001)

    ###
    # Todo only calc mean once per step
    ###

    if t == tearTime1
        tS = iround(tearSize1*fieldRes/fieldSize)
        t0 = iround(fieldRes/2)
        Mfield[1:fieldRes,t0:tS] = 0
    end

    if t == tearTime2
        tS = iround(tearSize1*fieldRes/fieldSize)
        t0 = iround(fieldRes/2)
        directionfield[1:fieldRes, t0:tS] = pi .* rand(fieldRes, length(t0:tS))
    end

    ###
    # Pushing the fields on the GPU
    ###

    cl.copy!(queue, buff_mfield, Mfield)
    cl.copy!(queue, buff_wfield, Wfield)
    cl.copy!(queue, buff_afield, Afield)
    cl.copy!(queue, buff_ffield, Ffield)
    cl.copy!(queue, buff_dfield, directionfield)

    ###
    # calculate potential based on repulsion
    ###
    potential!(buff_mfield, buff_wfield, buff_dfield, buff_mpot1, buff_wpot, MW_repulsion)
    potential!(buff_mfield, buff_afield, buff_dfield, buff_mpot2, buff_apot, MA_repulsion)

    add!(buff_mpot1, buff_mpot2, buff_mpot)

    ###
    # move molecules and update directionality
    ###

    diffusion!(buff_mfield, buff_mpot, buff_mlap)
    smul!(diffM, buff_mlap, buff_mlap)

    diffusion!(buff_wfield, buff_wpot, buff_wlap)
    smul!(diffW, buff_wlap, buff_wlap)

    diffusion!(buff_afield, buff_apot, buff_alap)
    smul!(diffA, buff_alap, buff_alap)


    # Laplacian for diffusion
    # Todo calculate in on GPU to be coherent and to minimize data transfers.
    laplacian!(buff_ffield, buff_flap)
    smul!(diffF, buff_flap, buff_flap)

    # update direction field based on alignment
    align!(buff_mfield, buff_dfield, buff_ndfield)
    cl.copy!(queue, directionfield, buff_ndfield)

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

    deltaCL!(buff_row1, buff_row2, buff_row3, buff_row4, buff_row5, buff_row6, buff_dA,
            fsm(a12, a11, kf1), fsm(a11, a12, kb1), fsm(a22, a21, kf2), fsm(a21, a22, kb2), fsm(a32, a31, kf3), fsm(a31, a32, kb3),
            fieldResY, fieldResX, ctx, queue, deltaProgram)
    delta2CL!(buff_dA, buff_mfield, buff_afield, buff_alap, buff_dA, decayA, fieldResY, fieldResX, ctx, queue, delta2Program)

    ##
    # Calculate dM
    ##

    deltaCL!(buff_row1, buff_row2, buff_row3, buff_row4, buff_row5, buff_row6, buff_dM,
            fsm(m12, m11, kf1), fsm(m11, m12, kb1), fsm(m22, m21, kf2), fsm(m21, m22, kb2), fsm(m32, m31, kf3), fsm(m31, m32, kb3),
            fieldResY, fieldResX, ctx, queue, deltaProgram)
    delta2CL!(buff_dM, buff_afield, buff_mfield, buff_mlap, buff_dM, decayM, fieldResY, fieldResX, ctx, queue, delta2Program)

    ##
    # Calculate dW
    ##

    deltaCL!(buff_row1, buff_row2, buff_row3, buff_row4, buff_row5, buff_row6, buff_dW,
            fsm(w12, w11, kf1), fsm(w11, w12, kb1), fsm(w22, w21, kf2), fsm(w21, w22, kb2), fsm(w32, w31, kf3), fsm(w31, w32, kb3),
            fieldResY, fieldResX, ctx, queue, deltaProgram)

    add!(buff_wlap, buff_dW, buff_dW)

    ##
    # Calculate dF
    ##

    deltaCL!(buff_row1, buff_row2, buff_row3, buff_row4, buff_row5, buff_row6, buff_dF,
            fsm(f12, f11, kf1), fsm(f11, f12, kb1), fsm(f22, f21, kf2), fsm(f21, f22, kb2), fsm(f32, f31, kf3), fsm(f31, f32, kb3),
            fieldResY, fieldResX, ctx, queue, deltaProgram)

    add!(buff_flap, buff_dF, buff_dF)

    cl.copy!(queue, dF, buff_dF)

    dF[FrefillBinMask] += flowRateF * (saturationF - Ffield[FrefillBinMask])

    dT = 0

    # update values
    smul!(stepIntegration, buff_dA, buff_dA)
    add!(buff_afield, buff_dA, buff_afield)
    cl.copy!(queue, Afield, buff_afield)

    Ffield += dF * stepIntegration
    #Tfield += dT * stepIntegration

    smul!(stepIntegration, buff_dM, buff_dM)
    add!(buff_mfield, buff_dM, buff_mfield)
    cl.copy!(queue, Mfield, buff_mfield)

    smul!(stepIntegration, buff_dW, buff_dW)
    add!(buff_wfield, buff_dW, buff_wfield)
    cl.copy!(queue, Wfield, buff_wfield)

    #save values for visualization
    meanAField = mean(Afield)
    meanMField = mean(Mfield)
    Avec[iround(t/stepIntegration)] = meanAField
    Fvec[iround(t/stepIntegration)] = mean(Ffield)
    Tvec[iround(t/stepIntegration)] = mean(Tfield)
    Mvec[iround(t/stepIntegration)] = meanMField
    Wvec[iround(t/stepIntegration)] = mean(Wfield)
    if enableVis
      DAvec[iround(t/stepIntegration)] = sumabs(cl.read(queue, buff_dA))
      DMvec[iround(t/stepIntegration)] = sumabs(cl.read(queue, buff_dM))
    end

    if t in tStoreFields
      history_A[:, :, iHistory] = Afield
      history_F[:, :, iHistory] = Ffield
      history_T[:, :, iHistory] = Tfield
      history_M[:, :, iHistory] = Mfield
      history_M_pot[:, :, iHistory] = cl.read(queue, buff_mpot)
      history_W[:, :, iHistory] = Wfield
      history_dir[:, :, iHistory] = directionfield
      iHistory += 1
    end


    if enableVis && (t % visInterval == 0)
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
      axis([0, timeTotal, 0, 0.4])

      subplot(244)
      plot(tx, DMvec, "-", linewidth=2)
      title("DMvec")
      axis([0, timeTotal, 0, 0.4])

      subplot(245)
      pcolormesh(Mfield, vmin=0, vmax=0.6)
      title("Mfield")

      subplot(246)
      pcolormesh(Afield, vmin=0, vmax=0.6)
      title("Afield")

      subplot(247)
      pcolormesh(Ffield, vmin=0, vmax=1)
      title("Ffield")

      # if enableDirFieldVis
        subplot(248)
        title("DirectionField")

        U = cos(directionfield)
        V = sin(directionfield)
        plt.quiver([1:fieldResY], [1:fieldResX], U, V, linewidth=1.5, headwidth = 0.5)
      # end
      yield()
    end
    end

    t += stepIntegration
    next!(p)
end # While

###
# Find the reason we terminated
###

print("Finished because after $t: ")
if t > timeTotal
  println("Time done")
elseif meanMField >= 2
  println("mean of MField exceeds 2")
elseif meanMField <= 0.001
  println("mean of MField is smaller than 0.001")
elseif meanAField >= 2
  println("mean of AField exceeds 2")
elseif meanAField <= 0.001
  println("mean of AField is smaller than 0.001")
else
  println("$(t <= timeTotal) && $(meanMField < 2) && $(meanMField > 0.001) && $(meanAField < 2) && $(meanAField > 0.001)")
end

###
# Export
###
dt=now()
matwrite("results/$(year(dt))-$(month(dt))-$(day(dt))_$(hour(dt)):$(minute(dt)):$(second(dt)).mat", {
    "history_A" => history_A,
    "history_F" => history_F,
    "history_T" => history_T,
    "history_M" => history_M,
    "history_M_pot" => history_M_pot,
    "history_W" => history_W,
    "history_dir" => history_dir,
    "Avec" => Avec,
    "Fvec" => Fvec,
    "Tvec" => Tvec,
    "Mvec" => Mvec,
    "Wvec" => Wvec,
    "DAvec" => DAvec,
    "DMvec" => DMvec,
    "fieldSize" => fieldSize,
    "fieldRes" => fieldRes,
    "fieldSizeY" => fieldSizeY,
    "fieldSizeX" => fieldSizeX,
    "fieldResY" => fieldResY,
    "fieldResX" => fieldResX,
    "timeTotal" => timeTotal,
    "stepIntegration" => stepIntegration,
    "visInterval" => visInterval,
    "stepVisualization" => stepVisualization,
    "xc" => xc,
    "yc" => yc,
    "mR" => mR,
    "mT" => mT,
    "fD" => fD,
    "avgM" => avgM,
    "avgA" => avgA,
    "avgF" => avgF,
    "avgT" => avgT,
    "tearTime1" => tearTime1,
    "tearTime2" => tearTime2,
    "tearSize1" => tearSize1,
    "tearSize2" => tearSize2,
    "diffusionF" => diffusionF,
    "diffusionT" => diffusionT,
    "diffusionA" => diffusionA,
    "diffusionM" => diffusionM,
    "diffusionW" => diffusionW,
    "attractionRate" => attractionRate,
    "MW_repulsion" => MW_repulsion,
    "long_direction" => long_direction,
    "decayA" => decayA,
    "decayM" => decayM,
    "flowRateF" => flowRateF,
    "saturationF" => saturationF,
    "noiseMean" => noiseMean,
    "noiseSD" => noiseSD,
    "a11" => a11,
    "a12" => a12,
    "a21" => a21,
    "a22" => a22,
    "a31" => a31,
    "a32" => a32,
    "f11" => f11,
    "f12" => f12,
    "f21" => f21,
    "f22" => f22,
    "f31" => f31,
    "f32" => f32,
    "m11" => m11,
    "m12" => m12,
    "m21" => m21,
    "m22" => m22,
    "m31" => m31,
    "m32" => m32,
    "w11" => w11,
    "w12" => w12,
    "w21" => w21,
    "w22" => w22,
    "w31" => w31,
    "w32" => w32,
    "kf1" => kf1,
    "kb1" => kb1,
    "kf2" => kf2,
    "kb2" => kb2,
    "kf3" => kf3,
    "kb3" => kb3
     })

println("Press any key to exit program.")
readline(STDIN)
end #Function

function fsm(x, y, k)
  if (k == zero(k)) || (x == y)
    return zero(k)
  else
    return ((x-y) * k )
  end
end