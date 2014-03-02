###
# Main
###

using MAT
using NumericExtensions
using ProgressMeter
using Datetime
using PyPlot
import OpenCL
const cl = OpenCL

include("config.jl")
include("drawcircle.jl")
#include("potential.jl")
#include("diffusion.jl")
include("laplacian.jl")
#include("align.jl")
include("cl/potentialCL.jl")
include("cl/diffusionCL.jl")
include("cl/addCL.jl")
include("cl/alignCL.jl")
include("cl/calcRowCL.jl")
include("cl/deltaCL.jl")

###
# set up initial configuration
###

function main(;enableVis = false, enableDirFieldVis = false, fileName = "", loadTime = 0)
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

    simulation(enableVis, enableDirFieldVis, fileName, loadTime, device, ctx, queue, CL64BIT, CL64BIT ? Float64 : Float32)

end

function simulation{T <: FloatingPoint}(enableVis, enableDirFieldVis, fileName, loadTime, device, ctx, queue, CL64BIT, :: Type{T} )
# initialize membrane fields
Afield = zeros(T, fieldResY, fieldResX)
Mfield = zeros(T, fieldResY, fieldResX)
Ffield = avgF * ones(T, fieldResY, fieldResX)
Tfield = zeros(T, fieldResY, fieldResX)
Wfield = ones(T, fieldResY, fieldResX)
directionfield = pi/2 * ones(T, fieldResY, fieldResX)

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
  dataFile = matopen("data/$(fileName).mat")
  histWfield = read(dataFile, "history_W")
  histAfield = read(dataFile, "history_A")
  histMfield = read(dataFile, "history_M")
  histFfield = read(dataFile, "history_F")

  Wfield = histWfield[:,:,loadTime]
  Afield = histAfield[:,:,loadTime]
  Mfield = histMfield[:,:,loadTime]
  Ffield = histFfield[:,:,loadTime]
end

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

M_pot  = zeros(T, fieldResY, fieldResX)
W_pot = zeros(T, fieldResY, fieldResX)
A_pot = zeros(T, fieldResY, fieldResX)

W_lap = zeros(T, fieldResY, fieldResX)
A_lap = zeros(T, fieldResY, fieldResX)
M_lap = zeros(T, fieldResY, fieldResX)
F_lap = zeros(T, fieldResY, fieldResX)

dA = zeros(T, fieldResY, fieldResX)
dW = zeros(T, fieldResY, fieldResX)
dM = zeros(T, fieldResY, fieldResX)
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
    # calculate potential based on repulsion %
    ###
    cl.copy!(queue, buff_mfield, Mfield)
    cl.copy!(queue, buff_wfield, Wfield)
    cl.copy!(queue, buff_afield, Afield)
    cl.copy!(queue, buff_ffield, Ffield)
    cl.copy!(queue, buff_dfield, directionfield)

    potentialCL!(buff_mfield, buff_wfield, buff_dfield, buff_mpot1, buff_wpot, MW_repulsion, long_direction, fieldResY, fieldResX, ctx, queue, potentialProgram)
    potentialCL!(buff_mfield, buff_afield, buff_dfield, buff_mpot2, buff_apot, MA_repulsion, long_direction, fieldResY, fieldResX, ctx, queue, potentialProgram)

    addCL!(buff_mpot1, buff_mpot2, buff_mpot, fieldResY, fieldResX, ctx, queue, addProgram)

    ###
    # move molecules and update directionality
    ###

    diffusionCL!(buff_mfield, buff_mpot, buff_mlap, fieldResY, fieldResX, ctx, queue, diffusionProgram)
    diffusionCL!(buff_wfield, buff_wpot, buff_wlap, fieldResY, fieldResX, ctx, queue, diffusionProgram)
    diffusionCL!(buff_afield, buff_apot, buff_alap, fieldResY, fieldResX, ctx, queue, diffusionProgram)

    cl.copy!(queue, W_lap, buff_wlap)
    cl.copy!(queue, A_lap, buff_alap)
    cl.copy!(queue, M_lap, buff_mlap)

    multiply!(W_lap, diffW)
    multiply!(A_lap, diffA)
    multiply!(M_lap, diffM)

    # Laplacian for diffusion
    F_lap = diffF * LaPlacian(Ffield)
    T_lap = diffT * LaPlacian(Tfield)

    # update direction field based on alignment
    alignCL!(buff_mfield, buff_dfield, buff_ndfield, attractionRate, stepIntegration, fieldResY, fieldResX, ctx, queue, alignProgram)
    cl.copy!(queue, directionfield, buff_ndfield)
    ###
    # reactions
    ##

    calcRowCL!(buff_mfield, buff_afield, buff_ffield, buff_wfield, buff_row1, m11, a11, f11, w11, fieldResY, fieldResX, ctx, queue, rowProgram)
    calcRowCL!(buff_mfield, buff_afield, buff_ffield, buff_wfield, buff_row2, m12, a12, f12, w12, fieldResY, fieldResX, ctx, queue, rowProgram)
    calcRowCL!(buff_mfield, buff_afield, buff_ffield, buff_wfield, buff_row3, m21, a21, f21, w21, fieldResY, fieldResX, ctx, queue, rowProgram)
    calcRowCL!(buff_mfield, buff_afield, buff_ffield, buff_wfield, buff_row4, m22, a22, f22, w22, fieldResY, fieldResX, ctx, queue, rowProgram)
    calcRowCL!(buff_mfield, buff_afield, buff_ffield, buff_wfield, buff_row5, m31, a31, f31, w31, fieldResY, fieldResX, ctx, queue, rowProgram)
    calcRowCL!(buff_mfield, buff_afield, buff_ffield, buff_wfield, buff_row6, m32, a32, f32, w32, fieldResY, fieldResX, ctx, queue, rowProgram)

    deltaCL!(buff_row1, buff_row2, buff_row3, buff_row4, buff_row5, buff_row6, buff_dA,
            r(a12, a11, kf1), r(a11, a12, kb1), r(a22, a21, kf2), r(a21, a22, kb2), r(a32, a31, kf3), r(a31, a32, kb3),
            fieldResY, fieldResX, ctx, queue, deltaProgram)

    cl.copy!(queue, dA, buff_dA)

    dA = add!(subtract!(divide!(dA, (1+Mfield)),decayA*Afield), A_lap)

    deltaCL!(buff_row1, buff_row2, buff_row3, buff_row4, buff_row5, buff_row6, buff_dM,
            r(m12, m11, kf1), r(m11, m12, kb1), r(m22, m21, kf2), r(m21, m22, kb2), r(m32, m31, kf3), r(m31, m32, kb3),
            fieldResY, fieldResX, ctx, queue, deltaProgram)

    cl.copy!(queue, dM, buff_dM)

    dM = add!(subtract!(divide!(dM, (1+Afield)),decayM*Mfield), M_lap)

    deltaCL!(buff_row1, buff_row2, buff_row3, buff_row4, buff_row5, buff_row6, buff_dW,
            r(w12, w11, kf1), r(w11, w12, kb1), r(w22, w21, kf2), r(w21, w22, kb2), r(w32, w31, kf3), r(w31, w32, kb3),
            fieldResY, fieldResX, ctx, queue, deltaProgram)

    addCL!(buff_wlap, buff_dW, buff_dW, fieldResY, fieldResX, ctx, queue, addProgram)

    cl.copy!(queue, dW, buff_dW)

    deltaCL!(buff_row1, buff_row2, buff_row3, buff_row4, buff_row5, buff_row6, buff_dF,
            r(f12, f11, kf1), r(f11, f12, kb1), r(f22, f21, kf2), r(f21, f22, kb2), r(f32, f31, kf3), r(f31, f32, kb3),
            fieldResY, fieldResX, ctx, queue, deltaProgram)

    addCL!(buff_flap, buff_dF, buff_dF, fieldResY, fieldResX, ctx, queue, addProgram)

    cl.copy!(queue, dF, buff_dF)

    dF[FrefillBinMask] += flowRateF * (saturationF - Ffield[FrefillBinMask])

    dT = 0


    # update values
    Afield += dA * stepIntegration
    Ffield += dF * stepIntegration
    Tfield += dT * stepIntegration
    Mfield += dM * stepIntegration
    Wfield += dW * stepIntegration

    #save values for visualization
    meanAField = mean(Afield)
    meanMField = mean(Mfield)
    Avec[iround(t/stepIntegration)] = meanAField
    Fvec[iround(t/stepIntegration)] = mean(Ffield)
    Tvec[iround(t/stepIntegration)] = mean(Tfield)
    Mvec[iround(t/stepIntegration)] = meanMField
    Wvec[iround(t/stepIntegration)] = mean(Wfield)
    if enableVis
      DAvec[iround(t/stepIntegration)] = sumsq(dA)
      DMvec[iround(t/stepIntegration)] = sumsq(dM)
    end

    if t in tStoreFields
      history_A[:, :, iHistory] = Afield
      history_F[:, :, iHistory] = Ffield
      history_T[:, :, iHistory] = Tfield
      history_M[:, :, iHistory] = Mfield
      history_M_pot[:, :, iHistory] = M_pot
      history_W[:, :, iHistory] = Wfield
      history_dir[:, :, iHistory] = directionfield
      iHistory += 1
    end


    if enableVis && (t % visInterval == 0)
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

      subplot(244)
      plot(tx, DMvec, "-", linewidth=2)
      title("DMvec")

      subplot(245)
      pcolormesh(Mfield)
      title("Mfield")

      subplot(246)
      pcolormesh(Afield)
      title("Afield")

      subplot(247)
      pcolormesh(Ffield)
      title("Ffield")

      if enableDirFieldVis
        subplot(248)
        title("DirectionField")

        U = cos(directionfield)
        V = sin(directionfield)
        plt.streamplot([1:fieldResY], [1:fieldResX], U, V)
      end
      yield()
    end

    t += stepIntegration
    next!(p)
end # While

###
# Find the reason we terminated
###

print("Finished because: ")
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
end

###
# Export
###
matwrite("results/$(now()).mat", {
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
    "DMvec" => DMvec})
end #Function

function r(x, y, k)
  if (k == zero(k)) || (x == y)
    return zero(k)
  else
    return (k * (x-y))
  end
end
