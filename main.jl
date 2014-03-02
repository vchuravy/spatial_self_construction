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
include("potential.jl")
include("diffusion.jl")
include("laplacian.jl")
include("align.jl")
include("potentialCL.jl")
include("diffusionCL.jl")
include("addCL.jl")

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

row1 = zeros(T, fieldResY, fieldResX)
row2 = zeros(T, fieldResY, fieldResX)
row3 = zeros(T, fieldResY, fieldResX)
row4 = zeros(T, fieldResY, fieldResX)
row5 = zeros(T, fieldResY, fieldResX)
row6 = zeros(T, fieldResY, fieldResX)

M_pot  = zeros(T, fieldResY, fieldResX)
W_pot = zeros(T, fieldResY, fieldResX)
A_pot = zeros(T, fieldResY, fieldResX)

W_lap = zeros(T, fieldResY, fieldResX)
A_lap = zeros(T, fieldResY, fieldResX)
M_lap = zeros(T, fieldResY, fieldResX)
F_lap = zeros(T, fieldResY, fieldResX)

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

buff_wlap = cl.Buffer(T, ctx, :rw, bufferSize)
buff_alap = cl.Buffer(T, ctx, :rw, bufferSize)
buff_mlap = cl.Buffer(T, ctx, :rw, bufferSize)

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

    # Laplacian for diffusion
    F_lap = diffF * LaPlacian(Ffield);
    T_lap = diffF * LaPlacian(Tfield);

    # update direction field based on alignment
    directionfield = align(Mfield, directionfield, attractionRate, stepIntegration)

    ###
    # reactions
    ##

    row1 = calcRow(Mfield, m11, Afield, a11, Ffield, f11, Wfield, w11)
    row2 = calcRow(Mfield, m12, Afield, a12, Ffield, f12, Wfield, w12)
    row3 = calcRow(Mfield, m21, Afield, a21, Ffield, f21, Wfield, w21)
    row4 = calcRow(Mfield, m22, Afield, a22, Ffield, f22, Wfield, w22)
    row5 = calcRow(Mfield, m31, Afield, a31, Ffield, f31, Wfield, w31)
    row6 = calcRow(Mfield, m32, Afield, a32, Ffield, f32, Wfield, w32)

    dA  = add!(subtract!(divide!(
          addRows(a11, a12, a21, a22, a31, a32, kf1, kb1, kf2, kb2, kf3, kb3, row1, row2, row3, row4, row5, row6),
          (1+Mfield)), decayA*Afield), A_lap)

    dM  = add!(subtract!(divide!(
          addRows(m11, m12, m21, m22, m31, m32, kf1, kb1, kf2, kb2, kf3, kb3, row1, row2, row3, row4, row5, row6),
          (1+Afield)), decayM*Mfield), M_lap)

    dW  = add!(addRows(w11, w12, w21, w22, w31, w32, kf1, kb1, kf2, kb2, kf3, kb3, row1, row2, row3, row4, row5, row6), W_lap)
    dF  = add!(addRows(f11, f12, f21, f22, f31, f32, kf1, kb1, kf2, kb2, kf3, kb3, row1, row2, row3, row4, row5, row6), F_lap)

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

function calcRow(Mfield, m, Afield, a, Ffield, f, Wfield, w)
  return multiply!(multiply!(multiply!( Mfield .^ m, Afield .^ a), Ffield .^ f), Wfield .^ w)
end

function cScale(x, y, k, row)
  if (k == zero(k)) | (x == y)
    return zero(row)
  else
    return (k * (x-y)) * row
  end
end

function addRows(e11, e12, e21, e22, e31, e32, kf1, kb1, kf2, kb2, kf3, kb3, row1, row2, row3, row4, row5, row6)
  return add!(add!(add!(add!(add!(
    cScale(e12, e11, kf1, row1),
    cScale(e11, e12, kb1, row2)),
    cScale(e22, e21, kf2, row3)),
    cScale(e21, e22, kb2, row4)),
    cScale(e32, e31, kf3, row5)),
    cScale(e31, e32, kb3, row5))
end
