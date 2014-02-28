###
# Main
###

using MAT
using NumericExtensions
using ProgressMeter
using Datetime

include("config.jl")
include("drawcircle.jl")
include("potential.jl")
include("diffusion.jl")
include("laplacian.jl")
include("align.jl")

###
# set up initial configuration
###

# initialize membrane fields
Afield = zeros(fieldResY, fieldResX)
Mfield = zeros(fieldResY, fieldResX)
Ffield = avgF * ones(fieldResY, fieldResX)
Tfield = zeros(fieldResY, fieldResX)
Wfield = ones(fieldResY, fieldResX)
directionfield = pi/2 * ones(fieldResY, fieldResX)

# draw membrane circle
M_circ = zeros(fieldResY, fieldResX)

for r in mR:0.01:(mR+mT-1)
    drawcircle!(M_circ, xc, yc, r)
end
Mfield += avgM * M_circ + 0.01* rand(fieldResY, fieldResX)

# fill in with autocatalyst
A_circ = zeros(fieldResY, fieldResX)

for r in 0:0.01:mR
    drawcircle!(A_circ, xc, yc, r)
end
Afield += avgA * A_circ

Wfield -= (Mfield + Afield)

# wFile = matopen("data/w4f5.mat")
# Wfield = read(wFile, "Wfield")

# aFile = matopen("data/a4f5.mat")
# Afield = read(aFile, "Afield")

# mFile = matopen("data/m4f5.mat")
# Mfield = read(mFile, "Mfield")

# fFile = matopen("data/f4f5.mat")
# Ffield = read(fFile, "Ffield")

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
history_A = zeros(fieldResY, fieldResX, length(tStoreFields))
history_F = zeros(fieldResY, fieldResX, length(tStoreFields))
history_T = zeros(fieldResY, fieldResX, length(tStoreFields))
history_M = zeros(fieldResY, fieldResX, length(tStoreFields))
history_M_pot = zeros(fieldResY, fieldResX, length(tStoreFields))
history_W = zeros(fieldResY, fieldResX, length(tStoreFields))
history_dir = zeros(fieldResY, fieldResX, length(tStoreFields))

# index of the current position in the history matrices
iHistory = 1

#vectors to save global concentrations across time

vecL = iround(timeTotal / stepIntegration)
Avec = zeros(1, vecL)
Fvec = zeros(1, vecL)
Tvec = zeros(1, vecL)
Mvec = zeros(1, vecL)
Wvec = zeros(1, vecL)

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


qX = zeros(fieldResX*fieldResY, 1)
qY = zeros(fieldResX*fieldResY, 1)
qU = zeros(fieldResX*fieldResY, 1)
qV = zeros(fieldResX*fieldResY, 1)

ind=1
for xx in 1:fieldResX
    for yy in 1:fieldResY
        qX[ind] = xx
        qY[ind] = yy
        ind += 1
    end
end


###
# Simulation
###

t = 1
p = Progress(length(tx), 1)

meanMField = mean(Mfield)
meanAField = mean(Afield)

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

    A_pot = zeros(fieldResY, fieldResX)

    M_pot1, W_pot = potential(Mfield, Wfield, directionfield, MW_repulsion, long_direction)
    M_pot2, A_pot = potential(Mfield, Afield, directionfield, MA_repulsion, long_direction)

    M_pot = M_pot1 + M_pot2

    ###
    # move molecules and update directionality
    ###

    M_lap = diffM * diffusion(Mfield, M_pot)
    W_lap = diffW * diffusion(Wfield, W_pot)
    A_lap = diffA * diffusion(Afield, A_pot)

    # Laplacian for diffusion
    F_lap = diffF * LaPlacian(Ffield);
    T_lap = diffF * LaPlacian(Tfield);

    # update direction field based on alignment
    directionfield = align(Mfield, directionfield, attractionRate, stepIntegration)

    ###
    # reactions
    ##

    dA  = A_lap +
          ((a12-a11)*kf1*Mfield.^m11.*Afield.^a11.*Ffield.^f11.*Wfield.^w11 +
          (a11-a12)*kb1*Mfield.^m12.*Afield.^a12.*Ffield.^f12.*Wfield.^w12)./(1+Mfield) -
          decayA*Afield

    dM  = M_lap +
          ((m12-m11)*kf1*Mfield.^m11.*Afield.^a11.*Ffield.^f11.*Wfield.^w11 +
          (m11-m12)*kb1*Mfield.^m12.*Afield.^a12.*Ffield.^f12.*Wfield.^w12  +
          (m22-m21)*kf2*Mfield.^m21.*Afield.^a21.*Ffield.^f21.*Wfield.^w21 +
          (m21-m22)*kb2*Mfield.^m22.*Afield.^a22.*Ffield.^f22.*Wfield.^w22 +
          (m32-m31)*kf3*Mfield.^m31.*Afield.^a31.*Ffield.^f31.*Wfield.^w31 +
          (m31-m32)*kb3*Mfield.^m32.*Afield.^a32.*Ffield.^f32.*Wfield.^w32)./(1+Afield) -
          decayM*Mfield

    dW  = W_lap +
          (w12-w11)*kf1*Mfield.^m11.*Afield.^a11.*Ffield.^f11.*Wfield.^w11 +
          (w11-w12)*kb1*Mfield.^m12.*Afield.^a12.*Ffield.^f12.*Wfield.^w12 +
          (w22-w21)*kf2*Mfield.^m21.*Afield.^a21.*Ffield.^f21.*Wfield.^w21 +
          (w21-w22)*kb2*Mfield.^m22.*Afield.^a22.*Ffield.^f22.*Wfield.^w22 +
          (w32-w31)*kf3*Mfield.^m31.*Afield.^a31.*Ffield.^f31.*Wfield.^w31 +
          (w31-w32)*kb3*Mfield.^m32.*Afield.^a32.*Ffield.^f32.*Wfield.^w32

    dF  = F_lap +
          (f12-f11)*kf1*Mfield.^m11.*Afield.^a11.*Ffield.^f11.*Wfield.^w11 +
          (f11-f12)*kb1*Mfield.^m12.*Afield.^a12.*Ffield.^f12.*Wfield.^w12 +
          (f22-f21)*kf2*Mfield.^m21.*Afield.^a21.*Ffield.^f21.*Wfield.^w21 +
          (f21-f22)*kb2*Mfield.^m22.*Afield.^a22.*Ffield.^f22.*Wfield.^w22 +
          (f32-f31)*kf3*Mfield.^m31.*Afield.^a31.*Ffield.^f31.*Wfield.^w31 +
          (f31-f32)*kb3*Mfield.^m32.*Afield.^a32.*Ffield.^f32.*Wfield.^w32

    dF[Frefill .> 0.5] += flowRateF * (saturationF - Ffield[Frefill .> 0.5])

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


    if t % visInterval == 0

    end

    t += stepIntegration
    next!(p)
end # While

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
    "Wvec" => Wvec})