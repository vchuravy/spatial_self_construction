###
# Main
###

include("config.jl")
include("drawcircle.jl")

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

vecL = iceil(timeTotal / stepIntegration)
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