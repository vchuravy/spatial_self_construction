using PyPlot
using MAT

function replay(fileName)

fileConfig = matread(fileName)

history_W = fileConfig["history_W"]
history_A = fileConfig["history_A"]
history_M = fileConfig["history_M"]
history_F = fileConfig["history_F"]
history_dir = fileConfig["history_dir"]

#history_DAvec = fileConfig["DAvec"]
#history_DMvec = fileConfig["DMvec"]

hold(false)
for t in 1:size(history_M, 3)
    subplot(151)
    pcolormesh(history_M[:,:,t], vmin=0, vmax=0.6)
    title("Mfield")

    subplot(152)
    pcolormesh(history_A[:,:,t], vmin=0, vmax=0.6)
    title("Afield")

    subplot(153)
    pcolormesh(history_F[:,:,t], vmin=0, vmax=1)
    title("Ffield")

    subplot(154)
    pcolormesh(history_W[:,:,t], vmin=0, vmax=1)
    title("Wfield")

    subplot(155)
    title("DirectionField")
    directionfield = history_dir[:,:,t]

    U = cos(directionfield)
    V = sin(directionfield)
    d1, d2 = size(directionfield)
    plt.quiver([1:d1], [1:d2], U, V, linewidth=1.5, headwidth = 0.5)
    yield()
end
end

if length(ARGS) == 1
    replay(first(ARGS))
end