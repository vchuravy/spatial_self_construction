function delta2Jl!(
    delta :: Matrix, AField :: Matrix, BField :: Matrix, Lap :: Matrix,
    Out :: Matrix, decay :: Real)

    d1, d2 = size(Out)
    for j in 1:d2
      for i in 1:d1
        Out[i,j] = delta[i,j] / (1 + AField[i,j]) - decay * BField[i,j] + Lap[i,j]
      end
    end
end