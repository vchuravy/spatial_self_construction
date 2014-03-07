function deltaJl!(
    R1 :: Matrix, R2 :: Matrix, R3 :: Matrix, R4 :: Matrix, R5 :: Matrix, R6 :: Matrix,
    Out :: Matrix,
    r1 :: Real, r2 :: Real, r3 :: Real , r4 :: Real, r5 :: Real, r6 :: Real)

    d1, d2 = size(Out)
    for j in 1:d2
      for i in 1:d1
        Out[i,j] = r1 * R1[i,j] + r2 * R2[i,j] + r3 * R3[i,j] + r4 * R4[i,j] + r5 * R5[i,j] + r6 * R6[i,j]
      end
    end
end