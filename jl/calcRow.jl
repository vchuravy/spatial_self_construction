function calcRowJl!(
    M :: Matrix, A :: Matrix, F :: Matrix, W :: Matrix,
    Out :: Matrix,
    m :: Real, _a :: Real, f :: Real, w :: Real)

    d1, d2 = size(M)

    for j in 1:d2
      for i in 1:d1
        Out[i,j] = pow(M[i, j], m) * pow(A[i, j], _a) * pow(F[i, j], f) * pow(W[i, j], w)
      end
    end
end

pow(x, y) = x ^ y