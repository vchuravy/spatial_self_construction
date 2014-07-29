include("simulation.jl")
include("jl/functions.jl")

import Simulation
const sim = Simulation

fieldRes = 10
fieldResY = fieldRes
fieldResX = fieldRes

T = Float64

function test_area(dfield, dir, name)
	old_area = create_n4(T)

	areaJl!(dfield, old_area, dir)
	new_area = sim.area(dfield, dir)

	test_area_exact = fill(false, fieldRes, fieldRes)
	test_area_approx = fill(false, fieldRes, fieldRes)

	for i in 1:size(old_area, 1)
		for j in 1:size(old_area, 2)
			oa = toarray(old_area[i, j])
			na = new_area[:, i, j]

			test_area_exact[i, j] = all(map(==, oa ,na))
			test_area_approx[i, j] = all(map(isapprox, oa, na))
		end
	end

	println("Area test $name result | exact: $(all(test_area_exact)) approx: $(all(test_area_approx))")
	(old_area, new_area)
end

dfield = pi/4 * ones(T, fieldRes, fieldRes)

test_area(dfield, 1, "dir=1")
test_area(dfield, 0.5, "dir=0.5")

dfield = pi .* rand(T, fieldRes, fieldRes)
old_area, new_area = test_area(dfield, 1, "random")

# Test potential
afield = rand(T, fieldRes, fieldRes)
bfield = rand(T, fieldRes, fieldRes)

apot_old = create(T)
bpot_old = create(T)
apot_new = create(T)
bpot_new = create(T)

sim.potential!(apot_new, bpot_new, afield, bfield, new_area, .1)
potentialJl!(afield, bfield, old_area, apot_old, bpot_old, .1)


test_bpot_exact = all(map(==, bpot_old, bpot_new))
test_bpot_approx = all(map(isapprox, bpot_old, bpot_new))

test_apot_exact = all(map(==, apot_old, apot_new))
test_apot_approx = all(map(isapprox, apot_old, apot_new))

println("Potential test result | exact: $test_bpot_exact & $test_apot_exact approx: $test_bpot_approx & $test_apot_approx")

