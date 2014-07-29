include("simulation.jl")
include("jl/functions.jl")

import Simulation
const sim = Simulation

fieldRes = 100
fieldResY = fieldRes
fieldResX = fieldRes

T = Float64

function test_area(dfield, dir, name)
	old_area = create_n4(T)

	areaJl!(dfield, old_area, dir)
	new_area = sim.area(dfield, dir)

	test_area_exact = fill(false, fieldRes, fieldRes)
	test_area_approx = fill(false, fieldRes, fieldRes)

	for i in 1:fieldRes
		for j in 1:fieldRes
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

# Test flow
flow_new = sim.flow(apot_new)
flow_old = flow(apot_new)

test_flow_exact = fill(false, 8, fieldRes, fieldRes)
test_flow_approx = fill(false, 8, fieldRes, fieldRes)

flow_order = [sim.north_west, sim.north, sim.north_east, sim.east, sim.south_east, sim.south, sim.south_west, sim.west]


for i in 1:fieldRes
	for j in 1:fieldRes
		of = toarray(flow_old[i, j])
		nf = Array(Float64, 8)

		for k in 1:8
			nf[k] = flow_new[i,j][flow_order[k]...]
		end

		test_flow_exact[:, i, j] = map(==, of ,nf)
		test_flow_approx[:, i, j] = map(isapprox, of, nf)
	end
end

println("Flow test result | exact: $(all(test_flow_exact)) approx: $(all(test_flow_approx))")

# Diffusion

alap_old = create(T)
alap_new = create(T)

diffusionJl!(afield, apot_new, alap_old)
sim.diffusion!(alap_new, afield, flow_new)

test_alap_exact = all(map(==, alap_old, alap_new))
test_alap_approx = all(map(isapprox, alap_old, alap_new))

println("Diffusion test result | exact: $test_alap_exact approx: $test_alap_approx")

# Laplacian

flap_old = create(T)
LaPlacianJl!(afield, flap_old)

flap_new = sim.la_placian(afield)

test_flap_exact = all(map(==, flap_old, flap_new))
test_flap_approx = all(map(isapprox, flap_old, flap_new))

println("LaPlacian test result | exact: $test_flap_exact approx: $test_flap_approx")

dfield_old = create(T)
alignJl!(afield, dfield, dfield_old, 0.75, 0.5)
dfield_new = sim.align(afield, dfield, 0.75, 0.5)

test_dfield_exact = all(map(==, dfield_old, dfield_new))
test_dfield_approx = all(map(isapprox, dfield_old, dfield_new))

println("Align test result | exact: $test_dfield_exact approx: $test_dfield_approx")