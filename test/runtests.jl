using Test
using Printf
using Random: randn, Xoshiro
using CompFin: GBM, drift, diffusion, compute_euler_step, ddiffusion, compute_milstein_step, compute_runge_kutta_step, simulate_step, simulate_value, simulate_path, simulate_mc_values, simulate_mc_paths, estimate_mc_result, HeatEquation, solve_pde

dt = 0.01
dwt = sqrt(dt)*randn()
mu = 0.05
sigma = 0.2
gbm = GBM(mu, sigma)
x0 = 100.0

@testset "SDE test" begin
    # GBM
    @test begin
        actual_drift = drift(gbm, x0)
        expected_drift = mu*x0
        isapprox(actual_drift, expected_drift; atol=1e-7)

        actual_diffusion = diffusion(gbm, x0)
        expected_diffusion = sigma*x0
        isapprox(actual_diffusion, expected_diffusion; atol=1e-7)

        actual_ddiffusion = ddiffusion(gbm, x0)
        expected_ddiffusion = sigma
        isapprox(actual_ddiffusion, expected_ddiffusion; atol=1e-7)
    end
end

@testset "Stochastic Scheme Test" begin
    actual_euler_result = compute_euler_step(gbm, x0, dt, dwt)

    # Euler-Maruyama
    @test begin
        expected_euler_result = x0 + drift(gbm, x0)*dt + diffusion(gbm, x0)*dwt
        isapprox(actual_euler_result, expected_euler_result; atol=1e-7)
    end

    # Milstein
    @test begin
        actual_milstein_result = compute_milstein_step(gbm, x0, dt, dwt)
        expected_milstein_result = actual_euler_result + 0.5*diffusion(gbm, x0)*ddiffusion(gbm, x0)*(dwt^2 - dt)
        isapprox(actual_milstein_result, expected_milstein_result; atol=1e-7)
    end

    # Runge-Kutta
    @test begin
        actual_runge_kutta_result = compute_runge_kutta_step(gbm, x0, dt, dwt)
        expected_runge_kutta_result = actual_euler_result + 0.5*((diffusion(gbm, compute_euler_step(gbm, x0, dt, dt^2)) - diffusion(gbm, x0))/(dt^2))*(dwt^2 - dt)
        isapprox(actual_runge_kutta_result, expected_runge_kutta_result; atol=1e-7)
    end
end

@testset "Stochastic Simulation Test" begin
    # Single step 
    @test begin
        actual_simulated_step = simulate_step(gbm, x0, dt, compute_euler_step; rng=Xoshiro(1234))
        dwt = sqrt(dt)*randn(Xoshiro(1234))
        expected_simulated_step = compute_euler_step(gbm, x0, dt, dwt)
        isapprox(actual_simulated_step, expected_simulated_step; atol=1e-7)
    end

    n_steps = 100
    # Many steps, single value
    @test begin
        actual_simulated_value = simulate_value(gbm, x0, dt, compute_euler_step, n_steps; rng=Xoshiro(1234))
        expected_simulated_value = x0
        rng = Xoshiro(1234)
        for _ in 1:n_steps
            expected_simulated_value = simulate_step(gbm, expected_simulated_value, dt, compute_euler_step; rng=rng)
        end
        isapprox(actual_simulated_value, expected_simulated_value; atol=1e-7)
    end

    # Whole path
    @test begin
        actual_simulated_path = simulate_path(gbm, x0, dt, compute_euler_step, n_steps; rng=Xoshiro(1234))
        expected_simulated_path = x0*ones(1, n_steps + 1)
        rng = Xoshiro(1234)
        for i in 1:n_steps
            expected_simulated_path[i + 1] = simulate_step(gbm, expected_simulated_path[i], dt, compute_euler_step; rng=rng)
        end
        isapprox(actual_simulated_path, expected_simulated_path; atol=1e-7)
    end
end

@testset "Monte Carlo Test" begin
    n_steps = 100
    n_paths = 1000
    # Simulate MC values stepwise
    @test begin
        actual_mc_values = simulate_mc_values(gbm, x0, dt, compute_euler_step, n_steps, n_paths, false, rng=Xoshiro(1234))
        expected_mc_values = x0*ones(n_paths, 1)
        rng = Xoshiro(1234)
        for _ = 1:n_steps
            for i = 1:n_paths
                expected_mc_values[i] = simulate_step(gbm, expected_mc_values[i], dt, compute_euler_step, rng=rng)
            end
        end
        isapprox(actual_mc_values, expected_mc_values; atol=1e-7)
    end

    # Simulate MC values pathwise
    @test begin
        actual_mc_values = simulate_mc_values(gbm, x0, dt, compute_euler_step, n_steps, n_paths, true, rng=Xoshiro(1234))
        rng = Xoshiro(1234)
        expected_mc_values = x0*ones(n_paths, 1)
        for i = 1:n_paths
            expected_mc_values[i] = simulate_value(gbm, x0, dt, compute_euler_step, n_steps; rng=rng)
        end
        isapprox(actual_mc_values, expected_mc_values; atol=1e-7)
    end

    # Simulate MC paths stepwise as matrix
    @test begin
        actual_mc_paths = simulate_mc_paths(gbm, x0, dt, compute_euler_step, n_steps, n_paths, false, rng=Xoshiro(1234), as_matrix=true)
        expected_mc_paths = x0*ones(n_paths, n_steps + 1)
        rng = Xoshiro(1234)
        for i = 1:n_steps
            for j = 1:n_paths
                expected_mc_paths[j, i + 1] = simulate_step(gbm, expected_mc_paths[j, i], dt, compute_euler_step; rng=rng)
            end
        end
        isapprox(actual_mc_paths, expected_mc_paths; atol=1e-7)
    end

    # Simulate MC paths stepwise as vector of vectors
    @test begin
        actual_mc_paths = simulate_mc_paths(gbm, x0, dt, compute_euler_step, n_steps, n_paths, false, rng=Xoshiro(1234), as_matrix=false)
        size(actual_mc_paths)
        expected_mc_paths = Vector{Vector{Float64}}(undef, n_steps + 1)
        expected_mc_paths[1] = [x0*ones(n_paths, 1)...]
        rng = Xoshiro(1234)
        for i = 1:n_steps
            expected_mc_paths[i + 1] = [zeros(n_paths, 1)...]
            for j = 1:n_paths
                expected_mc_paths[i + 1][j] = simulate_step(gbm, expected_mc_paths[i][j], dt, compute_euler_step; rng=rng)
            end
        end
        isapprox(actual_mc_paths, expected_mc_paths; atol=1e-7)
    end

    # Simulate MC paths pathwise as matrix
    @test begin
        actual_mc_paths = simulate_mc_paths(gbm, x0, dt, compute_euler_step, n_steps, n_paths, true; rng=Xoshiro(1234), as_matrix=true)
        rng = Xoshiro(1234)
        expected_mc_paths = x0*ones(n_paths, n_steps + 1)
        for i = 1:n_paths
            expected_mc_paths[i, :] = simulate_path(gbm, x0, dt, compute_euler_step, n_steps; rng=rng)
        end
        isapprox(actual_mc_paths, expected_mc_paths; atol=1e-7)
    end

    # Simulate MC paths pathwise as vector of vectors
    @test begin
        actual_mc_paths = simulate_mc_paths(gbm, x0, dt, compute_euler_step, n_steps, n_paths, true; rng=Xoshiro(1234), as_matrix=false)
        rng = Xoshiro(1234)
        expected_mc_paths = Vector{Vector{Float64}}(undef, n_paths)
        for i = 1:n_paths
            expected_mc_paths[i] = [simulate_path(gbm, x0, dt, compute_euler_step, n_steps; rng=rng)...]
        end
        isapprox(actual_mc_paths, expected_mc_paths; atol=1e-7)
    end

end

@testset "Explicit Scheme Heat Equation Test" begin
    init_cond = x -> x*(1 - x)
    left_bound = t -> 20*t^2
    right_bound = t -> 10*t

    pde = HeatEquation(1.0, init_cond, left_bound, right_bound)

    x_min = 0.0
    x_max = 1.0
    n_x = 4
    t_min = 0.0
    t_max = 0.03
    n_t = 3

    # Simple Explicit Scheme computation
    @test begin
        actual_u = solve_pde(pde, x_min, x_max, n_x, t_min, t_max, n_t)
        expected_u = [0 0.1875 0.25 0.1875 0; 0.002 0.1675 0.23 0.1675 0.1; 0.008 0.151 0.21 0.1667 0.2; 0.018 0.1376 0.1936 0.179 0.3]
        isapprox(actual_u, expected_u; atol=1e-4)
    end
end