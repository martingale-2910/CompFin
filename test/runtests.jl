using Test
using Printf
using Random: randn
using CompFin: GBM, drift, diffusion, compute_euler_step, ddiffusion, compute_milstein_step, compute_runge_kutta_step

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
    euler_actual_result = compute_euler_step(gbm, x0, dt, dwt)

    # Euler-Maruyama
    @test begin
        euler_expected_result = x0 + drift(gbm, x0)*dt + diffusion(gbm, x0)*dwt
        isapprox(euler_actual_result, euler_expected_result; atol=1e-7)
    end

    # Milstein
    @test begin
        milstein_actual_result = compute_milstein_step(gbm, x0, dt, dwt)
        milstein_expected_result = euler_actual_result + 0.5*diffusion(gbm, x0)*ddiffusion(gbm, x0)*(dwt^2 - dt)
        isapprox(milstein_actual_result, milstein_expected_result; atol=1e-7)
    end

    # Runge-Kutta
    @test begin
        runge_kutta_actual_result = compute_runge_kutta_step(gbm, x0, dt, dwt)
        runge_kutta_expected_result = euler_actual_result + 0.5*((diffusion(gbm, compute_euler_step(gbm, x0, dt, dt^2)) - diffusion(gbm, x0))/(dt^2))*(dwt^2 - dt)
        isapprox(runge_kutta_actual_result, runge_kutta_expected_result; atol=1e-7)
    end
end

@testset "Stochastic Simulation Test" begin

    # Single step 
    @test begin
        true
    end

    # Many steps, single value
    @test begin
        true
    end

    # Whole path
    @test begin
        true
    end
end

@testset "Monte Carlo Test" begin

    # Simulate MC values stepwise
    @test begin
        true
    end

    # Simulate MC values pathwise
    @test begin
        true
    end

    # Simulate MC paths stepwise
    @test begin
        true
    end

    # Simulate MC paths pathwise
    @test begin
        true
    end

    # Estimate MC result stepwise
    @test begin
        true
    end

    # Estimate MC result pathwise
    @test begin
        true
    end
end