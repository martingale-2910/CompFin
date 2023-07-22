module CompFin

using Random: randn, default_rng
using Statistics: mean, std
using LinearAlgebra: I, Tridiagonal

# SDEs

abstract type SDE end

struct GBM <: SDE
    mu::Float64
    sigma::Float64
end

function drift(gbm::GBM, x::Float64)
    return gbm.mu*x
end

function diffusion(gbm::GBM, x::Float64)
    return gbm.sigma*x
end

function ddiffusion(gbm::GBM, x::Float64)
    return gbm.sigma
end

# SCHEMEs

function compute_euler_step(sde::SDE, x0::Float64, dt::Float64, dwt::Float64)
    return x0 + drift(sde, x0)*dt + diffusion(sde, x0)*dwt
end

function compute_milstein_step(sde::SDE, x0::Float64, dt::Float64, dwt::Float64)
    return compute_euler_step(sde, x0, dt, dwt) + 0.5*diffusion(sde, x0)*ddiffusion(sde, x0)*(dwt^2 - dt)
end

function compute_runge_kutta_step(sde::SDE, x0::Float64, dt::Float64, dwt::Float64)
    x0_corr = compute_euler_step(sde, x0, dt, dt^2)
    return compute_euler_step(sde, x0, dt, dwt) + 0.5*((diffusion(sde, x0_corr) - diffusion(sde, x0))/(dt^2))*(dwt^2 - dt)
end

# SIMULATIONs

function simulate_step(sde::SDE, x0::Float64, dt::Float64, step_scheme::Function; rng=default_rng())
    dwt = sqrt(dt)*randn(rng)
    return step_scheme(sde, x0, dt, dwt)
end

function simulate_value(sde::SDE, x0::Float64, dt::Float64, step_scheme::Function, n_steps::Int64; rng=default_rng())
    return foldl((xi, i) -> simulate_step(sde, xi, dt, step_scheme; rng=rng), 1:n_steps; init=x0)
end

function simulate_path(sde::SDE, x0::Float64, dt::Float64, step_scheme::Function, n_steps::Int64; rng=default_rng())
    return [x0 hcat(accumulate((xi, i) -> simulate_step(sde, xi, dt, step_scheme; rng=rng), 1:n_steps; init=x0)...)]
end

# MC

function simulate_mc_values(sde::SDE, x0::Float64, dt::Float64, step_scheme::Function, n_steps::Int64, n_paths::Int64, pathwise::Bool; rng=default_rng(), as_matrix::Bool=true)
    if pathwise
        mc_values = simulate_value.(Ref(sde), x0*ones(n_paths, 1), Ref(dt), Ref(step_scheme), Ref(n_steps); rng=rng)
    else
        mc_values = foldl((xi, i) -> simulate_step.(Ref(sde), xi, Ref(dt), Ref(step_scheme); rng=rng), 1:n_steps; init=x0*ones(n_paths, 1))
    end
    if as_matrix
        return mc_values
    else
        return vcat(mc_values...)
    end
end

function simulate_mc_paths(sde::SDE, x0::Float64, dt::Float64, step_scheme::Function, n_steps::Int64, n_paths::Int64, pathwise::Bool; rng=default_rng(), as_matrix::Bool=true)
    if pathwise
        mc_paths = vcat(simulate_path.(Ref(sde), x0*ones(n_paths, 1), Ref(dt), Ref(step_scheme), Ref(n_steps); rng=rng)...)
    else
        mc_paths = [x0*ones(n_paths, 1) hcat(accumulate((xi, i) -> simulate_step.(Ref(sde), xi, Ref(dt), Ref(step_scheme); rng=rng), 1:n_steps; init=x0*ones(n_paths, 1))...)]
    end
    if as_matrix
        return mc_paths
    elseif pathwise
        return [mc_paths[i, :] for i in 1:size(mc_paths)[1]]
    else
        return [mc_paths[:, i] for i in 1:size(mc_paths)[2]]
    end
end

function estimate_mc_result(sde::SDE, x0::Float64, dt::Float64, step_scheme::Function, n_steps::Int64, n_paths::Int64, pathwise::Bool; rng=default_rng(), as_matrix::Bool=true)
    x = simulate_mc_values(sde, x0, dt, step_scheme, n_steps, n_paths, pathwise; rng=rng, as_matrix=as_matrix)
    return mean(x), std(x)
end 

# PDE

abstract type PDE end

struct HeatEquation <: PDE
    alpha::Float64
    initial_condition::Function
    left_boundary_condition::Function
    right_boundary_condition::Function
end

# FDM

# theta == 0.0 <=> Explicit method
# theta == 1.0 <=> Implicit method
# theta == 0.5 <=> Crank-Nicholson method
function solve_pde(heat_equation::HeatEquation, xmin::Float64, xmax::Float64, nx::Int64, tmin::Float64, tmax::Float64, nt::Int64; theta::Float64=0.0)::Matrix{Float64}
    dt = (tmax - tmin)/nt
    dx = (xmax - xmin)/nx
    lambda = (heat_equation.alpha)*dt/(dx^2)

    t = dt*collect(0:nt)
    x = dx*collect(0:nx)
    u = fill(0.0, nt + 1, nx + 1)

    u[:, 1] = heat_equation.left_boundary_condition.(t)
    u[:, end] = heat_equation.right_boundary_condition.(t)
    u[1, :] = heat_equation.initial_condition.(x)

    # TODO: Use SparseArrays once they resolve '[...] self-adjoint sparse system solve not implemented for sparse rhs B. [...]'
    M = lambda*Tridiagonal(vec(ones(nx - 2, 1)), -2*vec(ones(nx - 1, 1)), vec(ones(nx - 2, 1)))
    B = I - theta*M
    A = I + (1 - theta)*M
    for i = 1:nt
        m[[1, end]] = [u[i + 1, 1], u[i + 1, end]]
        b = theta*(-lambda)*m
        u[i + 1, 2:end - 1] = B(A*u[i, 2:end - 1] + a - b)
    end

    return u
end

end # module CompFin
