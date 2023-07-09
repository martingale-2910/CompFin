module CompFin

using Random: randn, default_rng
using Statistics: mean, std

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

function simulate_step(sde::SDE, x0::Float64, dt::Float64, step_scheme::Function; rng=default_rng())
    dwt = sqrt(dt)*randn(rng)
    return step_scheme(sde, x0, dt, dwt)
end

function simulate_value(sde::SDE, x0::Float64, dt::Float64, step_scheme::Function, n_steps::Int64; rng=default_rng())
    xi = x0
    for _ = 1:n_steps
        xi = simulate_step(sde, xi, dt, step_scheme; rng=rng)
    end
    return xi
end

function simulate_path(sde::SDE, x0::Float64, dt::Float64, step_scheme::Function, n_steps::Int64; rng=default_rng())
    x = x0*ones(1, n_steps + 1)
    for i = 1:n_steps 
        x[i + 1] = simulate_step(sde, x[i], dt, step_scheme; rng=rng)
    end
    return x
end

function simulate_mc_values(sde::SDE, x0::Float64, dt::Float64, step_scheme::Function, n_steps::Int64, n_paths::Int64, pathwise::Bool; rng=default_rng())
    x = x0*ones(n_paths, 1)
    if pathwise
        x = simulate_value.(sde, x, dt, step_scheme, n_steps; rng=rng)
    else
        for _ = 1:n_steps
            x = simulate_step.(sde, x, dt, step_scheme)
        end
    end
    return x
end

function simulate_mc_paths(sde::SDE, x0::Float64, dt::Float64, step_scheme::Function, n_steps::Int64, n_paths::Int64, pathwise::Bool; rng=default_rng())
    x = x0*ones(n_paths, n_steps + 1)
    if pathwise
        x = simulate_path.(sde, x, dt, step_scheme, n_steps; rng=rng)
    else
        for i = 1:n_steps
            x[i + 1] = simulate_step.(sde, x[i], dt, step_scheme; rng=rng)
        end
    end
    return x
end

function estimate_mc_value(sde::SDE, x0::Float64, dt::Float64, step_scheme::Function, n_steps::Int64, n_paths::Int64, pathwise::Bool; rng=default_rng())
    x = simulate_mc_values(sde, x0, dt, step_scheme, n_steps, n_paths, pathwise; rng=rng)
    return mean(x), std(x)
end 

end # module CompFin
