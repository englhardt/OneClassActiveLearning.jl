struct ParticleSwarmOptimization <: QuerySynthesisOptimizer
    eps::Float64
    swarmsize::Int
    maxiter::Int
    minstep::Float64
    minfunc::Float64
    function ParticleSwarmOptimization(; eps=0.1, swarmsize=100, maxiter=100)
        check_epsilon(eps)
        new(eps, swarmsize, maxiter, 1e-8, 1e-8)
    end
end

function query_synthesis_optimize(f::Function, optimizer::ParticleSwarmOptimization, data::Array{T, 2}, labels::Vector{Symbol})::Array{T, 2} where T <: Real
    lb, ub = vec.(data_boundaries(data[:, labels .!= :Lout]))
    x_opt, _ = pso(x -> vec(-f(x)), lb, ub;
                    swarmsize=optimizer.swarmsize,
                    maxiter=optimizer.maxiter,
                    minstep=optimizer.minstep,
                    minfunc=optimizer.minfunc)
    return reshape(x_opt, size(data, 1), 1)
end

"""
Vectorized version of the partial swarm optimization without constraints.
Original implementation: https://github.com/yuehhua/PSO.jl by Yueh-Hua Tu
"""
function pso(func::Function, lb::Vector, ub::Vector; args=(), kwargs=Dict(),
             swarmsize=100, ω=0.5, ϕp=0.5, ϕg=0.5, maxiter=100, minstep=1e-8, minfunc=1e-8, verbose=false)
    @assert length(ub) == length(lb)
    @assert all(ub .>= lb)

    function update_position!(x, p, fx, fp)
        i_update = (fx .< fp)
        p[:, i_update] .= copy(x[:, i_update])
        fp[i_update] .= fx[i_update]
    end

    obj = x -> func(x, args...; kwargs...)

    # Initialize the particle swarm
    vhigh = abs.(ub .- lb)
    vlow = -vhigh
    S = swarmsize
    D = length(lb)  # the number of dimensions each particle has

    x = lb .+ rand(D, S) .* (ub .- lb)  # particle positions
    v = vlow .+ rand(D, S) .* (vhigh .- vlow)  # particle velocities
    p = zeros(D, S)  # best particle positions

    fx = obj(x)  # current particle function values
    fp = ones(S) * Inf  # best particle function values

    g = copy(x[:, 1])  # best swarm position
    fg = Inf  # best swarm position starting value

    # Store particle's best position (if constraints are satisfied)
    update_position!(x, p, fx, fp)

    # Update swarm's best position
    i_min = argmin(fp)
    if fp[i_min] < fg
        g = copy(p[:, i_min])
        fg = fp[i_min]
    end

    # Iterate until termination criterion met
    it = 1
    while it <= maxiter
        rp = rand(D, S)
        rg = rand(D, S)

        # Update the particles' velocities and positions
        v = ω*v .+ ϕp * rp .*(p .- x) .+ ϕg * rg .*(g .- x)
        x += v
        # Correct for bound violations
        maskl = x .< lb
        masku = x .> ub
        x = x .* (.~(maskl .| masku)) .+ lb .* maskl .+ ub .* masku

        # Update objectives
        fx = obj(x)

        # Store particle's best position (if constraints are satisfied)
        update_position!(x, p, fx, fp)

        # Compare swarm's best position with global best position
        i_min = argmin(fp)
        if fp[i_min] < fg
            verbose && println("New best for swarm at iteration $(it): $(p[i_min, :]) $(fp[i_min])")

            p_min = copy(p[:, i_min])
            stepsize = √(sum((g .- p_min).^2))

            if abs.(fg .- fp[i_min]) <= minfunc
                verbose && println("Stopping search: Swarm best objective change less than $(minfunc)")
                return (g, fg, p, fp)
            end
            if stepsize <= minstep
                verbose && println("Stopping search: Swarm best position change less than $(minstep)")
                return (g, fg, p, fp)
            end

            g = copy(p_min)
            fg = fp[i_min]
        end

        verbose && println("Best after iteration $(it): $(g) $(fg)")
        it += 1
    end

    verbose && println("Stopping search: maximum iterations reached --> $(maxiter)")
    return (g, fg, p, fp)
end
