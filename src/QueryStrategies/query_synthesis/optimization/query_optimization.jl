using BlackBoxOptim
using Evolutionary

include("particle_swarm_optimization.jl")

abstract type QuerySynthesisOptimizer end

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

struct BlackBoxOptimization <: QuerySynthesisOptimizer
    method::Symbol
    eps::Float64
    params::Dict{Symbol, Any}
    function BlackBoxOptimization(method::Symbol; eps::Float64=0.1, params...)
        new(method, eps, Dict(params))
    end
end

function query_synthesis_optimize(f::Function, optimizer::BlackBoxOptimization, data::Array{T, 2}, labels::Vector{Symbol})::Array{T, 2} where T <: Real
    lb, ub = vec.(data_boundaries(data[:, labels .!= :Lout]))
    res = bboptimize(x -> -first(f(reshape(x, length(x), 1)));
        NumDimensions=size(data, 1), Method=optimizer.method, lowerBound=lb, upperBound=ub,
        TraceMode=:silent, optimizer.params...)
    return reshape(best_candidate(res), size(data, 1), 1)
end

struct EvolutionaryOptimization <: QuerySynthesisOptimizer
    method::Symbol
    params::Dict{Symbol, Any}
    function EvolutionaryOptimization(method::Symbol; params...)
        new(method, Dict(params))
    end
end

function query_synthesis_optimize(f::Function, optimizer::EvolutionaryOptimization, data::Array{T, 2}, labels::Vector{Symbol})::Array{T, 2} where T <: Real
    (x_opt, fitness, iterations) = eval(optimizer.method)(x -> -first(f(reshape(x, length(x), 1))), size(data, 1); optimizer.params...)
    return reshape(x_opt, size(data, 1), 1)
end
