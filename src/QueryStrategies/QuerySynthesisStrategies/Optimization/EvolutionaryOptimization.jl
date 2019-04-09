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
