struct BlackBoxOptimization <: QuerySynthesisOptimizer
    method::Symbol
    eps::Float64
    params::Dict{Symbol, Any}
    function BlackBoxOptimization(method::Symbol; eps::Float64=0.1, params...)
        new(method, eps, Dict(params))
    end
end

function query_synthesis_optimize(f::Function, optimizer::BlackBoxOptimization, data::Array{T, 2}, labels::Vector{Symbol})::Array{T, 2} where T <: Real
    lb, ub = vec.(data_boundaries(data[:, labels .!= :Lout], optimizer.eps))
    res = BlackBoxOptim.bboptimize(x -> -first(f(reshape(x, length(x), 1)));
        NumDimensions=size(data, 1), Method=optimizer.method,
        SearchRange=[(lb[i], ub[i]) for i in eachindex(lb)],
        TraceMode=:silent, optimizer.params...)
    return reshape(BlackBoxOptim.best_candidate(res), size(data, 1), 1)
end
