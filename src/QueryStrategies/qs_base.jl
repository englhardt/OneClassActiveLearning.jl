
function initialize_qs(qs, model::OCClassifier, data::Array{T, 2}, params)::qs where T <: Real
    if qs <: HybridPQs || qs <: HybridQss
        return qs(model, data; params...)
    elseif qs <: ModelBasedPQs || qs <: ModelBasedQss
        return qs(model; params...)
    elseif qs <: DataBasedPQs || qs <: DataBasedQss
        kernel = get_kernel(model)
        if typeof(kernel) == GaussianKernel
            return qs(data, bw_method=MLKernels.getvalue(strategy.kernel.alpha); params...)
        else
            return qs(data; params...)
        end
    elseif qs <: SubspaceQs
        return qs(model, data; params...)
    elseif qs <: QueryStrategy
        return qs(; params...)
    end
    throw(ErrorException("Unknown query strategy of type $(qs)."))
end
