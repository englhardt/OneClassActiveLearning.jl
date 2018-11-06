
struct KDEException <: Exception
    n::Int
    d::Int
    cov_det::Float64
    unique_values::Int
    KDEException(n, d, cov_det, unique_values) = new(n, d, cov_det, unique_values)
    KDEException(data) = new(size(data, 2), size(data, 1), det(cov(data; dims=2)), size(unique(data; dims=2), 2))
end

function Base.showerror(io::IO, kdee::KDEException)
    print(io, "KDE failed. ")
    if kdee.n < kdee.d
        print(io,"Cannot estimate covariance matrix if number of observations is less than the number of attributes." *
        "See https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices. ")
    end
    if kdee.cov_det ≈ 0
        print(io,"""Covariance matrix is singular. """)
    end
    print(io, "Debug info: the number of attributes is $(kdee.d) and the number of unique observations is $(kdee.unique_values). ")
    return nothing
end

struct MissingLabelTypeException <: Exception
    t::Symbol
end

function Base.showerror(io::IO, e::MissingLabelTypeException)
    print(io, "QS failed. ")
    print(io, "Debug info: labels do not contain '$(e.t)'.")
    return nothing
end

function filter_array(x::Array{T, 2}, remove_indices::Array{Int, 1})::Array{T, 2} where T <: Real
    return x[:, setdiff(1:size(x, 2), remove_indices)]
end

function multi_kde(x::Array{Int, 2}, bw_method="scott")::PyCall.PyObject
    return multi_kde(convert.(Float64, x), bw_method)
end

function multi_kde(x::Array{Float64, 2}, bw_method="scott")::PyCall.PyObject
    size(x, 2) >= size(x, 1) || throw(KDEException(x))
    !(det(cov(x; dims=2)) ≈ 0) || throw(KDEException(x))
    return gaussian_kde(x, bw_method)
end

function leave_out_one_cv_kde(x::Array{T, 2}, bw_method="scott")::Float64 where T <: Real
    mean(multi_kde(x[:, 1:end .!= i], bw_method)(x[:, i]) for i in 1:size(x, 2))[1]
end

function knn_indices(x::Array{T, 2}; k=1::Int)::Array{Int, 2} where T <: Real
    kdtree = KDTree(x)
    return hcat(knn(kdtree, x, k + 1, true)[1]...)[2:end,:]
end

function knn_mean_dist(x::Array{T, 2}; k=1)::Array{Float64,1} where T <: Real
    kdtree = KDTree(x)
    return map(d -> mean(d[2:end]), knn(kdtree, x, k + 1, true)[2])
end

function initialize_qs(qs::DataType, model::OCClassifier, data::Array{T, 2}, params::Dict{Symbol, <:Any})::QueryStrategy where T <: Real
    if qs <: HybridQs
        return qs(model, data; params...)
    elseif qs <: ModelBasedQs
        return qs(model; params...)
    elseif qs <: DataBasedQs
        kernel = get_kernel(model)
        if typeof(kernel) == GaussianKernel
            return qs(data, bw_method=MLKernels.getvalue(strategy.kernel.alpha); params...)
        else
            return qs(data; params...)
        end
    elseif qs <: QueryStrategy
        return qs(; params...)
    end
    throw(ErrorException("Unknown query strategy of type $(qs)."))
end
