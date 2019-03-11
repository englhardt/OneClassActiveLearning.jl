
function estimate_boundary_shift_epsilon(model::OCClassifier, data_outliers::Array{T, 2}; agg_func=:maximum)::Float64 where T <: Real
    return eval(agg_func)(SVDD.predict(model, data_outliers))
end

function check_epsilon(eps::T) where T <: Real
    if eps < 0
        throw(ArgumentError("Invalid epsilon $eps."))
    end
    return nothing
end

function check_epsilon(eps::Vector{T}) where T <: Real
    check_epsilon.(eps)
    return nothing
end

function check_data_limits(limits)
    if size(limits, 2) != 2 || length(limits[:, 1]) != length(limits[:, 2]) ||
        !all(limits[:, 1] .< limits[:, 2])
        throw(ArgumentError("Invalid limits $limits."))
    end
    return nothing
end

function extrema_arrays(x::Array{T, 2})::Tuple{Array{Float64, 2}, Array{Float64, 2}} where T <: Real
    ex = extrema(x, dims=2)
    x_minima = [x[1] for x in ex]
    x_maxima = [x[2] for x in ex]
    return x_minima, x_maxima
end

function data_boundaries(x::Array{T, 2}, p=0.0)::Tuple{Array{Float64, 2}, Array{Float64, 2}} where T <: Real
    x_minima, x_maxima = extrema_arrays(x)
    padding =  (x_maxima - x_minima) * p
    x_minima = x_minima - padding
    x_maxima = x_maxima + padding
    return x_minima, x_maxima
end

function rand_in_hyper_rect(x_minima, x_maxima, epsilon::T=0.0) where T <: Real
    return rand_in_hyper_rect(x_minima, x_maxima, fill(epsilon, length(x_minima)))
end

function rand_in_hyper_rect(x_minima, x_maxima, epsilon::Vector{T}) where T <: Real
    return rand(length(x_minima), 1) .* (x_maxima .- x_minima) .* (1 .+ epsilon) .+ x_minima
end
