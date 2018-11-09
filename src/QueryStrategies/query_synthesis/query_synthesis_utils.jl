
function estimate_margin_epsilon(model::OCClassifier, data_outliers::Array{T, 2})::Float64 where T <: Real
    return maximum(SVDD.predict(model, data_outliers))
end

function estimate_limit_epsilon(x::Array{T, 2}, p=0.1) where T <: Real
    return vec((maximum(x, dims=2) .- minimum(x, dims=2)) * p)
end

function check_epsilon(eps)
    if !all(eps .>= 0)
        throw(ArgumentError("Invalid epsilon $eps."))
    end
    return nothing
end

function check_limits(limits)
    if size(limits, 2) != 2 || length(limits[:, 1]) != length(limits[:, 2]) ||
        !all(limits[:, 1] .< limits[:, 2])
        throw(ArgumentError("Invalid limits $limits."))
    end
    return nothing
end
