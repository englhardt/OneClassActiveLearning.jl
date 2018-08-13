"""
Original publication:
A. Ghasemi, M. T. Manzuri, H. R. Rabiee, M. H.
Rohban, and S. Haghiri. Active one-class learning by
kernel density estimation. In 2011 IEEE International
Workshop on Machine Learning for Signal Processing,
pages 1–6, Sept 2011.
"""
struct MinimumLossQs <: DataBasedQs
    p_x::Array{Float64, 1}
    bw_method::Union{String, U} where U <: Real
    p_inlier::Float64
    function MinimumLossQs(x::Array{T, 2}; bw_method="scott", p_inlier=nothing) where T <: Real
        ((p_inlier === nothing) || !(0 <= p_inlier <= 1)) && throw(ArgumentError("Invalid inlier probability $(p_inlier)."))
        return new(multi_kde(x, bw_method)(x), bw_method, p_inlier)
    end
end

function qs_score(qs::MinimumLossQs, x::Array{T, 2}, labels::Dict{Symbol, Array{Int, 1}})::Array{Float64, 1} where T <: Real
    haskey(labels, :Lin) || throw(MissingLabelTypeException(:Lin))
    s_t_a(u) = leave_out_one_cv_kde(hcat(x[:, labels[:Lin]], u), qs.bw_method)
    s_t_b(u) = haskey(labels, :Lout) ? mean(multi_kde(hcat(x[:, labels[:Lin]], u), qs.bw_method)(x[:, labels[:Lout]])) : 0
    s_o_a = leave_out_one_cv_kde(x[:, labels[:Lin]], qs.bw_method)
    s_o_b(u) = haskey(labels, :Lout) ? mean(multi_kde(x[:, labels[:Lin]], qs.bw_method)(hcat(x[:, labels[:Lout]], u))) : 0
    s_t(u) = s_t_a(u) - s_t_b(u)
    s_o(u) = s_o_a - s_o_b(u)
    p_inlier = length(labels[:Lin]) / size(x, 2)
    s(u) = qs.p_inlier * s_t(u) + (1 - qs.p_inlier) * s_o(u)
    return [s(x[:, i]) for i in 1:size(x, 2)]
end
