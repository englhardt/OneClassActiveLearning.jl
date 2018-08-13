"""
Original publication:
A. Ghasemi, H. R. Rabiee, M. Fadaee, M. T. Manzuri,
and M. H. Rohban. Active learning from positive and
unlabeled data. In 2011 IEEE 11th International
Conference on Data Mining Workshops, pages
244–250, Dec 2011.
"""
struct MinimumMarginQs <: DataBasedQs
    p_x::Array{Float64,1}
    bw_method::Union{String, U} where U <: Real
    p_inlier::Float64
    function MinimumMarginQs(x::Array{T, 2}; bw_method="scott", p_inlier=nothing) where T <: Real
        ((p_inlier === nothing) || !(0 <= p_inlier <= 1)) && throw(ArgumentError("Invalid inlier probability $(p_inlier)."))
        return new(multi_kde(x, bw_method)(x), bw_method, p_inlier)
    end
end

function qs_score(qs::MinimumMarginQs, x::Array{T, 2}, labels::Dict{Symbol, Array{Int, 1}})::Array{Float64, 1} where T <: Real
    haskey(labels, :Lin) || throw(MissingLabelTypeException(:Lin))
    p_x_inlier = multi_kde(x[:, labels[:Lin]], qs.bw_method)(x)
    return -abs.((2 * p_x_inlier * qs.p_inlier - qs.p_x) ./ qs.p_x)
end
