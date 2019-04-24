"""
    SubspaceQs(single_space_strategy::QueryStrategy, combination_fct::Function, scale_fct::Function)

    Query Strategy for informativenss across multiple subspaces.

# Arguments
    - `single_space_strategy`: QueryStrategy that returns informativeness in a subspace
    - `scale_fct`: monotonous scaling function to scale scores within subspace
    - `combination_fct`: reduces subspaces scores to combine scores across subspaces

# Examples
    ```julia-repl
    julia> SubspaceQs(RandomPQs(), [[1,2], [3,4]], +, identity)
    ```
"""
struct SubspaceQs{Q <: QueryStrategy} <: SubspaceQueryStrategy
    single_space_strategy::Q
    subspaces::Vector{Vector{Int}}
    combination_fct::Function
    scale_fct::Function
end

"""
    Consumes params for SubspaceQs and passes the remaining
    params through to initialize the single_space_strategy.
"""
function SubspaceQs{Q}(model,
                    data::Array{T, 2} where T <: Real;
                    subspaces,
                    combination_fct = +,
                    scale_fct = identity,
                    params...) where Q <: QueryStrategy
    params = Dict(params)
    if length(params) == 0
        params = Dict{Symbol, Any}()
    end
    qs = initialize_qs(Q, model, data, params)
    SubspaceQs(qs, subspaces, combination_fct, scale_fct)
end

function qs_score(qs::SubspaceQs,
                  x::Array{T,2},
                  labels::Dict{Symbol, Array{Int, 1}}) where {T <: Real}
    size(x,1) >= maximum(maximum.(qs.subspaces)) || throw(DimensionMismatch("Maximum subspace dimension is
        larger than numver of dimensions in query data x."))
    subspace_scores = qs_score(qs.single_space_strategy, x, labels, qs.subspaces)
    try
        return mapreduce(qs.scale_fct, qs.combination_fct, subspace_scores)
    catch err
        error(getlogger(@__MODULE__), "Failed to reduce subspaces_scores with scale_fct = $(qs.scale_fct) and combination_fct = $(qs.combination_fct).
            Make sure that you use a function that is applicable to reduce (e.g., '+' and not 'sum'). \n $(sprint(io -> showerror(io, err)))")
    end
end

function qs_score(qs::QueryStrategy,
          x::Array{T, 2},
          labels::Dict{Symbol, Array{Int, 1}},
          subspaces::Vector{Vector{Int}}) where {T <: Real}
    error("Query strategy $(typeof(qs)) is not implemented for subspace classifiers.")
end
