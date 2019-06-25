
import Base: ==

function ==(a::ValueHistories.UnivalueHistory, b::ValueHistories.UnivalueHistory)
    return (a.values == b.values) && (a.lastiter == b.lastiter) && (a.iterations == b.iterations)
end

function ValueHistories.History(DT::Type{V}, lastiter::I, iterations::Vector{I}, values::Vector{V}) where {I,V}
    h = ValueHistories.History(DT)
    h.lastiter = lastiter
    h.iterations = iterations
    h.values = values
    return h
end

function ValueHistories.MVHistory(x::Dict{Symbol, ValueHistories.UnivalueHistory})
    mvh = ValueHistories.MVHistory()
    merge!(mvh.storage, x)
    return mvh
end

function unsafe_convert(::Type{Vector{Vector{Symbol}}}, a::Array{Any,1})
    [Symbol.(x) for x in a]
end

function unsafe_convert(::Type{Vector{Array{T, 2}}}, a::Array{Any,1}) where T <: Real
    [hcat(x...) for x in a]
end

function Unmarshal.unmarshal(DT::Type{ValueHistories.MVHistory}, parsedJson::AbstractDict, verbose::Bool = false, verboseLvl::Int = 0)
    mvh = Dict{Symbol, ValueHistories.UnivalueHistory}()
    for (k,v) in parsedJson["mvhistory"]
        # Meta.parse may not be the best solution as it introduces
        # many potential security risks, but it works
        # string is converted to expression which is then evaluated
        # now also generic types like Array{T,N} can be parsed correctly
        itype = eval(Meta.parse(parsedJson["mv_itypes"][k]))
        vtype = eval(Meta.parse(parsedJson["mv_vtypes"][k]))
        if vtype in [Vector{Symbol}, Array{Int, 2}, Array{Float64, 2}]
            values = unsafe_convert(Vector{vtype}, v["values"])
        else
            values = convert(Array{vtype,1}, v["values"])
        end
        push!(mvh, Symbol(k) => ValueHistories.History(vtype, v["lastiter"], convert(Vector{itype}, v["iterations"]), values))
    end
    return ValueHistories.MVHistory(mvh)
end

function JSON.json(x::ValueHistories.MVHistory)
    itypes = Dict(k => typeof(x[k]).parameters[1] for (k,v) in x.storage)
    vtypes = Dict(k => typeof(x[k]).parameters[2] for (k,v) in x.storage)
    return JSON.json(Dict(:mvhistory => x.storage, :mv_itypes => itypes, :mv_vtypes => vtypes))
end

function JSON.lower(x::T) where T <: SplitStrategy
    return JSON.lower(typeof(x))
end

function JSON.lower(x::T) where T <: MLKernels.Kernel
    return JSON.lower(sprint(print, x))
end

function JSON.lower(x::T) where T <: SVDD.InitializationStrategy
    return JSON.lower(sprint(print, x))
end

function JSON.lower(x::T) where T <: SubspaceQueryStrategy
    return JSON.lower(sprint(print, x))
end

function JSON.lower(x::T) where T <: Function
    JSON.lower(sprint(print, x))
end

function JSON.lower(x::T) where T <: Oracle
    return JSON.lower(typeof(x))
end

function JSON.json(res::Result)
    jsonstring = "{"
    jsonstring *= "\"id\":" * JSON.json(res.id) * ","
    jsonstring *= "\"experiment\":" * JSON.json(res.experiment) * ","
    jsonstring *= "\"worker_info\":" * JSON.json(res.worker_info) * ","
    jsonstring *= "\"data_stats\":" * JSON.json(res.data_stats) * ","
    jsonstring *= "\"al_history\":" * JSON.json(res.al_history) * ","
    jsonstring *= "\"al_summary\":" * JSON.json(res.al_summary) * ","
    jsonstring *= "\"status\":" * JSON.json(res.status)
    jsonstring *= "}"
    return jsonstring
end

function write_result_to_file(output_file, r::Result)
    r_reparsed = JSON.parse(JSON.json(r))
    d = DataStructures.OrderedDict()
    for k in ["id", "experiment", "worker_info", "data_stats", "al_history", "al_summary", "status"]
        d[k] = r_reparsed[k]
    end
    open(output_file, "w") do f
        JSON.print(f, d, 2)
    end
    return nothing
end

function unmarshal_al_summary(parsedJson::AbstractDict)
    al_summary = Dict()
    for (k_metric, v_metric) in parsedJson
        d = Dict{Symbol, Any}()
        for (k_summary, v_summary) in v_metric
            cur_res = nothing
            try
                if typeof(v_summary) <: Vector
                    cur_res = Unmarshal.unmarshal(Vector{Float64}, v_summary)
                else
                    cur_res = Unmarshal.unmarshal(Float64, v_summary)
                end
            catch ArgumentError e
                cur_res = typeof(v_summary) <: Vector ? zeros(length(v_summary)) : 0.0
            end
            d[Symbol(k_summary)] = cur_res
        end
        push!(al_summary, Symbol(k_metric) => d)
    end
    return al_summary
end

function convert_to_symbol_keys(x)
    helper(x) = x
    helper(d::Dict) = Dict(Symbol(k) => helper(v) for (k, v) in d)
    return helper(x)
end


function Unmarshal.unmarshal(DT::Type{Result}, parsedJson::AbstractDict)
    id = parsedJson["id"]
    al_history = Unmarshal.unmarshal(ValueHistories.MVHistory, parsedJson["al_history"])
    experiment = convert_to_symbol_keys(parsedJson["experiment"])
    worker_info = convert_to_symbol_keys(parsedJson["worker_info"])
    data_stats = Unmarshal.unmarshal(DataStats, parsedJson["data_stats"])
    al_summary = unmarshal_al_summary(parsedJson["al_summary"])
    status = Dict(:exit_code => Symbol(parsedJson["status"]["exit_code"]))
    return Result(id, experiment, worker_info, data_stats, al_history, al_summary, status)
end
