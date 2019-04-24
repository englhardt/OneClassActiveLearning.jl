
# Taken from julia/base/util.jl
const _mem_units = ["byte", "KiB", "MiB", "GiB", "TiB", "PiB"]
const _cnt_units = ["", " k", " M", " G", " T", " P"]
function prettyprint_getunits(value, numunits, factor)
    if value == 0 || value == 1
        return (value, 1)
    end
    unit = ceil(Int, log(value) / log(factor))
    unit = min(numunits, unit)
    number = value/factor^(unit-1)
    return number, unit
end

function format_bytes(bytes)
    bytes, mb = prettyprint_getunits(bytes, length(_mem_units), Int64(1024))
    if mb == 1
        @sprintf("%d %s%s", bytes, _mem_units[mb], bytes==1 ? "" : "s")
    else
        @sprintf("%.3f %s", bytes, _mem_units[mb])
    end
end

format_number(n::Int) = Formatting.format(n, commas=true)

format_observations(x::Array{T, 2}) where T <: Real = format_number(size(x, 2))

function log_experiment_info(experiment)
    split = experiment[:split_strategy]
    debug(LOGGER, "Experiment info: \n\tdata file: $(experiment[:data_file]),\n" *
        "\tparams: adjust_K = $(experiment[:param][:adjust_K]), num_al_iterations = $(experiment[:param][:num_al_iterations])\n" *
        "\tquery_strategy: $(experiment[:query_strategy][:type])\n" *
        "\tmodel: $(experiment[:model][:type])\n" *
        "\tinit_strategy: $(experiment[:model][:init_strategy])\n" *
        "\tinitial_pools: $(countmap(experiment[:param][:initial_pools]))\n" *
        "\tsplit_strategy: \n\t\ttrain: $(split.train_strat); \n\t\ttest:  $(split.test_strat); \n\t\ttrain: $(split.query_strat)")
end

function Base.push!(logger::Logger, handler::Handler, name::String)
    logger.handlers[name] = handler
end

get_val_type(x::Val{T}) where T = T
