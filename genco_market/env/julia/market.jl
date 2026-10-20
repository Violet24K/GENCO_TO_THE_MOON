# Market clearing on top of datakit's in-memory PowerModels data path.
# Loaded once per process after gridfm_datakit's init_julia() (which brings
# PowerModels, Ipopt and the _GFM_WORK buffer into Main).

using PowerModels
using Ipopt

# _gfm_state does not push ratings or pmax, so we set them on every call
# (the work dict is reused across solves, stale values would leak otherwise).
function gc_set_limits!(data, rate_a_pu, pmax_pu)
    for i in eachindex(rate_a_pu)
        br = data["branch"][string(i)]
        br["rate_a"] = Float64(rate_a_pu[i])
        br["rate_b"] = Float64(rate_a_pu[i])
        br["rate_c"] = Float64(rate_a_pu[i])
    end
    for i in eachindex(pmax_pu)
        data["gen"][string(i)]["pmax"] = Float64(pmax_pu[i])
    end
    return data
end

function _gc_optimizer(attempt::Int, max_iter::Int, max_wall::Float64)
    if attempt == 1
        return optimizer_with_attributes(Ipopt.Optimizer,
            "tol" => 1e-6, "print_level" => 0, "max_iter" => max_iter,
            "max_wall_time" => max_wall)
    end
    # retry: different barrier strategy, more iterations
    return optimizer_with_attributes(Ipopt.Optimizer,
        "tol" => 1e-6, "print_level" => 0, "max_iter" => 2 * max_iter,
        "max_wall_time" => max_wall,
        "mu_strategy" => "adaptive", "mu_init" => 1e-2,
        "bound_push" => 1e-4, "bound_frac" => 1e-4)
end

function _gc_set_starts!(data)
    for (_, bus) in data["bus"]
        bus["vm_start"] = bus["vm"]
        bus["va_start"] = bus["va"]
    end
    for (_, gen) in data["gen"]
        gen["pg_start"] = gen["pg"]
        gen["qg_start"] = gen["qg"]
    end
end

function _gc_clear_starts!(data)
    for (_, bus) in data["bus"]
        delete!(bus, "vm_start"); delete!(bus, "va_start")
    end
    for (_, gen) in data["gen"]
        delete!(gen, "pg_start"); delete!(gen, "qg_start")
    end
end

# Load shedding as virtual generators at VOLL (one per loaded bus). They are
# appended to the reused work dict and removed again after the solve.
function _gc_add_shed!(data, voll::Float64, n_gen::Int)
    mva = data["baseMVA"]
    k = n_gen
    shed_bus = Dict{Int,Int}()
    for (_, l) in data["load"]
        (l["status"] == 1 && l["pd"] > 0) || continue
        k += 1
        shed_bus[k] = l["load_bus"]
        qd = l["qd"]
        data["gen"][string(k)] = Dict{String,Any}(
            "index" => k, "gen_bus" => l["load_bus"], "gen_status" => 1,
            "pg" => 0.0, "qg" => 0.0, "pmin" => 0.0, "pmax" => l["pd"],
            "qmin" => min(0.0, qd), "qmax" => max(0.0, qd), "vg" => 1.0,
            "mbase" => mva, "model" => 2, "ncost" => 2, "cost" => [voll * mva, 0.0],
            "startup" => 0.0, "shutdown" => 0.0, "source_id" => Any["shed", k])
    end
    return k, shed_bus
end

function _gc_remove_shed!(data, n_gen::Int, k::Int)
    for i in (n_gen + 1):k
        delete!(data["gen"], string(i))
    end
end

"""
    gc_clear(data, kind, attempt, max_iter, max_wall, voll, bus_ids, n_gen, n_branch)

kind = "ac" (ACP polar) or "dc" (DCP), both solved with Ipopt.
attempt 1 = flat start; attempt 2 = warm start from the case voltages with
different Ipopt options. voll > 0 adds load shedding at that price (\$/MWh).
Returns (status, objective, lam[bus], pg[gen], pf[br], qf[br], pt[br],
qt[br], shed[bus]) in raw PowerModels units (per unit; lam is
the KCL dual, \$/h per p.u.). Missing components are NaN.
"""
function gc_clear(data, kind::String, attempt::Int, max_iter::Int, max_wall::Float64, voll::Float64, bus_ids, n_gen::Int, n_branch::Int)
    model = kind == "ac" ? ACPPowerModel : DCPPowerModel
    setting = Dict("output" => Dict("duals" => true))
    warm = kind == "ac" && attempt > 1
    warm && _gc_set_starts!(data)
    k, shed_bus = voll > 0 ? _gc_add_shed!(data, voll, n_gen) : (n_gen, Dict{Int,Int}())
    result = nothing
    try
        result = solve_opf(data, model, _gc_optimizer(attempt, max_iter, max_wall); setting = setting)
    finally
        warm && _gc_clear_starts!(data)
        _gc_remove_shed!(data, n_gen, k)
    end
    status = string(result["termination_status"])
    obj = Float64(get(result, "objective", NaN))
    nb = length(bus_ids)
    lam = fill(NaN, nb); pg = fill(NaN, n_gen)
    pf = fill(NaN, n_branch); qf = fill(NaN, n_branch)
    pt = fill(NaN, n_branch); qt = fill(NaN, n_branch)
    shed = zeros(nb)
    row = Dict(Int(b) => r for (r, b) in enumerate(bus_ids))
    sol = get(result, "solution", Dict())
    if haskey(sol, "bus")
        sb = sol["bus"]
        for (r, i) in enumerate(bus_ids)
            k = string(Int(i))
            haskey(sb, k) && (lam[r] = get(sb[k], "lam_kcl_r", NaN))
        end
    end
    if haskey(sol, "gen")
        for (key, g) in sol["gen"]
            i = parse(Int, key)
            if i <= n_gen
                pg[i] = get(g, "pg", NaN)
            else
                shed[row[shed_bus[i]]] += get(g, "pg", 0.0)
            end
        end
    end
    if haskey(sol, "branch")
        for (k, b) in sol["branch"]
            i = parse(Int, k)
            pf[i] = get(b, "pf", NaN); qf[i] = get(b, "qf", NaN)
            pt[i] = get(b, "pt", NaN); qt[i] = get(b, "qt", NaN)
        end
    end
    return (status, obj, lam, pg, pf, qf, pt, qt, shed)
end
