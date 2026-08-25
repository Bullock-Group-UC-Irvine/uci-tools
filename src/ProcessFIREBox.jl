module ProcessFIREBox

import PyCall
import Plots
import Printf
import HDF5
import DataFrames
import ProgressBars
import CSV
import IndexedDataFrames
import Statistics
import ..FIREBoxIO
import ..UCIToolsConfig

ENV["GKSwstype"] = "100"

conf = UCIToolsConfig.read_config()
firebox_dir = conf["uci_tools_paths"]["firebox_data_dir"]
firebox_snap = conf["uci_tools_paths"]["firebox_snap"]
output_dir = conf["uci_tools_paths"]["project_data_dir"]

function get_sfrs(
            ids,
            grp_ids;
            make_plots=true,
            verbose=false,
            only_bound=false
        )
    df = DataFrames.DataFrame(
        id=ids,
        grp_id=grp_ids,
        sfr=Any[fill(nothing, length(ids))...],
        sfr_unfiltered=Any[fill(missing, length(ids))...],
        ssfr=Any[fill(nothing, length(ids))...],
        Mstar=Any[fill(nothing, length(ids))...],
        bound_frac=Float64[fill(1., length(ids))...]
    )
    idf = IndexedDataFrames.IndexedDataFrame(df, "id")

    sfrs_gals = Float64[]
    missing_files = Int64[]
    zero_bound = Int64[]

    for (gal_id, grp_id) in ProgressBars.ProgressBar(zip(ids, grp_ids))
        id_str = string(gal_id)
        if verbose
            println(
                "\nObject " * id_str * " has group ID " * string(grp_id) * "."
            )
        end
        fname = joinpath(
            firebox_dir,
            firebox_snap,
            "particles_within_Rvir_object_" * id_str * ".hdf5"
        )
        if isfile(fname)
            sfrs, gas_masses, Mstar, snap_time, gas_ids = HDF5.h5open(
                        fname, 
                        "r"
                    ) do file
                sfrs = read(file, "gas_sfr")
                gas_masses = read(file, "gas_mass")
                gas_ids = Int.(read(file, "gas_id"))
                Mstar = read(file, "Mstar")
                snap_time = read(file, "time")
                return sfrs, gas_masses, Mstar, snap_time, gas_ids
            end
            if Mstar == 0.
                deleteat!(idf, gal_id)
                continue # Skip this galaxy.
            end
            if Int(grp_id) != -1 && only_bound
                # If the galaxy is not a host, filter for only bound particles.
                bound_ids = FIREBoxIO.get_bound_particles(gal_id)
                # Narrow down the bound gas IDs so the `in.` below is faster.
                bound_gas_ids = intersect(gas_ids, bound_ids)
                is_bound = in.(gas_ids, Ref(Set(bound_gas_ids)))
                bound_frac = Statistics.mean(is_bound)
                if verbose
                    Printf.@printf(
                        "%.0f%% of gas particles in the object are bound.",
                        bound_frac * 100.
                    )
                    println("\nFirst 100 bound particle IDs:")
                    println(bound_ids[1:100])
                end
                idf[gal_id, "bound_frac"] = bound_frac
                if sum(is_bound) == 0 
                    # If there's no overlap between the particle IDs in the
                    # bound particles file and those in the Rvir file, there's
                    # probably a problem. We should drop those galaxies for
                    # now.
                    deleteat!(idf, gal_id)
                    push!(zero_bound, gal_id)
                    continue # Skip to the next gal.
                end
                idf[gal_id, "sfr_unfiltered"] = sum(sfrs)
                sfrs = sfrs[is_bound]
            end
            sfr = sum(sfrs)
            #Printf.@printf("SFR: %.2f Msun / yr", sfr)
            idf[gal_id] = (sfr=sfr, ssfr=sfr/Mstar, Mstar=Mstar)
            push!(sfrs_gals, sfr) 
        else
            if verbose
                println("Could not find file " * fname)
            end
            push!(missing_files, gal_id)
            # Drop the galaxy
            deleteat!(idf, gal_id)
        end
    end

    if make_plots
        Plots.histogram(
            idf.ssfr,
            #yscale=:log10,
            ylabel="N_gal",
            xlabel="SFR / M_stellar [yr^-1])",
            legend=false
        )
        Plots.savefig("hist.png") 

        Plots.scatter(
            log10.(idf.Mstar),
            idf.ssfr,
            ylabel="SFR / M_stellar [yr^-1])",
            xlabel="log(Mstar / Msun)",
            legend=false
        )
        Plots.savefig("scatter.png")
    end

    println("\n$(length(missing_files)) missing satellite files:") 
    println(missing_files)
    if only_bound
        println(
            "\n$(length(zero_bound)) satellites with no overlap with" *
            " bound_particle file:"
        )
        println(zero_bound)
    end
    
    return idf
end

function get_all_sfrs(;save=false)
    gal_ids, grp_ids = FIREBoxIO.get_both()
    sfr_df = get_sfrs(gal_ids, grp_ids, make_plots=false, only_bound=false)
    if save
        CSV.write(
            joinpath(output_dir, "inst_sfrs_no_bound_filter.csv"),
            sfr_df.df
        )
    end
    return sfr_df
end

function compare_sats_b4_filtering()
    gal_ids, grp_ids = FIREBoxIO.get_sats()
    sfr_df = get_sfrs(
        gal_ids,
        grp_ids,
        make_plots=false,
    )
    return sfr_df
end

function get_avg_sfrs(ids, grp_ids, age; only_bound=false)
    cosmo = PyCall.pyimport("astropy.cosmology")
    astropy = PyCall.pyimport("astropy")

    z_age = cosmo.z_at_value(
        cosmo.Planck13.lookback_time, 
        age * astropy.units.Gyr
    )[1]
    a_age = 1. / (z_age + 1.)

    df = DataFrames.DataFrame(
        id=ids,
        grp_id=grp_ids,
        sfr=Any[fill(nothing, length(ids))...],
        ssfr=Any[fill(nothing, length(ids))...],
        Mstar=Any[fill(nothing, length(ids))...],
    )
    if only_bound
        df[:, "ssfr_unfiltered"] = Any[fill(missing, length(ids))...]
        bound_star_frac=Float64[fill(1., length(ids))...]
    end
    idf = IndexedDataFrames.IndexedDataFrame(df, "id")

    for (gal_id, grp_id) in ProgressBars.ProgressBar(zip(ids, grp_ids))
        id_str = string(gal_id)
        path = joinpath(
            firebox_dir,
            firebox_snap,
            "particles_within_Rvir_object_" * id_str * ".hdf5"
        )
        if isfile(path)
            file_contents = HDF5.h5open(path, "r") do file
                if !("stellar_tform" in keys(file))
                    # If there's no scale factor of stellar formation
                    # information for this galaxy, skip it and delete it from
                    # the DataFrame.
                    return -1
                end
                stellar_scale_facs = read(file, "stellar_tform")
                Mstar = read(file, "Mstar")
                # Stellar masses in units of M_sun
                stellar_masses = read(file, "stellar_mass") * 1.e10
                stellar_ids = read(file, "stellar_id")
                return stellar_scale_facs, Mstar, stellar_masses, stellar_ids
            end
            if file_contents == -1
                # If there's no scale factor of stellar formation
                # information for this galaxy, skip it and delete it from
                # the DataFrame.
                deleteat!(idf, gal_id)
                continue
                
            end
            (
                stellar_scale_facs,
                Mstar,
                stellar_masses,
                stellar_ids 
            ) = file_contents

            is_younger_age = stellar_scale_facs .>= a_age
            masses_younger_age = stellar_masses[is_younger_age]
            # SFR in M_sun / yr:

            if Int(grp_id) != -1 && only_bound
                # If the galaxy is not a host, filter for only bound particles.
                bound_ids = FIREBoxIO.get_bound_particles(gal_id)
                # Narrow down the bound IDs so the `in.` below is faster.
                bound_stellar_ids = intersect(stellar_ids, bound_ids)
                is_bound = in.(stellar_ids, Ref(Set(bound_stellar_ids)))
                bound_frac = Statistics.mean(is_bound)
                idf[gal_id, "bound_star_frac"] = bound_frac
                if sum(is_bound) == 0 
                    # If there's no overlap between the particle IDs in the
                    # bound particles file and those in the Rvir file, there's
                    # probably a problem. We should drop those galaxies for
                    # now.
                    deleteat!(idf, gal_id)
                    continue # Skip to the next gal.
                end
                idf[gal_id, "ssfr_unfiltered"] = (
                    sum(masses_younger_age) / age / 1.e9 / Mstar
                )
                masses_younger_age = stellar_masses[
                    is_bound .& is_younger_age
                ]
            end

            # SFR in units of M_solar / yr
            sfr = sum(masses_younger_age) / age / 1.e9
            Mstar_fr_particles = sum(stellar_masses)

            idf[gal_id, "Mstar"] = Mstar
            idf[gal_id, "sfr"] = sfr
            idf[gal_id, "ssfr"] = sfr / Mstar
        end
    end

    return idf
end

function get_all_avg_sfrs(age; save=false, debug_mode=false, only_bound=false)
    gal_ids, grp_ids = FIREBoxIO.get_both()
    if debug_mode
        gal_ids = gal_ids[1:5]
        grp_ids = grp_ids[1:5]
    end
    sfr_df = get_avg_sfrs(gal_ids, grp_ids, age, only_bound=only_bound)
    if save
        CSV.write(
            joinpath(
                output_dir,
                "avg_sfrs_$(age)Gyr_no_bound_filter.csv"
            ),
            sfr_df.df
        )
    end
    return sfr_df
end

function get_gas_mass_by_temp(id)
    uci = PyCall.pyimport("uci_tools")

    fname = "particles_within_Rvir_object_" * string(id) * ".hdf5"
    path = joinpath(firebox_dir, firebox_snap, fname)
    (
        gas_masses,
        he_fracs,
        e_abundances,
        energies
    ) = HDF5.h5open(path, "r") do file
        gas_masses = read(file, "gas_mass") # Units of 1e10 Msun
        he_fracs = read(file, "gas_metal_01")
        e_abundances = read(file, "gas_ne")
        energies = read(file, "gas_u")
        return gas_masses, he_fracs, e_abundances, energies
    end
    gas_temps = uci.tools.calc_temps(
        he_fracs,
        e_abundances,
        energies
    )
    temps = [1.e1, 1.e2, 1.e3, 1.e4, 1.e5, 1.e6]
    mass_d = Dict()
    for temp in temps
        is_below_temp = gas_temps .<= temp
        mass_d[temp] = sum(gas_masses[is_below_temp]) * 1.e10
    end
    return mass_d
end

function get_all_gas_mass_by_temp(; save=false, debug_mode=false)
    ids, grp_ids = FIREBoxIO.get_both(only_files=true)
    if debug_mode
        ids = ids[1:5]
        grp_ids = grp_ids[1:5]
    end
    df = DataFrames.DataFrame()
    for (id, grp_id) in ProgressBars.ProgressBar(zip(ids, grp_ids))
        mass_d = get_gas_mass_by_temp(id)
        for temp in sort(collect(keys(mass_d)))
            push!(df, (id=id, max_temp=temp, gass_mass=mass_d[temp]))
        end
    end
    CSV.write(joinpath(output_dir, "firebox_gas_mass_by_temp.csv"), df)
    return df
end

end #module ProcessFIREBox
