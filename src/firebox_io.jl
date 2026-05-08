module firebox_io 

using HDF5
import PyCall
import Plots
import Printf
import ProgressBars
import DataFrames
import CSV
import Statistics
import IndexedDataFrames
import ..UCIToolsConfig

ENV["GKSwstype"] = "100"

super_direc = ""
firebox_snap = ""
output_dir = ""

function __init__()
    conf = UCIToolsConfig.read_config()
    global super_direc = conf["uci_tools_paths"]["firebox_data_dir"]
    global firebox_snap = conf["uci_tools_paths"]["firebox_snap"]
    global output_dir  = conf["uci_tools_paths"]["project_data_dir"]
end

function get_grp_id(gal_id)
    fname = super_direc * 
       "global_sample_data/global_sample_data_snapshot_1200.hdf5"
    grp_ids, gal_ids = HDF5.h5open(fname) do file
        grp_ids = read(file, "groupID")
        gal_ids = read(file, "galaxyID")
        return grp_ids, gal_ids
    end
    grp_id = grp_ids[gal_id .== gal_ids]
    @assert length(grp_id) == 1
    return grp_id[1]
end

function get_sats()
    fname = super_direc * 
       "global_sample_data/global_sample_data_snapshot_1200.hdf5"
    grp_ids, gal_ids = HDF5.h5open(fname) do file
        grp_ids = Int.(read(file, "groupID"))
        gal_ids = Int.(read(file, "galaxyID"))
        return grp_ids, gal_ids
    end

    is_sat = grp_ids .!= -1
    sat_ids = gal_ids[is_sat]
    grp_ids = grp_ids[is_sat]
    println(String("N satellites: $(length(sat_ids))"))

    return sat_ids, grp_ids
end

function get_hosts()
    fname = super_direc * 
       "global_sample_data/global_sample_data_snapshot_1200.hdf5"
    grp_ids, gal_ids = HDF5.h5open(fname) do file
        grp_ids = read(file, "groupID")
        gal_ids = Int.(read(file, "galaxyID"))
        return grp_ids, gal_ids
    end

    is_host = grp_ids .== -1
    host_ids = gal_ids[is_host]
    println(String("N hosts: $(length(host_ids))"))

    return host_ids
end

function get_both(; only_files=true)
    fname = super_direc * 
        "global_sample_data/global_sample_data_snapshot_1200.hdf5"
    grp_ids, gal_ids = HDF5.h5open(fname) do file
        grp_ids = Int.(read(file, "groupID"))
        gal_ids = Int.(read(file, "galaxyID"))
        return grp_ids, gal_ids
    end

    println(String("N hosts and satellites: $(length(gal_ids))"))

    if only_files
        potential_files = [
            "particles_within_Rvir_object_" * 
                string(id) * 
                ".hdf5"
            for id in gal_ids
        ]
        direc = joinpath(super_direc, firebox_snap)
        println("Getting list of existing files.")
        #existing_files = filter(
        #    f -> isfile(joinpath(direc, f)) && endswith(f, ".hdf5"), 
        #    readdir(direc)
        #)
        existing_files = readdir(direc)
        println("Comparing to expected files.")
        exists = [p in existing_files for p in potential_files]

        gal_ids = gal_ids[exists]
        grp_ids = grp_ids[exists]
    end

    return gal_ids, grp_ids
end

function get_bound_particles(id)
    path = joinpath(
        super_direc,
        firebox_snap,
        "bound_particle_filters_object_" * string(id) * ".hdf5"
    )
    particle_ids = HDF5.h5open(path, "r") do file
        particle_ids = Int.(read(file, "particleIDs"))
        return particle_ids
    end
    return particle_ids
end

function summarize_gals(;save=false)
    ids, grp_ids = get_both()
    summary_fields = String[]

    df = DataFrames.DataFrame(
        id=ids,
        grp_id=grp_ids,
    )
    idf = IndexedDataFrames.IndexedDataFrame(df, "id")


    for (i, (gal_id, grp_id)) in ProgressBars.ProgressBar(
                enumerate(zip(ids, grp_ids))
            )
        id_str = string(gal_id)
        fname = joinpath(
            super_direc,
            firebox_snap,
            "particles_within_Rvir_object_" * id_str * ".hdf5"
        )
        if isfile(fname)
            h5open(
                        fname, 
                        "r"
                    ) do file
                # Use the first file to determine which data fields summarize
                # the galaxy as a whole as opposed to being an array of
                # particle values.
                if i == 1
                    all_fields = keys(file)
                    for key in all_fields
                        if read(file, key) isa Number
                            push!(summary_fields, key)
                            # Make the col for the df:
                            idf[:, key] = Any[fill(nothing, length(ids))...] 
                            # Add this gal's value to the col:
                            idf[gal_id, key] = read(file, key)      
                        end
                    end
                else
                    for key in summary_fields
                        idf[gal_id, key] = read(file, key)      
                    end
                end
            end
        else
            if verbose
                println("Could not find file " * fname)
            end
            # Drop the galaxy
            deleteat!(idf, gal_id)
        end
    end
    
    if save
        CSV.write(
            joinpath(output_dir, "firebox_summary_stats.csv"),
            idf.df 
        )
    end
    
    return idf
end

end # module firebox_io 
