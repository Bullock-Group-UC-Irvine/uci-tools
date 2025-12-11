module UCIToolsConfig

import ConfParser
import Pkg

export read_config

function read_config()
    toml_path = Pkg.API.project().path
    proj = Pkg.TOML.parsefile(toml_path)
    config_fname = "config_" * proj["name"] * ".ini"
    config_path = expanduser(joinpath("~/", config_fname))

    # Parse the file
    conf = ConfParser.ConfParse(config_path)
    ConfParser.parse_conf!(conf)

    data = conf._data

    # Prepare a clean dict with unwrapped strings
    clean = Dict{String, Dict{String, String}}()

    # iterate through sections and keys
    for (section, kvs) in data
        secdict = Dict{String,String}()
        for (key, valvec) in kvs
            # unwrap vector (ConfParser always returns Vector{String})
            valstr = _resolve(valvec[1], data)
            secdict[key] = valstr
        end
        clean[section] = secdict
    end

    return clean
end

# Recursive helper to resolve ${section:key} references
function _resolve(val::AbstractString, conf::Dict{Any,Any})
    # Look for ${section:key} patterns
    for m in eachmatch(r"\$\{([^:}]+):([^}]+)\}", val)
        sec = m.captures[1]
        key = m.captures[2]

        # retrieve referenced value and recurse
        repl = _resolve(conf[sec][key][1], conf)  # unwrap vector

        # replace in string
        val = replace(val, m.match => repl)
    end
    return val
end

end # module
