# uci-tools
General-purpose code expected to be useful for multiple group members.

Install in your active environment with `pip install .`.

## `config.ini`
The user should put a `config_<python_environment_name>.ini` in their home directory for certain methods that save data in this package to function properly. The user must name it according to which conda environment they plan to use: `config_<python_environment_name>.ini`. If you use the `base` environment (which you shouldn't), you can name the config file `config.ini`. It should include a uci_tools_paths section as follows
```
[uci_tools_paths]
project_data_dir = /direc/containing/data/for/analysis/that/you/personally/generated
snap_times = /path/to/snapshot/times/file
sat_image_dir = /direc/where/you/access/Courtneys/sat/images
host_iamge_dir = /direc/where/you/access/Courtneys/host/images
firebox_data_dir = /direc/containing/FIREBox/data/for/analysis
host_2d_shapes = /path/to/Courtneys/host/shape/results/file
host_2d_shapes = /path/to/Courtneys/sat/shape/results/file
```
If you don't create this and/or you don't have this section or any of its required entries before you import `uci_tools`, the package will create the missing requirements for you, and it will default to saving all outputs in `$HOME/output_<python_environment_name>/`. However, we have not yet built the functionality for the Julia portion of this repository to automatically generate its own config file. You will need to manually create it with the name `config_<JuliaProjectName>.ini`. The easiest thing to do is probably just symlink via `ln -s ~/config_<python_environment_name>.ini ~/config_<JuliaProjectName>.ini`.

## Using `refactor` script
Use `refactor` in the terminal by navigating to your project folder and then executing 

~~~
uci-tools-refactor -o old_text -n new_text
~~~
This will search all .py and .ipynb files in the project folder for `old_text`. `refactor` will give the user a preview of each instance where it finds `old_text` and ask for confirmation before replacing it with `new_text`.
